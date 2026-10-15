"""The production request pipeline (Phase 7): Phase 5's RAG and Phase 6's agent behind every production layer.
app.py exposes it over HTTP; the scripts call it directly.

    request ─► rate limit ─► input guardrails ─► cache ─► plan ─► access check ─► route ─┬─► pipeline: retrieve
    (app.py)   (per user)    PII redaction,      exact,   (LLM,    tickers vs     P5 or    │   (scoped) ─► ACL assert
                             moderation,         semantic  rule     the user's    agent    │   ─► injection scan ─►
                             length              (opt.)    fallback) allowlist             │   floor ─► spend check ─►
                                                                                           │   generate (stream) ─►
                                                                                           └─► agent: ReAct, steps
                                                                                               streamed, scoped tools
    ─► output guardrails ─► cache store ─► online-eval sample ─► trace (JSONL) ─► OTel export + Prometheus metrics

Everything the client sees is an EVENT, so streaming and non-streaming share one code path:
  meta        trace_id, route, cache result, degraded dependencies
  step        agent progress (tool calls, tool results): what the user watches during a 10 s agent run
  sources     the passages, numbered as cited
  token       a piece of the answer
  guardrails  the output check verdict (after the answer in "post" mode; before it in "buffer" mode)
  done        latency, cost, tokens
  error       a refusal or failure, with a code

Settings come from the environment (see Settings). LLM_BACKEND=stub replaces every LLM call with deterministic local
code (rule-based plan, extractive answer streamed with a simulated per-token delay): for load tests, CI, and for
working while the OpenAI key is unavailable. A stub run is labeled as such in traces, and its cache entries are
keyed by a different version string, so they are never served to a real-backend request.
"""
import importlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from common import telemetry
from common.access import ALL, User, load_users, scoped_filter, violations
from common.cache import ExactCache, SemanticCache
from common.config import CHAT_MODEL, DATA_DIR, EMBED_MODEL, RERANK_MODEL
from common.filters import extract_filters
from common.guardrails import CITE_RE, check_answer, format_untrusted, is_refusal
from common.moderation import moderate
from common.pii import redact
from common.query import COMPANIES, PROMPT_VERSION, catalog, expand_terms, plan
from common.ratelimit import RateLimiter, SpendLimiter, estimate_cost
from common.resilience import BREAKERS
from common.trace import Trace, cost_usd

ask5 = importlib.import_module("phase5_advanced_indexing.06_ask")
ask3 = importlib.import_module("phase3_query_intelligence.06_ask")
phase2 = importlib.import_module("phase2_better_retrieval.05_ask")

HERE = Path(__file__).parent
ONLINE_DIR = DATA_DIR / "online_eval"
FEEDBACK_DIR = DATA_DIR / "feedback"
MAX_QUESTION_CHARS = 1000
# Phase 6's routing rule (06_hybrid.AGGREGATE), applied to OUR plan so routing needs no extra LLM call
AGGREGATE = re.compile(r"\b(which of the (five|5)|rank|combined|total of|sum of|most|least|highest|lowest|fastest|"
                       r"by what percentage|how much more|grew more|partner|suppl|compet)\w*", re.I)
ADVICE = re.compile(r"\b(should i|is it a good (time|idea) to)\s+(buy|sell|invest|hold)|\bprice target\b|"
                    r"\b(buy|sell) (the )?stock\b|\bgood investment\b", re.I)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    index: str = field(default_factory=lambda: _env("PHASE7_INDEX", ask5.DEFAULT_INDEX))
    vector_db: str = field(default_factory=lambda: _env("VECTOR_DB", "chroma"))           # chroma | qdrant
    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "openai"))       # openai | stub
    stub_token_ms: float = field(default_factory=lambda: float(_env("STUB_TOKEN_MS", "15")))
    semantic_cache: bool = field(default_factory=lambda: _env("SEMANTIC_CACHE", "0") == "1")
    semantic_threshold: float = field(default_factory=lambda: float(_env("SEMANTIC_THRESHOLD", "0.95")))
    guardrail_mode: str = field(default_factory=lambda: _env("GUARDRAIL_MODE", "post"))   # post | buffer
    on_fail: str = field(default_factory=lambda: _env("GUARDRAIL_ON_FAIL", "warn"))       # warn | withhold (buffer only)
    pii_to_llm: str = field(default_factory=lambda: _env("PII_TO_LLM", "redact"))         # redact | allow
    moderation: bool = field(default_factory=lambda: _env("MODERATION", "1") == "1")
    agent: str = field(default_factory=lambda: _env("AGENT_ROUTING", "auto"))             # auto | off
    online_eval_rate: float = field(default_factory=lambda: float(_env("ONLINE_EVAL_RATE", "0.1")))
    max_output_tokens: int = field(default_factory=lambda: int(_env("MAX_OUTPUT_TOKENS", "600")))
    k: int = 6

    @property
    def version(self) -> str:
        """Part of every cache key: change any of these and old answers stop being served."""
        return f"{self.index}|{EMBED_MODEL}|{RERANK_MODEL}|{CHAT_MODEL}|plan-{PROMPT_VERSION}|{self.llm_backend}"


class Refusal(Exception):
    def __init__(self, code: str, message: str, http: int = 200):
        super().__init__(message)
        self.code, self.message, self.http = code, message, http


class ScopedRetriever:
    """The retriever as one user may use it: every search's filter is intersected with the user's tickers BEFORE it
    runs (pre-filter, inside Chroma/Qdrant and the BM25 mask). Shared by the pipeline and the agent's tools."""

    def __init__(self, R, user: User):
        self.R, self.user = R, user
        self.by_id, self.rows = R.by_id, R.rows

    def _meta(self, i):
        return self.R._meta(i)

    def retrieve(self, query: str, k: int = 5, filters=None, **kw) -> list[dict]:
        f = scoped_filter(extract_filters(query) if filters == "auto" else filters, self.user)
        return [] if f is None else self.R.retrieve(query, k=k, filters=f or None, **kw)

    def retrieve_multi(self, queries, rerank_query, k=5, n_candidates=50, filters=None, trace=None) -> list[dict]:
        f = scoped_filter(filters, self.user)
        return [] if f is None else self.R.retrieve_multi(queries, rerank_query, k, n_candidates, f or None, trace)


def rule_plan(question: str) -> tuple[dict, dict, list[str]]:
    """The planner's fallback when the LLM is unavailable (or stubbed): Phase 2's rule-based filters, one sub-question
    per named company with Phase 3's year policy and vocabulary. Same shape as common.query.plan()."""
    cat = catalog()
    if ADVICE.search(question):
        return ({"route": "investment_advice", "companies": [], "unsupported_companies": [], "sub_questions": []},
                {"input_tokens": 0, "output_tokens": 0, "cached": False}, ["rule_plan"])
    tickers = extract_filters(question).get("ticker", [])
    years = {int(y) for y in re.findall(r"\b(20\d\d)\b", question)}
    subs = []
    for t in tickers:
        want = sorted({y for x in years for y in (x, x + 1)} & set(cat[t]))
        subs.append({"question": expand_terms(f"{COMPANIES[t]}: {question}" if len(tickers) > 1 else question, t),
                     "ticker": t, "filing_years": want})
    subs = subs or [{"question": question, "ticker": None, "filing_years": []}]
    return ({"route": "answer", "companies": tickers, "unsupported_companies": [], "sub_questions": subs},
            {"input_tokens": 0, "output_tokens": 0, "cached": False}, ["rule_plan"])


_STOP = {"what", "was", "were", "the", "and", "did", "does", "how", "much", "many", "its", "for", "in", "of", "their",
         "which", "with", "from", "that", "this", "year", "fiscal", "company", "compare"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in _STOP}


def extractive_answer(question: str, hits: list[dict], max_lines: int = 3) -> str:
    """The generator's fallback: the best passage's lines that share the most words with the question, quoted and
    cited. Never invents a number (everything is copied), never synthesizes across passages."""
    if not hits:
        return phase2.REFUSAL
    qw = _words(question)
    h = max(hits, key=lambda x: x.get("rerank_score", x.get("score", 0)))
    n = hits.index(h) + 1
    lines = [ln.strip() for ln in h["text"].split("\n") if ln.strip() and not ln.startswith("[")]
    lines = [ln for ln in lines if not re.match(r"^\(\d+\)|^\*", ln)]  # footnotes
    if " | " in (lines[0] if lines else ""):  # a table: its header row, then the best-matching rows
        rows = sorted(lines[1:], key=lambda ln: -len(qw & _words(ln)))[: max_lines - 1]
        best = [lines[0]] + [ln for ln in lines[1:] if ln in rows]
    else:
        sents = [s for ln in lines for s in re.split(r"(?<=[.;])\s+(?=[A-Z])", ln)]
        top = sorted(sents, key=lambda s: -(len(qw & _words(s)) + 0.5 * bool(re.search(r"\d", s))))[:max_lines - 1]
        best = [s for s in sents if s in top]  # keep document order
    m = h["meta"]
    quote = "\n".join(f"> {ln}" for ln in best)
    return (f"(Generated answer unavailable: showing the most relevant passage from {m['ticker']} 10-K "
            f"FY{m['fiscal_year']}, {m['item']}.)\n\n{quote}\n\n[{n}]")


class RAGService:
    def __init__(self, settings: Settings | None = None, users_path: Path = HERE / "users.json"):
        self.s = settings or Settings()
        self.ix = ask5.Index(self.s.index)
        self.R = self.ix.R
        if self.ix.parents is not None:
            raise SystemExit("Phase 7 serves flat indexes (structctx350 by default), not parent-child")
        if self.s.vector_db == "qdrant":
            from common.store import collection_name
            from common.vectordb import QdrantCollection
            chunker, size, _ = ask5.INDEXES[self.s.index]
            self.R.col = QdrantCollection(collection_name(chunker, size))
            if self.R.col.count() != len(self.R.rows):
                raise SystemExit("Qdrant collection missing or stale: run phase7_production.05_vector_db --migrate")
        self.R.degrade_dense = True
        import common.llm
        common.llm.embed_breaker = BREAKERS["embed"]  # wraps only the embedding API call; cached embeddings still work
        self.users = load_users(users_path)
        self.exact = ExactCache(self.s.version)
        self.semantic = SemanticCache(self.s.version, self.s.semantic_threshold) if self.s.semantic_cache else None
        self.rate, self.spend = RateLimiter(), SpendLimiter()
        self._graph = None
        self._io_lock = threading.Lock()
        if self.s.llm_backend == "openai":
            from common.llm import configure_client
            try:
                configure_client(max_retries=1, timeout_s=20)  # fail fast: the fallback is better than a minute of retries
            except SystemExit:
                pass  # no key: every LLM call fails over to its fallback
        telemetry.setup_otel()

    def warmup(self) -> None:
        self.R.warmup()
        if self.semantic:
            self.semantic._vec("warm up")

    def user(self, api_key: str | None) -> User | None:
        return self.users.get(api_key or "")

    def admit(self, user: User) -> float:
        """0.0, or seconds to wait (rate limited). Called by the app BEFORE streaming starts, so it can return 429."""
        return self.rate.check(user.id, user.rpm)

    # ------------------------------------------------------------------ the pipeline as a stream of events
    def stream(self, question: str, user: User, mode: str = "auto"):
        t0 = time.perf_counter()
        q_red, pii_types = redact(question)
        tr = Trace("ask_v7", question=q_red, user=user.id, scope=",".join(user.scope), index=self.s.index,
                   vector_db=self.s.vector_db, llm_backend=self.s.llm_backend, prompt_version=PROMPT_VERSION,
                   chat_model=CHAT_MODEL, guardrail_mode=self.s.guardrail_mode)
        ctx = SimpleNamespace(tr=tr, degraded=[], cost=0.0, route="pipeline", cache="miss", ttft=None, t0=t0)
        try:
            q_llm = q_red if self.s.pii_to_llm == "redact" else question
            self._input_guardrails(question, q_red, pii_types, ctx)
            cached = self._cache_lookup(q_red, user, ctx)
            if cached:
                yield from self._replay(cached, ctx)
                return
            p = self._plan(q_llm, ctx)
            self._access_check(p, user, ctx)
            ctx.route = self._route(p, question, mode, ctx)
            yield {"event": "meta", "data": {"trace_id": tr.id, "route": ctx.route, "cache": "miss",
                                             "degraded": list(ctx.degraded), "pii_redacted": pii_types}}
            if ctx.route == "agent":
                result = yield from self._agent(q_llm, user, ctx)
            else:
                result = yield from self._pipeline(q_llm, p, user, ctx)
            answer, hits, report = result
            self._after(q_red, user, answer, hits, report, ctx)
            yield self._done(ctx, report)
        except Refusal as r:
            tr.set(outcome=f"refused:{r.code}", refused=True, refused_by=r.code, answer=r.message)
            yield {"event": "error", "data": {"trace_id": tr.id, "code": r.code, "message": r.message, "http": r.http}}
        except Exception as e:  # noqa: BLE001: the client gets a clean error event, the trace gets the details
            tr.set(outcome="error", error=f"{type(e).__name__}: {e}")
            yield {"event": "error", "data": {"trace_id": tr.id, "code": "internal_error", "message": "Internal error", "http": 500}}
        finally:
            tr.set(route=ctx.route, cache=ctx.cache, degraded=ctx.degraded, cost_usd=round(ctx.cost, 6),
                   ttft_ms=ctx.ttft)
            record = tr.end()
            telemetry.export_trace(record)
            telemetry.record_metrics(record)
            if ctx.ttft is not None:
                telemetry.TTFT.observe(ctx.ttft / 1000)

    def ask(self, question: str, user: User, mode: str = "auto") -> dict:
        """Non-streaming: the same events, collected."""
        out = {"answer": "", "steps": []}
        for ev in self.stream(question, user, mode):
            d = ev["data"]
            if ev["event"] == "token":
                out["answer"] += d["text"]
            elif ev["event"] == "step":
                out["steps"].append(d)
            else:
                out[ev["event"]] = d
        return out

    # ------------------------------------------------------------------ stages
    def _input_guardrails(self, question: str, q_red: str, pii_types: list, ctx) -> None:
        with ctx.tr.span("input_guardrails") as sp:
            sp.update(chars=len(question), pii=pii_types)
            if len(question) > MAX_QUESTION_CHARS:
                raise Refusal("too_long", f"Questions are limited to {MAX_QUESTION_CHARS} characters.", 422)
            if self.s.moderation and self.s.llm_backend != "stub":
                m = moderate(q_red)
                sp.update(moderation=m)
                if m["source"] == "unavailable":
                    ctx.degraded.append("moderation")
                if m["flagged"]:
                    telemetry.GUARDRAIL.labels("moderation", "flagged").inc()
                    raise Refusal("moderation", "This request can't be processed.")
        if pii_types:
            telemetry.GUARDRAIL.labels("pii_input", "redacted").inc()

    def _cache_lookup(self, q_red: str, user: User, ctx):
        with ctx.tr.span("cache_lookup", semantic=self.semantic is not None) as sp:
            hit = self.exact.get(q_red, user.scope)
            if hit:
                ctx.cache = sp["result"] = "exact"
            elif self.semantic:
                hit, sim, matched = self.semantic.lookup(q_red, user.scope)
                sp.update(best_similarity=round(sim, 4), matched=matched)
                if hit:
                    ctx.cache = sp["result"] = "semantic"
            sp.setdefault("result", "miss")
        telemetry.CACHE.labels("exact", "hit" if ctx.cache == "exact" else "miss").inc()
        if self.semantic and ctx.cache != "exact":
            telemetry.CACHE.labels("semantic", "hit" if ctx.cache == "semantic" else "miss").inc()
        return hit

    def _replay(self, cached: dict, ctx):
        ctx.route = cached["route"]
        ctx.tr.set(outcome="answered", answer=redact(cached["answer"])[0], cached_from=cached["trace_id"],
                   guardrails_passed=cached["guardrails"]["passed"], cited=cached["guardrails"].get("cited"))
        yield {"event": "meta", "data": {"trace_id": ctx.tr.id, "route": cached["route"], "cache": ctx.cache,
                                         "cached_from": cached["trace_id"], "degraded": []}}
        yield {"event": "sources", "data": {"sources": cached["sources"]}}
        ctx.ttft = round((time.perf_counter() - ctx.t0) * 1000, 1)
        yield {"event": "token", "data": {"text": cached["answer"]}}
        yield {"event": "guardrails", "data": cached["guardrails"]}
        yield self._done(ctx, cached["guardrails"])

    def _plan(self, q_llm: str, ctx) -> dict:
        with ctx.tr.span("plan", version=PROMPT_VERSION) as sp:
            if self.s.llm_backend == "stub":
                (p, usage, overrides), degraded = rule_plan(q_llm), False
            else:
                (p, usage, overrides), degraded = BREAKERS["plan"].call(plan, rule_plan, q_llm)
            c = cost_usd(CHAT_MODEL, usage["input_tokens"], usage["output_tokens"]) or 0.0
            ctx.cost += c
            sp.update(route=p["route"], overrides=overrides, cached=usage["cached"], cost_usd=c, fallback=degraded,
                      input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
                      sub_questions=[[s["ticker"], s["filing_years"], s["question"]] for s in p["sub_questions"]])
        if degraded:
            ctx.degraded.append("plan")
        if p["route"] != "answer":
            names = p["unsupported_companies"]
            raise Refusal(f"router:{p['route']}", ask3.REFUSALS[p["route"]].format(
                names=", ".join(names) or "That company", verb="is" if len(names) < 2 else "are"))
        return p

    def _access_check(self, p: dict, user: User, ctx) -> None:
        denied = sorted({s["ticker"] for s in p["sub_questions"] if s["ticker"] and s["ticker"] not in user.tickers})
        with ctx.tr.span("access_check", allowed=list(user.scope)) as sp:
            sp["denied"] = denied
        if denied:
            telemetry.GUARDRAIL.labels("access", "denied").inc()
            raise Refusal("access_denied", f"Your account doesn't have access to the filings of {', '.join(denied)}.", 403)

    def _route(self, p: dict, question: str, mode: str, ctx) -> str:
        if mode in ("pipeline", "agent"):
            want = mode
        elif self.s.agent == "off":
            want = "pipeline"
        else:
            want = "agent" if len(p["sub_questions"]) >= 3 or AGGREGATE.search(question) else "pipeline"
        if want == "agent" and (self.s.llm_backend == "stub" or not BREAKERS["generate"].allow()):
            ctx.degraded.append("agent")  # the agent needs the LLM at every step: no fallback inside it
            return "pipeline"
        return want

    def _pipeline(self, q_llm: str, p: dict, user: User, ctx):
        tr = ctx.tr
        scoped = SimpleNamespace(R=ScopedRetriever(self.R, user), parents=None, name=self.s.index)
        hits = ask5.retrieve(scoped, q_llm, p["sub_questions"], self.s.k, tr)
        if any(s["attrs"].get("degraded") for s in tr.spans if s["name"] == "dense_search"):
            ctx.degraded.append("embed")
        hits = self._assert_access(hits, user, ctx)
        hits, quarantined = ask5.scan(hits, tr)
        with tr.span("relevance_floor", floor=phase2.RELEVANCE_FLOOR) as sp:
            best = max((h.get("rerank_score", float("-inf")) for h in hits), default=float("-inf"))
            sp.update(best_score=round(best, 3), blocked=best < phase2.RELEVANCE_FLOOR)
        if not hits or (sp["blocked"] and RERANK_MODEL == phase2.FLOOR_MODEL):
            raise Refusal("relevance_floor", phase2.REFUSAL)
        yield {"event": "sources", "data": {"sources": [self._source(n, h) for n, h in enumerate(hits, 1)]}}

        messages = [{"role": "system", "content": ask5.SYSTEM},
                    {"role": "user", "content": f"Context:\n\n{format_untrusted(hits)}\n\nQuestion: {q_llm}"}]
        tokens = []
        buffer = self.s.guardrail_mode == "buffer"
        for piece in self._generate(messages, q_llm, hits, user, ctx):
            tokens.append(piece)
            if not buffer:
                if ctx.ttft is None:
                    ctx.ttft = round((time.perf_counter() - ctx.t0) * 1000, 1)
                yield {"event": "token", "data": {"text": piece}}
        answer = "".join(tokens)
        report = self._output_guardrails(answer, hits, ctx)
        report["quarantined"] = [h["id"] for h in quarantined]
        if buffer:
            if not report["passed"] and self.s.on_fail == "withhold":
                answer = ("I found relevant passages but couldn't verify every number in the drafted answer against "
                          "them, so I'm not showing it. The sources are listed above.")
                report["withheld"] = True
            ctx.ttft = round((time.perf_counter() - ctx.t0) * 1000, 1)
            yield {"event": "guardrails", "data": report}
            yield {"event": "token", "data": {"text": answer}}
        else:
            yield {"event": "guardrails", "data": report}
        return answer, hits, report

    def _assert_access(self, hits: list[dict], user: User, ctx) -> list[dict]:
        bad = violations(hits, user)
        if bad:  # the pre-filter failed somewhere: drop and record, never serve
            telemetry.GUARDRAIL.labels("access_violation", "dropped").inc()
            with ctx.tr.span("access_violation") as sp:
                sp["dropped"] = bad
        return [h for h in hits if h["id"] not in bad]

    def _generate(self, messages: list[dict], question: str, hits: list[dict], user: User, ctx):
        """Yields answer pieces. Pre-call spend check, streaming OpenAI call, fallback to the extractive answer."""
        tr = ctx.tr
        est_tokens, est_cost = estimate_cost(CHAT_MODEL, messages, self.s.max_output_tokens)
        stub = self.s.llm_backend == "stub"
        with tr.span("generate", model="stub" if stub else CHAT_MODEL, streamed=True) as sp:
            sp.update(estimated_input_tokens=est_tokens, estimated_cost_usd=round(est_cost, 6))
            if stub:
                for w in re.findall(r"\S+\s*", extractive_answer(question, hits)):
                    time.sleep(self.s.stub_token_ms / 1000)
                    yield w
                sp["backend"] = "stub"
                return
            if not self.spend.reserve(user.id, user.daily_usd, est_cost):
                sp["blocked"] = "daily_spend_limit"
                raise Refusal("spend_limit", f"Daily usage limit reached (${user.daily_usd:.2f}). Try again tomorrow.", 429)
            usage, started, actual = None, False, 0.0
            try:
                if not BREAKERS["generate"].allow():
                    raise RuntimeError("breaker open")
                from common.llm import client
                stream = client().chat.completions.create(
                    model=CHAT_MODEL, messages=messages, temperature=0.0, max_tokens=self.s.max_output_tokens,
                    stream=True, stream_options={"include_usage": True})
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        started = True
                        yield chunk.choices[0].delta.content
                    if chunk.usage:
                        usage = chunk.usage
                BREAKERS["generate"].success()
            except Exception as e:  # noqa: BLE001
                if str(e) != "breaker open":
                    BREAKERS["generate"].failure(e)
                sp["error_fallback"] = f"{type(e).__name__}: {str(e)[:200]}"
                ctx.degraded.append("generate")
                if started:
                    yield "\n\n[The answer was interrupted.]"
                else:
                    yield extractive_answer(question, hits)
            finally:
                if usage:
                    actual = cost_usd(CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens) or 0.0
                    sp.update(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, cost_usd=actual)
                self.spend.settle(user.id, est_cost, actual)
                ctx.cost += actual

    def _output_guardrails(self, answer: str, hits: list[dict], ctx) -> dict:
        with ctx.tr.span("guardrails", tolerant=True, mode=self.s.guardrail_mode) as sp:
            report = check_answer(answer, [h["text"] for h in hits], tolerant=True)
            _, answer_pii = redact(answer)
            source_pii = {k for h in hits for k in redact(h["text"])[1]}
            report["pii_not_in_sources"] = sorted(set(answer_pii) - source_pii)
            sp.update(report)
        telemetry.GUARDRAIL.labels("output", "passed" if report["passed"] else "failed").inc()
        return report

    # ------------------------------------------------------------------ agent route
    def _agent(self, q_llm: str, user: User, ctx):
        from langchain_core.messages import AIMessage, HumanMessage

        from common.agent_limits import AgentLimits, Budget
        from common.agent_tools import Evidence, make_tools
        from common.react_agent import build_graph
        tr, limits = ctx.tr, AgentLimits()
        self._graph = self._graph or build_graph()
        ev, budget = Evidence(), Budget(limits, CHAT_MODEL)
        tools = make_tools(ScopedRetriever(self.R, user), ev, tr, allowed_tickers=set(user.tickers))
        run = {"tools": tools, "allowed": sorted(limits.allowed_tools & set(tools)), "budget": budget, "trace": tr, "evidence": ev}
        state = {"messages": [HumanMessage(q_llm)], "stop_reason": None, "checked": False}
        answer, stop_reason = "", None
        with tr.span("agent", max_steps=limits.max_steps) as asp:
            for update in self._graph.stream(state, config={"configurable": {"run": run}, "recursion_limit": 4 * limits.max_steps + 4},
                                             stream_mode="updates"):
                for node, out in update.items():
                    out = out or {}
                    msgs = out.get("messages", [])
                    stop_reason = out.get("stop_reason", stop_reason)
                    if node == "agent" and msgs and msgs[-1].tool_calls:
                        yield {"event": "step", "data": {"step": budget.steps, "node": "agent",
                                                         "tool_calls": [{"name": c["name"], "args": c["args"]} for c in msgs[-1].tool_calls]}}
                    elif node == "tools":
                        yield {"event": "step", "data": {"step": budget.steps, "node": "tools", "results": len(msgs),
                                                         "evidence": len(ev.hits)}}
                    elif node in ("check", "finalize"):
                        yield {"event": "step", "data": {"step": budget.steps, "node": node,
                                                         **({"stop_reason": stop_reason} if node == "finalize" else {})}}
                    for m in msgs:
                        if isinstance(m, AIMessage) and not m.tool_calls:
                            answer = m.content
            asp.update(**budget.summary(), stop_reason=stop_reason)
        ctx.cost += budget.cost
        bad = violations(ev.hits, user)
        if bad:  # evidence entered the agent's context: the answer may use it, so it is withheld, not just filtered
            telemetry.GUARDRAIL.labels("access_violation", "withheld").inc()
            with tr.span("access_violation") as sp:
                sp["dropped"] = bad
            raise Refusal("access_violation", "This answer can't be shown with your access level.", 403)
        yield {"event": "sources", "data": {"sources": [self._source(n, h) for n, h in enumerate(ev.hits, 1)]}}
        ctx.ttft = round((time.perf_counter() - ctx.t0) * 1000, 1)
        yield {"event": "token", "data": {"text": answer}}
        report = self._output_guardrails(answer, ev.hits, ctx)
        report["stop_reason"] = stop_reason
        yield {"event": "guardrails", "data": report}
        return answer, ev.hits, report

    # ------------------------------------------------------------------ after the answer
    def _after(self, q_red: str, user: User, answer: str, hits: list[dict], report: dict, ctx) -> None:
        refused = is_refusal(answer)
        ctx.tr.set(outcome="refused:generator" if refused else "answered", answer=redact(answer)[0], refused=refused,
                   guardrails_passed=report["passed"], cited=report.get("cited"),
                   best_rerank=max((h.get("rerank_score", float("-inf")) for h in hits), default=None))
        # stub answers are cacheable too: Settings.version contains the backend, so they never mix with real answers
        cacheable = report["passed"] and not ctx.degraded and not refused and not report.get("withheld")
        if cacheable:
            value = {"answer": answer, "sources": [self._source(n, h) for n, h in enumerate(hits, 1)],
                     "guardrails": report, "trace_id": ctx.tr.id, "route": ctx.route}
            self.exact.put(q_red, user.scope, value)
            if self.semantic:
                self.semantic.put(q_red, user.scope, value)
        if not refused and random.random() < self.s.online_eval_rate:
            cited = sorted({int(n) for n in CITE_RE.findall(answer) if 1 <= int(n) <= len(hits)})
            self._append(ONLINE_DIR / "queue.jsonl", {
                "trace_id": ctx.tr.id, "question": q_red, "answer": redact(answer)[0], "route": ctx.route,
                "degraded": ctx.degraded, "passages": [_source_line(h) for h in ([hits[n - 1] for n in cited] or hits)]})
            ctx.tr.set(online_eval_sampled=True)

    def _done(self, ctx, report: dict) -> dict:
        return {"event": "done", "data": {"trace_id": ctx.tr.id, "latency_ms": round((time.perf_counter() - ctx.t0) * 1000, 1),
                                          "ttft_ms": ctx.ttft, "cost_usd": round(ctx.cost, 6), "route": ctx.route,
                                          "cache": ctx.cache, "degraded": ctx.degraded, "guardrails_passed": report.get("passed")}}

    @staticmethod
    def _source(n: int, h: dict) -> dict:
        m = h["meta"]
        return {"n": n, "id": h["id"], "ticker": m["ticker"], "fiscal_year": m["fiscal_year"], "item": m["item"],
                "section": m["section"], "url": m["url"], "score": round(h.get("score", 0.0), 3)}

    def feedback(self, trace_id: str, user: User, rating: str, comment: str = "") -> None:
        telemetry.FEEDBACK.labels(rating).inc()
        self._append(FEEDBACK_DIR / "feedback.jsonl", {"trace_id": trace_id, "user": user.id, "rating": rating,
                                                       "comment": redact(comment)[0][:2000],
                                                       "at": time.strftime("%Y-%m-%dT%H:%M:%S")})

    def _append(self, path: Path, row: dict) -> None:
        with self._io_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")

    def health(self) -> dict:
        return {"index": self.s.index, "chunks": len(self.R.rows), "vector_db": self.s.vector_db,
                "llm_backend": self.s.llm_backend, "breakers": {k: b.status() for k, b in BREAKERS.items()},
                "cache_items": len(self.exact.store.items), "users": len(self.users), "all_tickers": list(ALL)}


def _source_line(h: dict) -> str:
    m = h["meta"]
    return f"{m['ticker']} 10-K FY{m['fiscal_year']}, {m['item']}: {m['section']}\n{h['text']}"
