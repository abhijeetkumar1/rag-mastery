"""Phase 5 RAG: Phase 3's plan -> retrieval from a Phase 5 index (optionally child -> parent expansion) -> injection scan
-> relevance floor -> generation over DELIMITED untrusted passages -> tolerant output guardrails. Traced as "ask_v5".

Run: uv run python -m phase5_advanced_indexing.06_ask                                  # demo questions, default index
     uv run python -m phase5_advanced_indexing.06_ask "What was Tesla's total gross margin in 2025?" --show-context
     uv run python -m phase5_advanced_indexing.06_ask --index recursive350 "..."      # same pipeline, Phase 1 chunks
     uv run python -m phase1_naive_rag.06_traces --name ask_v5

Indexes (04_index): recursive350 (Phase 1 chunks), structured350, structctx350, llmctx350, parent-child
(child200 searched, parent800 read). Guardrails, in order: router (input) -> injection scan on retrieved passages
-> relevance floor -> delimited prompt -> output checks.
"""
import argparse
import importlib
from contextlib import nullcontext

from common.chunking import count_tokens
from common.config import CHAT_MODEL, EMBED_MODEL, RERANK_MODEL
from common.guardrails import UNTRUSTED_RULES, check_answer, detect_injection, format_untrusted
from common.llm import chat_completion
from common.parent_child import expand_to_parents, load_parents
from common.query import PROMPT_VERSION, plan
from common.retriever import Retriever
from common.trace import Trace, cost_usd

phase1 = importlib.import_module("phase1_naive_rag.04_ask")
phase2 = importlib.import_module("phase2_better_retrieval.05_ask")
ask3 = importlib.import_module("phase3_query_intelligence.06_ask")

INDEXES = {  # name: (chunker, size, parent set)
    "recursive350": ("recursive", 350, None),
    "structured350": ("structured", 350, None),
    "structctx350": ("structctx", 350, None),
    "llmctx350": ("llmctx", 350, None),
    "parent-child": ("child", 200, "parent800"),
}
DEFAULT_INDEX = "structctx350"
CHILD_FACTOR = 3  # retrieve 3x as many children as parents wanted: several children often share a parent
SYSTEM = phase1.SYSTEM + "\n" + UNTRUSTED_RULES

DEMO = [
    "What was Tesla's total gross margin in 2025?",                         # Phase 3/4: segment margin 16.2% vs total 18.0%
    "How much did Amazon spend on purchases of property and equipment in 2025?",  # gross $131.8B vs net $128.3B
    "Compare the total revenue growth of Microsoft and Amazon in their latest fiscal year.",
    "What were NVIDIA's cash dividends per share in fiscal 2026?",            # Phase 4: another year's row
]


class Index:
    """A Retriever over one chunk set, plus its parents if it is a child index."""

    def __init__(self, name: str = DEFAULT_INDEX):
        chunker, size, parent_set = INDEXES[name]
        self.name = name
        self.R = Retriever(chunker, size)
        self.parents = load_parents(parent_set) if parent_set else None

    def warmup(self) -> None:
        self.R.warmup()


def open_index(spec: str) -> tuple[Index, int | None]:
    """"parent-child@3" -> (the parent-child index, k=3). The @k form compares configs at an equal TOKEN budget:
    3 parents of ~550 tokens ≈ 6 chunks of ~280."""
    name, _, k = spec.partition("@")
    return Index(name), int(k) if k else None


def retrieve(ix: Index, question: str, subs: list[dict], k: int = 6, tr=None) -> list[dict]:
    """Phase 3's per-sub-question retrieval on this index. For parent-child: retrieve CHILD_FACTOR*k children, then per
    sub-question replace children by their parents (deduplicated, also across sub-questions), k split as in Phase 3."""
    tr = tr or Trace("retrieve_only")
    if ix.parents is None:
        hits = ask3.retrieve_for_plan(ix.R, question, subs, k, tr=tr)
        for h in hits:
            h["ctx_header"] = ix.R.by_id[h["id"]].get("ctx_header")
        return hits
    children = ask3.retrieve_for_plan(ix.R, question, subs, k * CHILD_FACTOR, tr=tr)
    k_each = k if len(subs) == 1 else max(2, k // len(subs))
    with tr.span("parent_expansion", child_factor=CHILD_FACTOR) as sp:
        groups: dict[str, list[dict]] = {}
        for h in children:
            groups.setdefault(h["sub_question"], []).append(h)
        out, seen = [], set()
        for sub_q, group in groups.items():
            fresh = [h for h in group if ix.R.by_id[h["id"]]["parent_id"] not in seen]
            for p in expand_to_parents(fresh, ix.R.by_id, ix.parents, k_each):
                p["ctx_header"] = ix.parents[p["id"]]["ctx_header"]
                seen.add(p["id"])
                out.append(p)
        sp.update(n_children=len(children), n_parents=len(out), mapping=[[p["id"], p["via"]] for p in out],
                  child_tokens=sum(count_tokens(p["child_text"]) for p in out),
                  parent_tokens=sum(count_tokens(p["text"]) for p in out))
    return out


def scan(hits: list[dict], tr=None, quarantine: bool = True) -> tuple[list[dict], list[dict]]:
    """Injection scan on retrieved passages: returns (kept, quarantined). Logged per passage in the trace."""
    flagged = [(h, detect_injection(h["text"])) for h in hits]
    bad = [h for h, f in flagged if f]
    if tr:
        with tr.span("injection_scan", detector="regex", quarantine=quarantine) as sp:
            sp.update(n=len(hits), flagged=[[h["id"], f] for h, f in flagged if f])
    return ([h for h in hits if h not in bad] if quarantine else hits), bad


def generate(question: str, hits: list[dict], delimit: bool = True, tr=None) -> str:
    """delimit=False is Phase 1's prompt and formatting (for the injection ablation in 07_injection)."""
    system, ctx = (SYSTEM, format_untrusted(hits)) if delimit else (phase1.SYSTEM, phase1.format_context(hits))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"Context:\n\n{ctx}\n\nQuestion: {question}"}]
    span = tr.span if tr else (lambda *a, **kw: nullcontext({}))
    with span("generate", model=CHAT_MODEL, delimited=delimit) as sp:
        resp = chat_completion(messages)
        u = resp.usage
        sp.update(input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens,
                  cost_usd=cost_usd(CHAT_MODEL, u.prompt_tokens, u.completion_tokens))
    return resp.choices[0].message.content


def ask(ix: Index, question: str, k: int = 6, quarantine: bool = True) -> tuple[str, list[dict], dict]:
    tr = Trace("ask_v5", question=question, k=k, index=ix.name, prompt_version=PROMPT_VERSION, embed_model=EMBED_MODEL,
               rerank_model=RERANK_MODEL, chat_model=CHAT_MODEL)
    try:
        with tr.span("plan", version=PROMPT_VERSION) as sp:
            p, usage, overrides = plan(question)
            sp.update(route=p["route"], overrides=overrides, cached=usage["cached"],
                      cost_usd=cost_usd(CHAT_MODEL, usage["input_tokens"], usage["output_tokens"]),
                      sub_questions=[[s["ticker"], s["filing_years"], s["question"]] for s in p["sub_questions"]])
        if p["route"] != "answer":
            names = p["unsupported_companies"]
            msg = ask3.REFUSALS[p["route"]].format(names=", ".join(names) or "That company", verb="is" if len(names) < 2 else "are")
            tr.set(answer=msg, refused=True, refused_by=f"router:{p['route']}", guardrails_passed=True)
            return msg, [], {"refused": True, "refused_by": f"router:{p['route']}", "cited": [], "passed": True}

        hits = retrieve(ix, question, p["sub_questions"], k, tr)
        with tr.span("context", index=ix.name) as sp:  # what the LLM will see: ids, contextual headers, size
            sp.update(ids=[h["id"] for h in hits], ctx_headers=sorted({h["ctx_header"] for h in hits if h.get("ctx_header")}))
        hits, quarantined = scan(hits, tr, quarantine)

        with tr.span("relevance_floor", floor=phase2.RELEVANCE_FLOOR) as sp:
            best = max((h.get("rerank_score", float("-inf")) for h in hits), default=float("-inf"))
            sp.update(best_score=round(best, 3), blocked=best < phase2.RELEVANCE_FLOOR)
        if sp["blocked"] and RERANK_MODEL == phase2.FLOOR_MODEL:
            tr.set(answer=phase2.REFUSAL, refused=True, refused_by="relevance_floor", guardrails_passed=True)
            return phase2.REFUSAL, hits, {"refused": True, "refused_by": "relevance_floor", "best_score": best,
                                          "cited": [], "passed": True, "quarantined": [h["id"] for h in quarantined]}

        answer = generate(question, hits, delimit=True, tr=tr)
        with tr.span("guardrails", tolerant=True) as sp:
            report = check_answer(answer, [h["text"] for h in hits], tolerant=True)
            report["quarantined"] = [h["id"] for h in quarantined]
            sp.update(report)
        tr.set(answer=answer, refused=report["refused"], guardrails_passed=report["passed"])
        return answer, hits, report
    finally:
        tr.end()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--index", choices=list(INDEXES), default=DEFAULT_INDEX)
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()
    ix = Index(args.index)
    ix.warmup()
    for q in [args.question] if args.question else DEMO:
        answer, hits, report = ask(ix, q, args.k)
        print(f"Q: {q}   [index: {ix.name}]\n\n{answer}\n")
        for n, h in enumerate(hits, 1):
            m = h["meta"]
            mark = "*" if n in report.get("cited", []) else " "
            via = f" via {','.join(v.rsplit('_', 1)[-1] for v in h['via'])}" if "via" in h else ""
            print(f" {mark}[{n}] {h['score']:6.2f}  {m['ticker']} FY{m['fiscal_year']} {m['item']:8s} ({h['id']}{via})")
            if args.show_context:
                print("      " + h["text"][:400].replace("\n", "\n      ") + "...")
        if report.get("quarantined"):
            print(f"\n⚠ quarantined (injection scan): {report['quarantined']}")
        if report.get("refused_by"):
            print(f"\nguardrails: refused by {report['refused_by']}")
        elif report["passed"]:
            print("\nguardrails: ✅ citations valid, numbers grounded"
                  + (f" ({len(report['derived_numbers'])} derived)" if report.get("derived_numbers") else ""))
        else:
            print(f"\nguardrails: ⚠ invalid cites {report['invalid_citations']}, ungrounded {report['ungrounded_numbers']}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
