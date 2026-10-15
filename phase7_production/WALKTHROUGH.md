# Phase 7 walkthrough: production

Phases 1–6 answer questions from a script. Phase 7 puts the same RAG (Phase 5's pipeline, Phase 6's agent) behind a **service**:
- an HTTP API that streams answers
- authentication, per-user rate limits and spend limits
- per-user access control inside retrieval
- PII redaction, moderation, caching
- planned fallbacks for every remote dependency
- OpenTelemetry traces, Prometheus metrics, a dashboard, online evaluation and user feedback

It finishes with a load test and a system-design mock ("RAG for 10M documents", in `NOTES.md` §4).

> **The OpenAI key was invalid for this whole phase** (`401 expired_secret_key`, later `token_invalidated`). So everything here was built and measured **without an LLM**. The service runs on its fallbacks (rule-based plan, BM25 when a query embedding isn't cached, an extractive answer), or on `LLM_BACKEND=stub`, a deterministic stand-in that streams the extractive answer at LLM speed. That is the first production lesson: the dependency you call on every request *will* be unavailable some day, and the service has to keep answering. With a working key, the same code calls OpenAI; nothing else changes.

Time to read: about 2 hours. Every number below comes from runs on 2026-10-15.

---

## 0. Architecture

### 0a. The request path

```
                       ┌──────────────────────────── phase7_production/app.py (FastAPI) ─────────────────────────────┐
 client ──HTTP──►  auth: X-API-Key → User (tickers, rpm, $/day)  ──►  rate limit (token bucket) ──► 429 + Retry-After
 (02_client)                                                       │
                                                                   ▼
                       ┌────────────────────────── phase7_production/service.py (RAGService.stream) ─────────────────┐
                       │ input guardrails: length · PII redaction (common/pii.py) · moderation (common/moderation.py) │
                       │ cache lookup: exact (common/cache.py) [+ semantic, off by default]  ──hit──► replay events   │
                       │ plan: common.query.plan (LLM)  ──breaker/fallback──►  rule_plan                              │
                       │ access check: named companies ⊆ user's tickers, else 403 event                               │
                       │ route: pipeline | agent (Phase 6 rule; agent needs the LLM, else falls back to pipeline)      │
                       │                                                                                              │
                       │  pipeline                                        agent                                       │
                       │  ScopedRetriever (pre-filter by user's tickers)  ReAct graph (Phase 6), tools on the         │
                       │  └─ Retriever: dense ──(embed fails)──► skip     ScopedRetriever + graph tool filter,        │
                       │      + BM25 → RRF → rerank (lock)                steps streamed as "step" events             │
                       │  ACL assert (drop) → injection scan → floor      ACL assert (withhold the answer)            │
                       │  spend check (pre-call estimate) → generate      → output guardrails                         │
                       │  (OpenAI stream | stub | extractive fallback)                                                │
                       │  → output guardrails (post | buffer mode)                                                    │
                       │                                                                                              │
                       │ cache store (only clean answers) → online-eval sample → Trace.end()                          │
                       └──────────────┬──────────────────────────────────┬────────────────────────────────────────────┘
                                      ▼                                  ▼
               data/traces/*.jsonl (ask_v7)            common/telemetry.py: OTel spans → Phoenix (OTLP) + data/otel/spans.jsonl
               data/online_eval/queue.jsonl                                Prometheus counters/histograms → GET /metrics
               data/feedback/feedback.jsonl
                                      ▼
               07_dashboard.py → data/dashboard.html          08_online_eval.py → data/online_eval/scores.jsonl
```

### 0b. Components

| File | What it is | New / changed |
|---|---|---|
| `phase7_production/service.py` | `RAGService`: the request pipeline as a stream of events; `Settings`; `ScopedRetriever`; `rule_plan`; `extractive_answer` | new |
| `phase7_production/app.py` | FastAPI app: `/health`, `/ready`, `/ask`, `/ask/stream` (SSE), `/feedback`, `/metrics`; auth and rate-limit dependencies | new |
| `phase7_production/users.json` | demo API keys → users (ticker allowlist, rpm, $/day) | new |
| `common/resilience.py` | `CircuitBreaker`, one per dependency (`BREAKERS`) | new |
| `common/cache.py` | `normalize`, `signature`, `ExactCache`, `SemanticCache` | new |
| `common/pii.py` | `find_pii`, `redact` (regex + Luhn + SSN rules) | new |
| `common/ratelimit.py` | `TokenBucket`, `RateLimiter`, `SpendLimiter`, `estimate_cost` | new |
| `common/access.py` | `User`, `load_users`, `scoped_filter`, `violations` | new |
| `common/moderation.py` | `moderate` (OpenAI moderation, fail open/closed) | new |
| `common/vectordb.py` | `QdrantCollection` (a Chroma-shaped adapter), `migrate_from_chroma` | new |
| `common/telemetry.py` | OTel setup, the Trace → OTel span-tree bridge, Prometheus metrics | new |
| `common/retriever.py` | `degrade_dense` + `_dense_or_skip`: an embedding failure → BM25 only | changed (off by default) |
| `common/llm.py` | `embed_breaker` around the embedding API call only; `configure_client` (retries/timeout); a lock around local HF encoding | changed |
| `common/rerank.py` | a lock around `predict()` (MPS concurrency crash) | changed |
| `common/agent_tools.py` | `make_tools(..., allowed_tickers=)`: the graph tool filters its edges | changed (default unchanged) |
| `common/config.py` | `OPENAI_MAX_RETRIES`, `OPENAI_TIMEOUT_S` | changed |

The changes to shared code are all opt-in or behavior-preserving. Phase 2's probe evaluation reproduces exactly after them (dense 8/12, BM25 8/12, RRF 7/12, rerank 10/12: the numbers in Phase 2's walkthrough).

### 0c. Data model: how a request changes shape

| Stage | Record | Fields |
|---|---|---|
| HTTP in | `AskIn` (pydantic) | `question` (3–1000 chars), `mode` (`auto`/`pipeline`/`agent`) |
| auth | `User` (frozen dataclass) | `id`, `tickers` (allowlist), `rpm`, `daily_usd`; `.scope` = sorted tickers |
| input guardrails | `q_red`, `pii_types` | the question with `[EMAIL]`/`[SSN]`/… substituted; the types found (never the values) |
| plan | Phase 3 plan dict | `route`, `companies`, `sub_questions[{question, ticker, filing_years}]`; `overrides` contains `"rule_plan"` when the fallback produced it |
| retrieval | hit dict (Phase 2–5) | `id`, `text`, `meta{ticker, fiscal_year, item, section, url}`, `rrf_score`, `rerank_score`, `score`, `sub_question`, `ctx_header` |
| out | event | `{"event": meta\|step\|sources\|token\|guardrails\|done\|error, "data": {...}}`; over SSE: `event: token` / `data: {"text": "..."}` |
| cache | value | `answer`, `sources`, `guardrails`, `trace_id` (where it came from), `route` |
| trace (`ask_v7`) | JSONL line | attrs `question` (redacted), `user`, `scope`, `index`, `vector_db`, `llm_backend`, `route`, `cache`, `degraded[]`, `outcome`, `cost_usd`, `ttft_ms`, `best_rerank`, `guardrails_passed`, `online_eval_sampled`; spans as in Phases 1–6 plus `input_guardrails`, `cache_lookup`, `access_check`, `access_violation`, `agent` |
| OTel | spans | root `rag.request` with the trace attrs; one child per span, **nested** (agent steps under `agent`), `openinference.span.kind`, `gen_ai.usage.*` |
| online eval | queue row | `trace_id`, `question` and `answer` (redacted), `route`, `degraded`, `passages` (the cited ones, with source lines) |
| feedback | row | `trace_id`, `user`, `rating` (`up`/`down`), `comment` (redacted), `at` |

---

## Step 1. The service pipeline (`service.py`)

Nothing to run on its own; every later step uses it. Read it top to bottom once.

### Settings (`service.py:71`)
Every behavior is an environment variable with a default, so the same code serves production and tests:

| Variable | Default | Meaning |
|---|---|---|
| `PHASE7_INDEX` | `structctx350` | Phase 5's winning index (flat chunks + contextual header) |
| `VECTOR_DB` | `chroma` | or `qdrant` (Step 5) |
| `LLM_BACKEND` | `openai` | `stub`: rule plan + extractive answer streamed at `STUB_TOKEN_MS` (15) per word |
| `SEMANTIC_CACHE` / `SEMANTIC_THRESHOLD` | `0` / `0.95` | off, because of Step 3 |
| `GUARDRAIL_MODE` | `post` | `buffer`: hold the answer until the output checks ran |
| `GUARDRAIL_ON_FAIL` | `warn` | `withhold` (buffer mode only): replace a failing answer with a notice |
| `PII_TO_LLM` | `redact` | `allow`: send the raw question to the LLM (logs are always redacted) |
| `MODERATION` | `1` | call the moderation API (skipped in stub mode) |
| `AGENT_ROUTING` | `auto` | `off`: never route to the agent |
| `ONLINE_EVAL_RATE` | `0.1` | share of answered requests queued for judging |
| `MAX_OUTPUT_TOKENS` | `600` | the generation's `max_tokens`, and the output side of the pre-call estimate |

`Settings.version` (`:88`) = index | embedding model | reranker | chat model | prompt version | backend. It goes into every cache key, so changing any of them stops old answers being served.

### `stream()` (`service.py:213`), the whole request as events
1. Redact the question (`redact`), open a `Trace("ask_v7")` with the **redacted** question. The trace file must never hold raw PII.
2. `_input_guardrails` (`:268`): length limit (a `Refusal` with http 422), moderation (`degraded += "moderation"` if the API is down; fail open).
3. `_cache_lookup` (`:284`): exact cache on (redacted question, scope), then the semantic cache if enabled. A hit → `_replay` (`:300`) sends the stored events and the request ends in ~1 ms.
4. `_plan` (`:311`): `BREAKERS["plan"].call(plan, rule_plan, q)`. The LLM planner, or the rule-based plan if it fails. A non-`answer` route → a `Refusal` with Phase 3's message.
5. `_access_check` (`:330`): companies in the plan that the user may not see → `Refusal("access_denied", http 403)`. **We refuse the whole question**, even a comparison where one company is allowed: answering half of it would reveal that the other half exists.
6. `_route` (`:338`): Phase 6's rule (≥ 3 sub-questions, or wording like "rank", "which of the five", "partner") → `agent`. But the agent needs the LLM at every step, so with the stub or an open `generate` breaker it falls back to the pipeline (`degraded += "agent"`).
7. `_pipeline` (`:350`) or `_agent` (`:456`); both yield events and return `(answer, hits, report)`.
8. `_after` (`:505`): trace attributes, cache store, online-eval sampling.
9. `finally`: `tr.end()` → `telemetry.export_trace` + `record_metrics`. This runs even when the client disconnects (the generator is closed) or the request errors.

`Refusal` (`:93`) carries a `code`, a user-facing message and an HTTP status. Any other exception becomes `{"code": "internal_error"}` with the details in the trace only: **never send stack traces to clients.**

### `_pipeline()` (`service.py:350`)
- `ScopedRetriever` (`:99`) wraps the shared `Retriever`. Its `retrieve` and `retrieve_multi` intersect every filter with the user's tickers (`access.scoped_filter`) **before** the search runs. Phase 5's `retrieve()` and Phase 3's `retrieve_for_plan()` run unchanged on it, because they only call those two methods.
- After retrieval: if any `dense_search` span recorded `degraded` (the embedding failed and BM25 ran alone), the request gets `degraded += "embed"`.
- `_assert_access` (`:390`): any hit outside the user's tickers is dropped and logged as an `access_violation` span. With a correct pre-filter this never fires (Step 4 simulates a bug).
- Then Phase 5's injection scan and relevance floor, then `sources`, then generation.

### `_generate()` (`service.py:398`)
- **Pre-call spend check.** `estimate_cost` counts the prompt tokens exactly and adds `max_tokens` as the output bound. `SpendLimiter.reserve` refuses (http 429) if the estimate would cross the user's daily limit. `finally: settle` swaps the reservation for the actual cost.
- **Streaming.** `stream=True, stream_options={"include_usage": True}` (OpenAI sends the token usage in the last chunk; without the option a streamed call reports no usage and the cost would be unknown).
- **Fallback.** Any failure before the first token yields `extractive_answer(...)` and `degraded += "generate"`. A failure mid-stream appends "[The answer was interrupted.]": the text already sent can't be taken back.
- The client uses `max_retries=1, timeout=20 s` (`configure_client`, set in `__init__`). Evaluation keeps 8 retries; a user waiting for an answer shouldn't.

### `extractive_answer()` (`service.py:146`)
The best passage by rerank score; from it, the table's header row plus the rows sharing the most content words with the question, or the best two sentences of prose. Footnotes are skipped. The answer is quoted and cited `[n]`. It can't invent a number (everything is copied), so the numeric guardrail always passes. It also can't answer a comparison or do arithmetic. It's labeled "(Generated answer unavailable: …)".

### `rule_plan()` (`service.py:119`)
Phase 2's regex filters (company aliases) + Phase 3's year policy (year X → filings X and X+1) + Phase 3's vocabulary expansion, one sub-question per named company, and a small investment-advice regex. **Its known gap:** it can't recognize a company we *don't* cover. "What was Google's advertising revenue?" gets no ticker, so it searches everything and the floor (uncalibrated, Phase 5) doesn't stop it. The LLM planner refuses it. Degraded mode is less safe, so it must be visible on the dashboard.

### `_agent()` (`service.py:456`)
Phase 6's ReAct graph, run with `graph.stream(stream_mode="updates")`: LangGraph yields `{node_name: node_output}` after every node. That becomes `step` events: the tool calls the agent chose, how many results came back, the check and finalize nodes. The tools are built on the `ScopedRetriever` **and** with `allowed_tickers` (Step 4 shows why both are needed). After the run, `violations(ev.hits)` is checked: if foreign evidence got in, the answer is **withheld** (`Refusal("access_violation", 403)`), because the agent has already read it.

### `_output_guardrails()` (`service.py:445`)
Phase 3's tolerant `check_answer` (citations valid, numbers grounded or derivable) plus a new check, `pii_not_in_sources`: PII types in the answer that appear in none of the passages. Those would have come from the user or the model, not the filings.

> ### 📘 Deep dive: events as the service's single interface
> **What:** the pipeline is a Python generator that yields events; `/ask/stream` forwards them as SSE, and `/ask` collects them into one JSON (`RAGService.ask`, `:254`).
> **Why:** there's one code path for streaming and non-streaming, so the two can't drift apart. Tests and scripts (`09_agent_stream`) consume the same events as the browser.
> **How errors work:** once the first byte is sent, the HTTP status is fixed at 200, so a refusal later in the stream (access denied after planning, a spend limit before generation) arrives as an `error` event with an `http` field. `/ask` maps it back to a real status code.
> **Gotcha:** a generator's `finally` runs only when the generator is closed or exhausted. `app.py` closes it explicitly when the client disconnects, so the trace still ends and the upstream LLM stream is closed (no tokens billed for nobody).
> **Interview angle:** "Every stage emits events; the transport is a detail. That's how we streamed agent steps without a second implementation."

> ### 📘 Deep dive: circuit breaker and graceful degradation (`common/resilience.py`)
> **What:** a per-dependency state machine. **closed** (calls go through) → after `threshold`=3 consecutive failures → **open** (calls are skipped, the fallback is used immediately) → after `cooldown_s`=30 s → **half-open** (one trial call: success closes it, failure re-opens it at once).
> **Why:** without it, an outage costs every request a full timeout × retries before falling back. Measured here: the first failing embedding call took **3.5 s**, the third **0.8 s**. After that the breaker was open and dense search was skipped in microseconds.
> **Ours:** `BREAKERS = {embed, plan, generate, moderation}`, shared by all requests in the process (`threading.Lock` around the counters). `CircuitBreaker.call(fn, fallback)` returns `(result, degraded)`.
> **Where the embed breaker sits matters.** The first version wrapped the whole `Retriever.dense()`, so once it opened, even *cached* query embeddings were skipped, every answer was "degraded" and nothing was ever cached. The load test showed it (2 cache hits out of 24 repeats). Now `common/llm.py:_guarded` wraps only the remote API call for cache misses (`llm.py:64`), and the Retriever catches the failure and runs BM25 alone (`retriever.py:45`).
> **Gotchas:** a breaker per *dependency*, not per input; count only failures that mean "the dependency is down" (a 400 for a bad request shouldn't open it; ours counts all exceptions, which is fine for this demo); multi-process deployments have one breaker per process.
> **Alternatives:** a second provider as the fallback (e.g. a different LLM API), a smaller self-hosted model, queuing the request for later, or failing fast with a clear message. Libraries: `pybreaker`, `tenacity` (retries), service meshes (Envoy outlier detection).
> **Interview angle:** "For each dependency I decide in advance what the user gets when it's down: BM25-only retrieval, a rule-based plan, an extractive answer. Degraded answers are labeled, never cached, and counted on the dashboard, because a fallback hides the outage from users."

---

## Step 2. Serve it and stream answers (`app.py`, `01_serve.py`, `02_client.py`)

```bash
uv run python -m phase7_production.01_serve --stub          # terminal 1 (no LLM calls); drop --stub when the key works
uv run python -m phase7_production.02_client "What was Apple's total net sales in fiscal 2025?" --feedback up
uv run python -m phase7_production.02_client --key demo-key-aapl "What was Microsoft's revenue in fiscal 2025?"
curl -s localhost:8000/ready | python -m json.tool
curl -s localhost:8000/metrics | grep rag_
open http://localhost:8000/docs                             # OpenAPI UI, generated from the pydantic models
```

### Functions
- `lifespan` (`app.py:30`): builds `RAGService` and calls `warmup()` **before** the server accepts requests: loading the index, BM25 and the reranker took 15.3 s here. That is Phase 2's lesson (a cold reranker made the first request 12 s) moved into the server lifecycle.
- `AskIn`, `FeedbackIn` (`:47`, `:52`): pydantic validation. A 2-character question or a bad `trace_id` is a 422 before any code runs.
- `current_user` (`:64`) and `rate_limited` (`:71`): FastAPI **dependencies**. Endpoints declare `user: User = Depends(rate_limited)`, which runs auth, then the rate limit, and returns 401 or 429 (with `Retry-After`) before the endpoint body starts.
- `ask` (`:90`): a plain `def` endpoint. FastAPI runs it in a worker thread, so the blocking pipeline (Chroma, torch, the sync OpenAI client) doesn't block the event loop.
- `ask_stream` (`:98`): wraps the service generator in `EventSourceResponse(..., ping=10)`. The `finally: gen.close()` handles disconnects.
- `/health` vs `/ready`: liveness ("the process runs", never touches dependencies) vs readiness (index loaded, breaker states). A load balancer routes traffic only to ready instances; an orchestrator restarts instances that fail liveness. Mixing them up makes a dependency outage restart every pod.

### What came back (stub mode)
```
[meta] trace f5e64dbb0009  route=pipeline  cache=miss  degraded=[]
[sources] [1] AAPL FY2025 Item 8  [2] AAPL FY2025 Item 8 ... [6] AAPL FY2025 Item 8
(Generated answer unavailable: showing the most relevant passage from AAPL 10-K FY2025, Item 8.)
> The Company’s proportion of net sales by disaggregated revenue source was generally consistent ...
> As of September 27, 2025 and September 28, 2024, the Company had total deferred revenue of $13.7 billion and $12.8 billion, respectively.
[1]
[guardrails] passed
[done] 2469 ms (first token 947.0 ms)  $0.00000  cache=miss  degraded=[]
[feedback] up -> 200
---
[error] access_denied: Your account doesn't have access to the filings of MSFT.
```
The extractive answer shows the reranker's top passage, a notes passage, not the income statement row. That's Phase 5's known weakness (the MS MARCO cross-encoder under-ranks bare tables), now visible because no LLM picks the right row out of the six passages. A missing key returns `401 {"detail": "missing or invalid X-API-Key"}`.

A question with an SSN: the trace stores `My SSN is [SSN], what was Tesla's total revenues in 2025?` and the `meta` event says `pii_redacted: ['SSN']`.

### How to check it worked
- `/ready` shows `"llm_backend": "stub"`, 3,043 chunks and all breakers `closed` at start.
- `uv run python -m phase1_naive_rag.06_traces --name ask_v7` lists the requests with per-stage timings (the Phase 1 viewer works on the new trace name).

> ### 📘 Deep dive: FastAPI, ASGI and uvicorn (as installed: fastapi 0.142.2, starlette 1.7.0, uvicorn 0.54.0, anyio 4.15.1)
> **What:** ASGI is the async successor of WSGI: a server (uvicorn) calls the app with `(scope, receive, send)`, and the app can stream many `send`s per request. Starlette is the ASGI toolkit (routing, responses, threadpool); FastAPI adds pydantic validation, dependency injection and generated OpenAPI docs on top.
> **How it runs our sync code:** `async def` endpoints run on the event loop and must never block it. `def` endpoints (and sync generators passed to a streaming response) run in AnyIO's worker threads, with a default limit of **40 threads** (checked: `current_default_thread_limiter().total_tokens == 40`). So one process handles up to 40 blocking requests at once; the 41st waits.
> **Workers:** `--workers N` starts N processes. Each loads its own index, BM25 and models (memory × N) and has its own caches, rate limits and breakers. That's why production keeps caches and limits in Redis.
> **Gotchas:** the GIL limits CPU-bound Python to about one core per process, but torch, NumPy and network calls release it; anything shared across threads needs a lock (Step 6: the MPS crash).
> **Alternatives:** gunicorn + uvicorn workers; Litestar, Flask (WSGI, no native streaming); managed endpoints (Modal, AWS Lambda with response streaming).
> **Interview angle:** "The pipeline is blocking, so the endpoints are sync and run in the threadpool; throughput scales with processes and replicas, and shared state lives in Redis."

> ### 📘 Deep dive: Server-Sent Events (sse-starlette 3.5.0)
> **What:** a long-lived HTTP response with `Content-Type: text/event-stream` and blocks like `event: token\ndata: {"text": "..."}\n\n`. Browsers consume it with `EventSource`, which reconnects automatically.
> **How we use it:** `EventSourceResponse(events(), ping=10)`. Checked in the installed source: a sync iterator is wrapped with `iterate_in_threadpool` (`sse.py:296`); it sets `Cache-Control: no-store` and `X-Accel-Buffering: no` (stops nginx buffering the stream; `sse.py:315-318`); the default ping interval is 15 s and we use 10. Pings are comment lines that keep proxies from closing a quiet connection while retrieval or agent steps run.
> **SSE vs WebSocket vs chunked JSON:** SSE is one-way, plain HTTP and proxy-friendly, the standard for LLM streaming (OpenAI's own API streams SSE). WebSockets are bidirectional, for voice or collaborative UIs. Newline-delimited JSON over a chunked response works for server-to-server.
> **Gotchas:** `EventSource` only does GET (our client uses `httpx.stream("POST", ...)`; browsers use `fetch` with a stream reader for POST); proxies with response buffering break streaming; HTTP/1.1 browsers allow ~6 connections per host.
> **Interview angle:** "Time to first token is the latency users feel; SSE gets the first token to them through any proxy, and the final event carries the guardrail verdict."

---

## Step 3. Caching: exact yes, semantic no (`03_cache.py`)

```bash
uv run python -m phase7_production.03_cache          # local bge-small, free, ~10 s
```

### What it does
`cache_cases.py` has 8 base questions, **24 paraphrases** with the same answer, and **30 near-misses** with a different answer, labeled by what changed (company, year, metric, sub-component, relative time). They were written by hand before running anything. `03_cache` embeds each pair with the local bge-small (the embedder the `SemanticCache` uses) and asks: at threshold t, how many paraphrases would be served (recall), and how many served answers would be wrong (1 − precision)? It does this without and with the entity guard.

### Functions
- `normalize` (`cache.py:42`): lowercase, curly → straight apostrophes, `FY2025` / `fiscal year 2025` → `fiscal 2025`, punctuation and whitespace collapsed. Only rewrites that can't change the answer.
- `signature` (`cache.py:51`): (tickers, years, canonical relative-time words). Two questions must share it for a semantic hit.
- `ExactCache.key` (`cache.py:91`): sha256(version | scope | normalized question).
- `SemanticCache.lookup` (`cache.py:117`): the nearest stored question in the same scope (and signature), served if cosine ≥ threshold.
- `_TTLStore` (`cache.py:58`): LRU + TTL with a lock: an in-process stand-in for Redis.

### Results
```
cosine, same-answer pairs:       min 0.841  median 0.968  max 0.997
cosine, different-answer pairs:  min 0.799  median 0.950  max 0.981

by kind of change (different-answer pairs): median cosine, and how many the entity guard blocks
  company        n= 7  median 0.851  guard blocks 7/7
  metric         n= 9  median 0.943  guard blocks 0/9
  relative time  n= 1  median 0.957  guard blocks 1/1
  sub-component  n= 6  median 0.953  guard blocks 0/6
  year           n= 7  median 0.956  guard blocks 7/7
  the guard also blocks 0/24 same-answer pairs (false blocks)

threshold   no guard: hits right/wrong  precision recall  |  guard: hits right/wrong  precision recall
  0.90       22 / 23    0.49   0.92            |   22 / 15    0.59   0.92
  0.95       16 / 16    0.50   0.67            |   16 / 8     0.67   0.67
  0.97       11 / 6     0.65   0.46            |   11 / 4     0.73   0.46
  0.98        6 / 1     0.86   0.25            |    6 / 1     0.86   0.25

exact cache (normalized string match) catches 0/24 paraphrases, 0 wrong answers by construction
closest DIFFERENT golden dev questions: 0.959 "Tesla total debt ... December 31, 2024" vs "... 2025" (guard blocks)
```

### What it means
- **No threshold works.** A year swap ("fiscal 2024" vs "2025") scores a median 0.956, higher than many true paraphrases ("How much did Apple sell in total in FY2025?" vs the base: 0.84). Embeddings encode *topic*; a different year or line item is the same topic.
- **The guard fixes what it can see** (companies, years, "latest/prior"), but not metric or sub-component swaps: "Tesla's *total* gross margin" vs "*automotive* gross margin" is 0.975 and passes the guard. That's Phase 3–5's "right number, wrong label" problem again, this time in the cache.
- **At the default 0.95 with the guard, 1 hit in 3 is a wrong answer.** The service ships with `SEMANTIC_CACHE=0`.
- **A bug the experiment found:** the first guard used `\b20\d\d\b` on the raw question, which can't match "FY2025" (no word boundary between `Y` and `2`), and treated "most recent" ≠ "latest". It blocked 4/24 paraphrases. Running `signature` on the normalized text with canonical time words fixed it (0/24).
- **The exact cache catches 0/24 paraphrases** but serves no wrong answers. It pays off for repeats: retries, popular questions, a shared link, a dashboard that refreshes.

> ### 📘 Deep dive: semantic caching
> **What:** store (question embedding → answer); for a new question, find the nearest stored one and serve its answer if similar enough. Products: GPTCache, Redis (vector search) semantic cache, LangChain's `RedisSemanticCache`, gateway features in some LLM proxies.
> **How ours works:** local bge-small (384-d, ~5 ms; the cache must be cheaper than what it saves and must not depend on the API it might be covering for), linear scan over entries in the same scope, optional signature guard, TTL and a size cap.
> **Where it works:** FAQ and support traffic with a small set of distinct intents ("how do I reset my password" × 1,000 phrasings), where a near-miss answer is still acceptable.
> **Where it fails:** precise lookups whose answer depends on a detail the embedding barely encodes: a year, a segment, net vs gross, a negation. Our data shows that's exactly where finance questions differ.
> **Safer variants:** cache at lower layers (embeddings, plans, retrieval results); require identical extracted entities; verify a semantic hit (re-score the cached answer's evidence against the new question with the cross-encoder); serve semantic hits only as "similar question answered before" with a link, not as the answer.
> **Interview angle:** "I measured it before shipping it: near-misses were *closer* than paraphrases, so the semantic cache would have served one wrong number in three. We cache exact repeats and the deterministic layers instead."

---

## Step 4. The guardrail layer, with its false positives and negatives (`04_guardrails.py`)

```bash
uv run python -m phase7_production.04_guardrails                       # all (moderation calls the API, if available)
uv run python -m phase7_production.04_guardrails --only pii access     # offline sections only
```

### 4a. PII (`common/pii.py`)
- `find_pii` (`pii.py:40`) runs six regexes (EMAIL, SSN, CARD, PHONE, IBAN, IP), validates cards with **Luhn** (`_luhn`, `:31`) and SSNs with the area/group rules (no `000`, `666`, `9xx`), and keeps non-overlapping matches, earliest and longest first.
- `redact` (`:58`) replaces each match with `[TYPE]` and returns the types only.

Results on 20 labeled strings: **18/20 correct**. Precision 1.00 for every type; recall 1.00 except SSN 0.50 and EMAIL 0.67.
- **False negatives (by design, documented):** `my ssn is 078 05 1120` (no dashes) and `jane at example dot com` (obfuscated). Catching these takes context words ("ssn is …") or NER, and costs false positives elsewhere.
- **Negatives that must pass, and do:** `$416,161 million`, `Commission File Number: 001-36743`, CUSIPs and share counts, `2026-2064` maturities, a 16-digit number that fails Luhn, an ISBN, `Rule 10b5-1`.
- **On the corpus:** 14 of 3,043 chunks contain a PHONE match: the companies' public headquarters numbers on cover pages (`AAPL_FY2025_Cover_002: '(408) 996-1010'`, `AMZN_FY2024_Cover_006: '(206) 266-1000'`). That's why we redact the **user's input and the logs**, not the corpus. Redacting public filings would only damage answers ("What is Apple's phone number?").

### 4b. Access control (`common/access.py`, `ScopedRetriever`)
```
pre-filter: 30 dev questions (all companies) as an AAPL-only user -> 0 non-AAPL chunks returned, 0 questions with no results
simulated filter bug: 2/60 retrieved chunks are not AAPL; the post-check flags 2 of them
agent graph tool, allowed_tickers=None:     12 evidence chunks, 12 not AAPL
agent graph tool, allowed_tickers={'AAPL'}:  0 evidence chunks, 0 not AAPL
cache isolation: same question, other scope -> None; same scope -> {'answer': 'built from all filings'}
```
- **Pre-filter works:** 30 questions about all five companies, as an AAPL-only user, return only Apple chunks. Every question still gets results (the filter narrows the search instead of emptying it).
- **The post-check catches a bug:** a subclass that "forgets" the scope returns 2 foreign chunks out of 60; `violations()` flags both. In the service they are dropped before generation and logged as an `access_violation` span and a `rag_guardrail_total{check="access_violation"}` metric.
- **The finding: the agent's graph tool bypassed the retriever's filter.** `company_relations` reads chunks *by id* from `graph_edges.json` (Phase 6), not through `R.retrieve`, so the ScopedRetriever never saw those reads: 12/12 chunks from other companies. Fix: `make_tools(..., allowed_tickers=)` filters the edges by their source chunk's ticker (`agent_tools.py:105`). **Lesson: every path to a document needs the check**: retrieval, by-id lookups, parent expansion, graph traversal, caches.
- **Cache isolation:** the scope is part of the key, so an answer built from all five filings is never served to the AAPL-only user.

### 4c. Rate and spend limits (`common/ratelimit.py`)
```
rpm= 60: 15 requests at once -> 10 allowed, then Retry-After 1.0 s
rpm=  3: 15 requests at once -> 1 allowed, then Retry-After 20.0 s
rpm=60 for 3 s of continuous requests: 12 allowed (burst 10 + ~1/s refill)
pre-call prompt-token estimate vs actual (Phase 5 dev runs, n=124): median error +0.0%, range +0.0% .. +0.0%
actual output tokens: median 26, p95 67, max 107 -> reserving max_tokens=600 overestimates the output cost ~23x
$0.002/day user: estimate $0.00065/request, actual ~$0.00034 -> 4 requests granted
```
- **Token bucket** (`ratelimit.py:31`): capacity = 10 s of traffic (`rpm/6`), refill `rpm/60` per second. A burst of 10 goes through, then 1/s.
- **The pre-call estimate is exact for the prompt.** Rebuilding the 124 Phase 5 dev prompts (structctx350, from the trace's `context` ids) and counting with tiktoken (o200k_base) plus 4 tokens per message and 3 for the reply priming gives **exactly** the `prompt_tokens` OpenAI billed, every time. The output side is the uncertain part: answers used a median of 26 tokens against the 600 reserved. The reservation is released after the call, so the only effect is that a user near their limit is refused a little early.
- This is the fix for Phase 6's open issue ("Budget overshoots by one call"): check the estimate **before** each call.

### 4d. Moderation (`common/moderation.py`)
```
fail_open=True   0.70 s  flagged=False source=unavailable  'What was Apple's total net sales in fiscal 20'
fail_open=False  0.38 s  flagged=True  source=unavailable  'What was Apple's total net sales in fiscal 20'
fail_open=True   0.41 s  flagged=False source=unavailable  'I want to hurt the people at Tesla's factory'
fail_open=False  0.00 s  flagged=True  source=unavailable  'I want to hurt the people at Tesla's factory'
breaker: {'state': 'open', 'failures': 3, 'last_error': "AuthenticationError: Error code: 401 ... token_invalidated"}
```
With the API down, the two policies show their failure modes. **Fail open** let the violent message through (a false negative, caused by the outage, not the classifier). **Fail closed** refused an ordinary finance question (a false positive for every user during the outage). The 4th call took 0.00 s because the breaker had opened. With a working key, the second message should come back `flagged=True` from `omni-moderation-latest`; that run is still to do.

> ### 📘 Deep dive: token bucket rate limiting
> **What:** each user has a bucket of `capacity` tokens refilled at `rate` per second; a request takes one or is rejected with the time until the next token (`Retry-After`).
> **Compared:** fixed window (counter per minute; lets 2× through at the window boundary); sliding log (exact, stores every timestamp); sliding window counter (weighted mix of two windows, cheap and close); leaky bucket (smooths output to a constant rate, queueing the burst).
> **Ours:** `RateLimiter.check(user, rpm)` (`ratelimit.py:56`), in memory, one bucket per user, behind a lock.
> **Gotchas:** per-process state multiplies the limit by the number of workers and replicas. Production uses Redis (a Lua script does refill + take atomically) or the API gateway (Envoy, Kong, AWS API Gateway). Limit by user *and* by what costs money (tokens or dollars per day), not just requests: one request can cost 100× another (Phase 6's agent).
> **Interview angle:** "Requests per minute stops abuse; dollars per day with a pre-call estimate stops overspend. Both are checked before any work."

> ### 📘 Deep dive: access control in retrieval
> **What:** a user may only see chunks they're entitled to. The check is a **metadata pre-filter** inside the index: Chroma's `where`, Qdrant's `Filter(must=[...])`, our BM25 boolean mask.
> **Ours:** `scoped_filter` (`access.py:43`) intersects the requested filter with the user's tickers and returns `None` when the intersection is empty (the caller searches nothing; it must never fall back to an unfiltered search). `violations` (`:58`) is the post-check.
> **Why not post-filter or prompt:** post-filtering returns fewer than k results (sometimes zero) and leaks information through counts and timing. A prompt instruction ("ignore documents from X") gives the model the forbidden text and hopes; never an access control.
> **At scale:** per-tenant collections or namespaces (strong isolation, simple deletes, many small indexes) vs one shared index with a tenant filter (efficient, but very selective filters hurt HNSW recall). Document-level ACLs that change often must be checked at query time (or re-indexed fast), never baked into embeddings.
> **Interview angle:** "The filter lives in retrieval, every document path applies it, and an assertion after retrieval turns a filter bug into a logged event instead of a leak. For agents the assertion withholds the answer, because the model has already read the evidence."

> ### 📘 Deep dive: PII detection
> **What:** find personal data in text so it can be masked before it's stored or sent on.
> **Approaches:** regex + checksums (structured identifiers: email, phone, SSN, card/Luhn, IBAN, IP; fast, precise, no names); NER models (spaCy, transformers: names, locations, organizations; slower, context-dependent false positives); frameworks like **Microsoft Presidio** (regex recognizers + NER + context words + per-entity anonymizers: mask, hash, encrypt, replace with a fake).
> **Luhn:** double every second digit from the right, subtract 9 from results above 9, sum; valid card numbers sum to a multiple of 10. It rejects ~90% of random digit strings: our 16-digit "version number" fails it.
> **Where to apply it:** before logs, traces, analytics, evaluation queues and feedback; optionally before the LLM; on outputs for data that didn't come from the sources. Reversible pseudonymization (a token map kept server-side) is needed when the answer must contain the original value (support bots).
> **Interview angle:** "I measure PII detection like any classifier, with labeled positives and corpus negatives. On public 10-Ks the only matches were headquarters phone numbers, so we redact inputs and observability, not the corpus."

---

## Step 5. A second vector database: Qdrant (`05_vector_db.py`)

```bash
uv run python -m phase7_production.05_vector_db --migrate      # copy Chroma -> Qdrant, then compare (~1 min)
uv run python -m phase7_production.01_serve --stub --qdrant    # serve from Qdrant
```

### Functions
- `QdrantCollection` (`vectordb.py:52`): implements the two methods `Retriever` calls on a Chroma collection, `query(query_embeddings, n_results, where)` and `count()`, and returns Chroma's result shape, including cosine **distance** (`1 - score`), which `store.search_by_vector` converts back. `Retriever` runs on it unchanged: `self.R.col = QdrantCollection(...)` in `RAGService.__init__`.
- `where_to_filter` (`vectordb.py:38`): Chroma's `{"$and": [{"ticker": {"$in": [...]}}, ...]}` → `Filter(must=[FieldCondition(key, match=MatchAny(any=[...]))])`.
- `migrate_from_chroma` (`vectordb.py:72`): reads ids, **stored vectors**, documents and metadata from Chroma in batches of 500 and upserts them as points. The chunk id becomes a UUID5 (Qdrant ids must be unsigned ints or UUIDs) and is kept in the payload.

### Results
```
migrated 3043 points to Qdrant (data/qdrant) in 3.8 s, no re-embedding
on disk: Chroma dir 378 MB (all 6 collections), Qdrant 51 MB
166 golden questions with cached query embeddings

recall@10 vs exact brute force      unfiltered (n=166)   ticker-filtered (n=163)
  chroma  HNSW   mean 0.996  min 0.8  perfect 161/166       mean 0.998  min 0.9
  qdrant  exact   mean 0.998  min 0.9  perfect 162/166       mean 0.998  min 0.9

dense query latency (ms): chroma p50 1.5 / p95 1.7   qdrant p50 3.8 / p95 4.1   numpy p50 0.4 / p95 0.5
full hybrid Retriever top-6 (dense+BM25+RRF+rerank), Chroma vs Qdrant backend: overlap 1.000 on 40 questions
```
- **Chroma's HNSW is nearly exact at this size:** 161/166 queries have a perfect top-10, the worst loses 2 of 10. Recall loss from approximation grows with collection size and with selective filters. Here the ticker filter (20% of the data) didn't hurt.
- **Qdrant local "exact" isn't 1.000, and that's not approximation.** Local mode is brute force (checked: `local_collection.py:820` is an `np.argsort` over all scores). Each of its 4 misses is a near-tie at rank 10/11 (e.g. 0.6446513 vs 0.6443564) that comes out in a different order because of float rounding (Qdrant normalizes vectors on insert).
- **What matters end to end:** the full hybrid retriever returns the **same top-6 on 40/40 questions** on both backends.
- **Brute-force NumPy is the fastest (0.4 ms)** at 3,043 × 1536. Indexes matter at millions of vectors, not thousands (Phase 0's conclusion, confirmed).

> ### 📘 Deep dive: Qdrant (qdrant-client 1.19.1)
> **What:** an open-source vector database written in Rust. Collections of points (id, vector(s), JSON payload); HNSW index; payload indexes for filtering; scalar, product and binary quantization; sharding and replication in cluster mode; REST and gRPC APIs.
> **How filtering works:** Qdrant filters *during* the HNSW traversal (filterable HNSW, extra links per payload value) and its query planner switches to a payload-index scan plus exact scoring when a filter is very selective. That's the main difference from bolting filters onto a plain HNSW.
> **Our config:** `QdrantClient(path="data/qdrant")`, **local mode**: an embedded pure-Python implementation for development and tests. It does exact search, has no HNSW, and warns above `LARGE_DATA_THRESHOLD = 20_000` points (checked in `qdrant_local.py:69`). Vectors 1536-d, `Distance.COSINE`, payload = Chroma's metadata + `chunk_id` + `document`.
> **In production:** the server (`docker run -p 6333:6333 qdrant/qdrant`, or Qdrant Cloud) with the same client API, `QdrantClient(url=...)`. Create payload indexes on `ticker` and `fiscal_year` (`create_payload_index`), set `hnsw_config` (m, ef_construct) and quantization for memory.
> **Gotchas:** point ids are uint64 or UUID only (hence UUID5); local mode takes a file lock (one client per process: `qdrant_client()` caches it); a client's `__del__` during interpreter shutdown raises a harmless `ImportError` unless you `close()` it (we do).
> **Alternatives:** **pgvector** (Postgres extension: HNSW/IVFFlat, SQL joins with your ACL and metadata tables, transactions; the default if you already run Postgres; good to tens of millions of vectors per node). **Milvus** (distributed, GPU indexes, billions of vectors). **Weaviate** (hybrid search built in). **Elasticsearch/OpenSearch** (BM25 + kNN in one engine). **Vespa** (ranking pipelines at scale). **Pinecone** (managed).
> **Interview angle:** "The retriever depended on two methods, so changing the vector DB was an adapter plus a migration that copied vectors instead of re-embedding. I validated it with recall against brute force and identical end-to-end top-k before switching."

---

## Step 6. Load test (`06_load_test.py`)

```bash
uv run python -m phase7_production.06_load_test --workers 2      # ~7 min; results/load_test.json
```

### What it does
For each scenario it starts a **fresh** uvicorn server (`LLM_BACKEND=stub`, `STUB_TOKEN_MS=15`, `ONLINE_EVAL_RATE=0`, `TRACE_TAG=loadtest`), warms it with 2 questions **not** in the measured set, then sends 48 requests (golden dev questions) through `/ask/stream` with an asyncio semaphore for concurrency. It records total latency, time to first `token` event, status and cache result. The stub streams the extractive answer word by word at 15 ms, close to gpt-4o-mini's speed, so the numbers are **our overhead plus a realistic stream**. Real runs add ~100–300 ms per uncached embedding call and the LLM's own time to first token (~0.3–0.7 s).

### The first finding: concurrency crashed the server
The first run died at concurrency 4: `peer closed connection without sending complete message body`. The server log showed

```
-[AGXG15GFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoderWithConfig:nextEncoderClass:]:1168:
 failed assertion `A command encoder is already encoding to this command buffer'
```
and one trace recorded `AcceleratorError: index -4822602704412016640 is out of bounds`. Both models (the cross-encoder and bge-small) run on **`mps:0`**, Apple's GPU (checked: `CrossEncoder(...).device`). The server's thread pool called `predict()` from several threads at once, and Metal's command buffers aren't thread-safe. The fix is a `threading.Lock` around inference (`rerank.py:17`, `llm.py` `_hf_lock`). It works, but it makes the reranker a **serial section**: every request queues for it.

### Results (clean run, `results/load_test.json`)
| Scenario (48 requests) | req/s | total p50 / p95 | first token p50 / p95 | cache hits |
|---|---|---|---|---|
| concurrency 1 | 0.58 | 1.76 / 2.42 s | 0.44 / 0.73 s | 0 |
| concurrency 4 | 2.11 | 1.82 / 2.78 s | 0.47 / 1.26 s | 0 |
| concurrency 16 | **2.79** | **5.19 / 6.26 s** | **3.99 / 4.97 s** | 0 |
| concurrency 4, repeats arrive later | **3.70** | 1.30 / 2.35 s | 0.37 / 1.12 s | **22/24** |
| concurrency 4, repeats in flight together | 2.06 | 1.79 / 2.76 s | 0.55 / 1.19 s | **0/24** |
| concurrency 4, `GUARDRAIL_MODE=buffer` | 2.09 | 1.85 / 2.67 s | **1.84 / 2.67 s** | 0 |
| concurrency 16, 2 workers | 2.86 | 5.37 / 6.67 s | 4.31 / 5.58 s | 0 |

Median stage time per scenario, from the traces (the scenarios run in order, 50 traces each):
```
c1             rerank  383 ms   dense 26   bm25 3   generate 1242   total 1775
c4             rerank  386 ms   dense 27   bm25 3   generate 1245   total 1843
c16            rerank 3810 ms   dense 36   bm25 3   generate 1271   total 5169
c16 2 workers  rerank 4017 ms   dense 56   bm25 4   generate 1237   total 5227
```
- **The reranker is the ceiling.** From 1 to 4 concurrent requests, throughput rises almost linearly (0.58 → 2.11 req/s) because the stub's streaming (1.2 s of sleeping) overlaps across threads. At 16, the rerank span (which includes waiting for the lock) grows **10×** to 3.8 s while generation stays flat. Throughput levels off at ~2.8 req/s, about 1 / 0.38 s of rerank work per request.
- **A second worker didn't help** (2.86 req/s, rerank 4.0 s): both processes share one GPU, so the device, not the lock, is the queue. More workers help only when they add hardware.
- **Exact cache:** 22 of 24 repeats were served from cache. The other 2 are small stampedes: the traces show those repeats started 0.25 s and 1.14 s after their first copy, which took ~2.2 s, so nothing was cached yet. Throughput rose 75% (3.70 vs 2.11 req/s).
- **Cache stampede:** when the duplicate arrives while the first is still running, **0/24** hit, because nothing is stored until an answer finishes. Popular questions arriving together all pay full price. The fix is request coalescing: the second request waits on the first's in-flight result (a per-key lock or future, "single-flight").
- **Buffer mode** costs exactly what it should: time to first token equals total latency (1.84 s vs 0.47 s in post mode). The stub's ~1.2 s of streaming becomes waiting. With a real LLM that's 1–3 s of blank screen per answer.
- Before the fix, the first run (`load.log`) also showed "2 cache hits" in the all-distinct scenarios: the warm-up requests reused the first two measured questions. The warm-up now uses questions outside the measured set.

> ### 📘 Deep dive: serving a model inside a web server
> **What went wrong:** a GPU model is one shared device with one command queue; a threaded web server makes concurrent calls by default.
> **Options:** (1) a lock: correct, but requests queue (ours). (2) **Dynamic batching:** collect concurrent requests for a few ms and run one batched forward pass (a cross-encoder scores 32 pairs in about the time of 8). (3) A separate **inference server** (NVIDIA Triton, Hugging Face TEI for embeddings and rerankers, vLLM for LLMs) that batches, scales and is monitored on its own. (4) More processes: each has its own model copy and its own lock, but on one GPU they still share the device (our 2-worker run).
> **Interview angle:** "Under load, the first thing that broke wasn't the LLM API, it was our reranker: concurrent GPU calls crashed the process. In production the reranker is its own service with dynamic batching, scaled on GPU utilization."

---

## Step 7. Dashboard and metrics (`07_dashboard.py`, `/metrics`)

```bash
uv run python -m phase7_production.07_dashboard                        # -> data/dashboard.html
uv run python -m phase7_production.07_dashboard --exclude-tag loadtest
open data/dashboard.html
curl -s localhost:8000/metrics | grep -E "^rag_(requests|cache|guardrail|degraded)"
```

### Functions
- `compute` (`07_dashboard.py`): from the `ask_v7` traces computes latency and TTFT percentiles, outcomes, refusal, error, guardrail-pass, degraded and cache-hit rates, per-stage p50/p95, PSI drift and feedback.
- `psi`: Population Stability Index of the **best rerank score per answered, uncached request**. The baseline is Phase 5's evaluated dev run on the same index (`ask_v5`, `eval:dev`, structctx350); bins are the baseline's deciles.
- `render`: a static HTML page: a KPI row of stat tiles, per-stage latency bars, outcomes, degraded dependencies, the score-drift histogram (baseline vs current, two series with a legend), feedback, and thumbs-down traces. Every chart has hover tooltips and a table view, and light and dark mode (palette checked with the dataviz validator).
- Prometheus (`common/telemetry.py`): `rag_requests_total{route,outcome}`, `rag_request_seconds{route,cached}`, `rag_time_to_first_token_seconds`, `rag_stage_seconds{stage}`, `rag_cost_usd_total{stage}`, `rag_tokens_total{direction}`, `rag_cache_total{layer,result}`, `rag_guardrail_total{check,result}`, `rag_degraded_total{dependency}`, `rag_feedback_total{rating}`, recorded by `record_metrics` from each finished trace.

### Results (all `ask_v7` traces of the day: 33 interactive + 1,022 load test)
```
1055 requests  latency p50 1.99 s p95 6.27 s  first token p50 0.85 s p95 4.82 s  cost $0.0074
outcomes {'answered': 1049, 'refused:access_denied': 2, 'refused:router:investment_advice': 1, 'error': 1, 'refused:access_violation': 2}
guardrails passed 100% (fresh answers)  degraded 4% {'embed': 45}  cache hits 6% {'miss': 994, 'exact': 61}
score drift: PSI 0.150 (baseline n=124, current n=988; best rerank median 7.55 -> 7.58)
slowest stages (p50 ms): generate 1232, agent 690, tool:search_tables 681, rerank 516, dense_search 27
```
- The **cost** ($0.0074) is the fake agent's token counts priced as gpt-4o-mini (Step 9); every other request ran without an LLM.
- **The 1 error** is the MPS crash from Step 6 (`AcceleratorError`), kept in the record.
- **Drift:** with the load test included, PSI 0.15 ("moderate"): the same dev questions, but answered with the rule-based plan instead of the LLM planner, so retrieval differs. With `--exclude-tag loadtest` (26 answered requests, most of the morning's degraded BM25-only runs) PSI is **1.08** and the median best rerank score drops from 7.55 to 6.84. That's the signal working: degraded retrieval shows up in the score distribution. 26 requests is a small sample, though; PSI needs a few hundred to be stable.
- **A dashboard bug the first render showed:** "guardrails passed 94%". All 61 "failures" were cache replays, whose traces didn't record the verdict. `_replay` now copies the stored verdict, and the dashboard counts fresh answers only.
- The screenshot check (desktop and a 520 px window) caught clipped labels ("d:router:investment_advice"), counts printed as "2.00" and a wrapping latency tile; all fixed.

> ### 📘 Deep dive: Prometheus metrics (prometheus-client 0.26.0)
> **What:** a pull-based time-series system. The app exposes current counter and histogram values as text at `/metrics`; Prometheus scrapes it every 15–60 s; Grafana queries it with PromQL (`histogram_quantile(0.95, rate(rag_request_seconds_bucket[5m]))`).
> **Types:** Counter (only goes up: requests, dollars), Gauge (goes up and down: queue depth), Histogram (counts per bucket: latency; percentiles are computed at query time, so choose buckets around your SLO), Summary (client-side quantiles, can't be aggregated across instances).
> **Ours:** a private `CollectorRegistry` (so a module reload can't register metrics twice) and buckets from 50 ms to 32 s for latency.
> **Gotcha: cardinality.** Every distinct label combination is its own time series. `route`, `stage`, `outcome` are fine; a `user_id` or `question` label would create millions of series and take Prometheus down. Per-user numbers belong in traces or logs.
> **With several workers** each process has its own registry; use the client's multiprocess mode (`PROMETHEUS_MULTIPROC_DIR`) or scrape each worker.
> **Interview angle:** "Metrics for alerting (p95, error rate, refusal rate, degraded rate, spend per hour), traces for debugging one request. Labels stay low-cardinality."

> ### 📘 Deep dive: drift and the Population Stability Index
> **What:** PSI = Σ over bins (current% − baseline%) · ln(current% / baseline%). It's a symmetric divergence between two histograms. Rules of thumb: < 0.1 stable, 0.1–0.25 moderate shift, > 0.25 large.
> **What to monitor in RAG:** retrieval scores (best rerank score, top-k mean): a shift means the questions changed or retrieval broke; query length and language; embedding centroid distance; refusal rate; answer length; citation count.
> **What it can't tell you:** whether answers are *right*. A drop in scores can be harmless (users ask broader questions) or a broken index. It's an alarm that triggers a look at traces and online-eval scores.
> **Interview angle:** "We compare the live distribution of retrieval scores with the distribution on the evaluated golden set; a PSI above 0.25 pages someone to look at sampled traces."

---

## Step 8. Online evaluation and feedback (`08_online_eval.py`, `POST /feedback`)

```bash
uv run python -m phase7_production.08_online_eval --dry-run     # queue size and estimated judge cost
uv run python -m phase7_production.08_online_eval               # judge (gpt-4.1); needs the API
uv run python -m phase7_production.02_client --feedback down "..."
```

- **Sampling** (`service.py:_after`, `:505`): a share `ONLINE_EVAL_RATE` (10%) of answered requests is appended to `data/online_eval/queue.jsonl`, **redacted**, with the passages the answer cited (with source lines: Phase 5's lesson that the judge needs the year and section).
- **Judging** (`08_online_eval.py`): Phase 4's faithfulness judge v2 (per-attribute booleans, the verdict computed in code). It's reference-free, so it works on real traffic that has no gold answer. Already-judged trace ids are skipped; if the API fails mid-run it stops and keeps what's done. It runs as a batch (cron or a queue worker), never on the request path.
- **Today:** `queue 1, ... ~1,081 input tokens, ~$0.005 with gpt-4.1`; nothing judged, because the API is down.
- **Feedback** (`service.py:feedback`, `:536`): `{trace_id, user, rating, comment (redacted), at}` → `data/feedback/feedback.jsonl` and the `rag_feedback_total` counter. The dashboard lists thumbs-down traces with their question: those are golden-set candidates (after review, as in Phase 4's `01_build_golden`).

> ### 📘 Deep dive: online vs offline evaluation
> **Offline** (Phase 4): a fixed golden set with reference answers. Comparable across versions; the regression gate. But it only measures the questions you thought of.
> **Online:** sample real traffic, judge reference-free properties (faithfulness to the cited passages, answer relevance, refusal appropriateness), and collect explicit (thumbs) and implicit (copy, rephrase, abandon) feedback.
> **The loop:** thumbs-down and low online scores → triage → new golden items → the offline gate catches the regression next time.
> **Gotchas:** judge cost scales with the sampling rate (a dial); judges have their own biases (Phase 4); thumbs are sparse and skewed toward complaints; never put raw PII in the queue.
> **Interview angle:** "Offline eval tells me a change didn't break the golden set; online eval tells me how the system does on questions users actually ask. Thumbs-down traces feed the next golden set."

---

## Step 9. OpenTelemetry export and the agent route (`common/telemetry.py`, `09_agent_stream.py`)

```bash
PHOENIX_WORKING_DIR=data/phoenix uv run phoenix serve        # optional, terminal 2: http://localhost:6006
uv run python -m phase7_production.09_agent_stream           # scripted fake LLM, no API
curl -s "localhost:6006/v1/projects/rag-mastery/spans?limit=50"   # what Phoenix received
tail -3 data/otel/spans.jsonl                                 # the same spans, from the file exporter
```

### The OTel bridge
- `setup_otel` (`telemetry.py:63`) creates a **private** `TracerProvider` (Phoenix's `register()` from Phase 6 may own the global one) with `service.name=rag-mastery`. It exports with a `BatchSpanProcessor` (defaults checked in the SDK: 5,000 ms schedule delay, batches of 512, queue of 2,048) to OTLP/HTTP at `localhost:6006/v1/traces` when Phoenix is listening, and always to `_FileExporter` → `data/otel/spans.jsonl`.
- `export_trace` (`telemetry.py:96`) converts one finished JSONL record into spans with the **original timings**. Our spans are flat (no parent ids), so the bridge **recovers nesting**: a span is appended when it ends, so a parent comes *after* its children in the record, contains their interval, and starts strictly before them. Two simpler rules failed on real traces. Time containment alone nested 0 ms spans into their neighbour (timings are rounded to 0.1 ms: `access_check` ended up "inside" `plan`); adding record order still nested a span that started on the same tick.
- `_attrs` (`:81`) turns lists and dicts into JSON strings (OTel attributes must be scalars), renames `input_tokens` → `gen_ai.usage.input_tokens` (OTel GenAI conventions), and **redacts every string again**: the exporter is the last place PII can be stopped. Phase 6's open issue was Phoenix storing full prompts.
- Span kinds (`SPAN_KIND`): RETRIEVER, RERANKER, LLM, GUARDRAIL, AGENT, TOOL, CHAIN, so Phoenix renders icons and token counts.

What Phoenix received for the agent run (read back through its REST API):
```
rag.request [CHAIN]
  input_guardrails [GUARDRAIL] · cache_lookup · plan · access_check [GUARDRAIL]
  agent [AGENT]
    agent_step [LLM] 1500 tokens · tool:search_tables [TOOL] · agent_step [LLM] 1900 · tool:company_relations [TOOL] · agent_step [LLM] 2300 · output_check [GUARDRAIL]
  access_violation [GUARDRAIL]
```

### The agent route, with a fake LLM
`FakeLLM` replaces `ChatOpenAI` inside `common.react_agent` and plays a script: `search_tables` (Apple net sales) → `company_relations(PARTNERS_WITH)` → an answer citing [1]. Everything else is real (the graph, tools, budget, retriever, service checks).
```
--- 1. full access         step 1 agent search_tables(...) · step 1 tools evidence 5 · step 2 agent company_relations(...)
                           · step 2 tools evidence 17 · step 3 check · sources: 17 passages from [AAPL, AMZN, MSFT, NVDA, TSLA]
--- 2. AAPL-only           ... step 2 tools evidence 5 · sources: 5 passages from ['AAPL']
--- 3. AAPL-only, graph-tool filter removed
                           ... step 2 tools evidence 17 · error: access_violation: This answer can't be shown with your access level. (403)
```
Steps stream as they happen (Phase 6's open item: "streaming agent steps"). The two access layers are each shown working: the tool filter (run 2) and, with the tool filter deliberately removed, the service's post-check (run 3).

**A cache bug the script found:** the first version of run 3 was answered from the exact cache, because run 2 had just cached the same question for the same scope. The replayed `done` event also said `route=pipeline` instead of the cached route. Both are fixed: `_replay` restores the route, and the script clears the cache before run 3.

> ### 📘 Deep dive: OpenTelemetry (opentelemetry-sdk 1.45.0) and OpenInference
> **What:** the CNCF standard for traces, metrics and logs: an API (what code calls), an SDK (sampling, processors, exporters) and **OTLP**, the wire protocol (gRPC or HTTP/protobuf) every major backend accepts (Jaeger, Tempo, Honeycomb, Datadog, Phoenix, Langfuse, LangSmith).
> **Data model:** a trace = spans sharing a `trace_id`; each span has a `span_id`, a `parent_id`, start/end timestamps in ns, attributes, events and a status. Context propagation (W3C `traceparent` header) links spans across services.
> **Conventions for LLM apps:** OTel's GenAI semantic conventions (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, …) and Arize's **OpenInference** (`openinference.span.kind` = LLM, RETRIEVER, RERANKER, EMBEDDING, TOOL, AGENT, CHAIN, GUARDRAIL, EVALUATOR; `llm.token_count.*`). Phoenix reads OpenInference.
> **Ours:** manual spans built after the fact from our own trace record (the bridge), not auto-instrumentation, so no pipeline code changed, and exactly one span per stage (Phase 6 saw doubled spans from two instrumentors).
> **Gotchas:** the batch processor drops spans when its queue (2,048) is full; short scripts must flush (`TracerProvider` shuts down at exit, which flushes); never put raw prompts or user data in attributes without redaction; sample in production (head sampling by trace id, or tail sampling in the Collector to keep all errors and slow traces).
> **The OTel Collector:** a separate process that receives OTLP, batches, redacts, samples and fans out to several backends. Apps then only know one endpoint.
> **Interview angle:** "We trace with OpenTelemetry using GenAI/OpenInference conventions, so the same spans go to Phoenix for LLM debugging and to our APM for latency, and we redact at the exporter."

---

## Step 10. System-design mock

Read `NOTES.md` §4 ("Design RAG for 10M documents"), then do it live: ask me to run the mock interview. I'll play the interviewer, push on the deep dives (ANN at scale, filtered search, freshness, multi-tenancy, evaluation), and give feedback against the rubric. The sizing numbers there are computed, not recalled: 500M chunks, 150B tokens ≈ $3,000 to embed, 3.1 TB of fp32 vectors vs 32 GB with 64-byte PQ codes.

---

## What to rerun after a change

| You changed | Rerun |
|---|---|
| `common/cache.py` (normalize, signature, threshold) | `03_cache`; `09_agent_stream` (cache isolation) |
| `common/pii.py` | `04_guardrails --only pii` |
| `common/access.py`, `ScopedRetriever`, `agent_tools.make_tools` | `04_guardrails --only access`, `09_agent_stream` |
| `common/ratelimit.py` | `04_guardrails --only limits` |
| `common/retriever.py`, `common/rerank.py`, `common/llm.py` | Phase 2 `02_hybrid_rrf` + `04_rerank` (must reproduce 8/8/7/10 of 12), `05_vector_db`, `06_load_test` |
| `common/vectordb.py`, the Chroma index | `05_vector_db --migrate` |
| `service.py` / `app.py` | `02_client` (a few questions, both keys), `09_agent_stream`, `06_load_test --quick` |
| `common/telemetry.py` | `09_agent_stream` with Phoenix running; check the tree via `/v1/projects/rag-mastery/spans` |
| the OpenAI key works again | `02_client` without `--stub`; `04_guardrails --only moderation`; `08_online_eval`; Phase 6's test3 run |

## Glossary
- **ASGI / WSGI:** Python's async / sync interfaces between web servers and apps.
- **Breaker (circuit breaker):** stops calling a failing dependency for a cooldown; closed / open / half-open.
- **Degraded answer:** served with a fallback; labeled, not cached, counted.
- **Exact / semantic cache:** keyed by the normalized question / by embedding similarity.
- **Fail open / fail closed:** when a check is unavailable, allow / refuse.
- **Liveness / readiness:** "the process runs" / "it can serve traffic now".
- **Luhn:** checksum on card numbers.
- **OTLP:** OpenTelemetry's wire protocol.
- **OpenInference:** span conventions for LLM apps (span kinds, token counts), read by Phoenix.
- **Pre-filter / post-filter:** restrict the search itself / drop results afterwards.
- **PSI:** Population Stability Index, a drift score between two histograms.
- **SSE:** Server-Sent Events, one-way streaming over HTTP.
- **Scope:** the set of tickers (documents) a user may see; part of every cache key.
- **Stub backend:** deterministic stand-in for the LLM (rule plan + extractive answer at LLM speed).
- **TTFT:** time to first token.
- **Token bucket:** rate limiter with a burst capacity and a refill rate.

## Self-check
1. Why is the semantic cache off, and which kinds of near-miss does the entity guard catch?
2. A comparison names MSFT and AAPL; the user may only see AAPL. What happens and why not answer the AAPL half?
3. Where exactly does the embed breaker sit, and what went wrong when it wrapped `Retriever.dense`?
4. Why can't a refusal after streaming started be an HTTP 403, and how does the client learn about it?
5. The pre-call estimate is exact for the prompt. Why does a user near their limit still get refused early?
6. Why did the agent's graph tool leak chunks past the ScopedRetriever, and what are the two fixes?
7. What crashed at concurrency 4, why, and what are three ways to fix it properly?
8. How does the OTel bridge recover parent/child relationships from flat spans, and why did pure time containment fail?
9. Which dashboard numbers would reveal that the embedding API has been down for an hour, even though users still get answers?
10. What does online evaluation measure that the golden set can't, and why is it reference-free?
