# Phase 7: Production (notes)

Phases 1–6 built a RAG system that gives the right answer. Phase 7 makes it something you can run for other people: a service that streams, stays up when a dependency fails, doesn't leak data across users, costs what you planned, and tells you when it's getting worse. Then a system-design mock: the same system at 10M documents.

All numbers below are from this repo's runs (2026-10-15). The OpenAI key was invalid the whole day (401), so **every measurement here was made without an LLM**: rule-based planning, extractive answers and a stub that streams them. That outage is also the first lesson.

---

## 1. Results at a glance

| Experiment | Script | Result |
|---|---|---|
| Semantic cache: can a threshold separate paraphrases from near-misses? | `03_cache` | **No.** Paraphrase cosine median 0.968; *different-answer* pairs median 0.950. At 0.95 + entity guard: 16 right hits, **8 wrong answers** (33%). At 0.98: 6 right, 1 wrong, recall 0.25. Semantic cache stays **off**. |
| Entity guard (same tickers / years / relative-time words) | `03_cache` | Blocks 7/7 company swaps, 7/7 year swaps, 1/1 "latest vs prior"; **0/9 metric swaps, 0/6 sub-component swaps**. A bug in v1 (couldn't see the year in "FY2025") blocked 4/24 paraphrases; fixed → 0/24. |
| Exact cache (normalized) | `03_cache` | Catches 0/24 paraphrases: it only helps for repeated identical questions (retries, popular questions, a shared link). |
| PII regexes | `04_guardrails` | 18/20 labeled strings right; misses: SSN without dashes, "jane at example dot com". **14 corpus chunks** match PHONE: companies' public HQ numbers. → redact input and logs, never the corpus. |
| Access control | `04_guardrails`, `09_agent_stream` | Pre-filter: 30 dev questions as an AAPL-only user → 0 foreign chunks. Simulated filter bug → post-check catches 2/2. **The agent's graph tool bypassed the retriever filter** (12/12 foreign chunks) until it got its own filter; with a fake LLM, the agent route withholds the answer when foreign evidence enters its context. |
| Pre-call cost estimate | `04_guardrails` | tiktoken + 4 tokens/message + 3: **0.0% error on all 124** Phase 5 dev prompts. Output is the loose part: median 26 tokens vs 600 reserved (~23×), released after the call. |
| Circuit breaker | service traces | First failing embedding call 3.5 s, 3rd 0.8 s, then the breaker opens and dense search is skipped instantly (BM25-only). |
| Chroma → Qdrant | `05_vector_db` | 3,043 vectors in 3.8 s, no re-embedding. Chroma HNSW recall@10 vs brute force 0.996 (161/166 perfect); Qdrant local (exact) 0.998, its misses are float-precision ties at rank 10/11 (Δ < 0.0003). **Hybrid top-6 identical on 40/40.** |
| Load test (stub LLM) | `06_load_test` | Throughput levels off at **~2.8 req/s**; p95 time to first token 0.73 s (c1) → 4.97 s (c16), all from queueing at the reranker. Exact cache +75% throughput on repeats; **0 hits for concurrent duplicates** (stampede). Buffer mode: TTFT = total. 2 workers: no gain (shared GPU). §6 |
| Concurrency bug | `06_load_test` | 4 concurrent requests **crashed the server**: two threads in the MPS (Apple GPU) cross-encoder at once. Fixed with an inference lock; the lock is now the throughput ceiling. |

---

## 2. The production request path

```
auth (API key → user, scope) → rate limit (token bucket, 429 + Retry-After)
→ input guardrails (length, PII redaction, moderation) → cache (exact; semantic optional)
→ plan (LLM, rule-based fallback) → access check (named companies vs entitlement)
→ route (pipeline | agent) → retrieval with the user's filter (pre-filter) → ACL assert (post-check)
→ injection scan → relevance floor → spend check (pre-call estimate) → generate (stream; extractive fallback)
→ output guardrails (citations, numeric grounding, PII not in sources) → cache store (only clean answers)
→ online-eval sample → trace (JSONL) → OTel export + Prometheus metrics
```

Every check that can refuse runs **before** the expensive step it protects. Rate limiting comes before any work, the access check before retrieval, the spend check before the LLM call, and the relevance floor before generation.

---

## 3. Theory and trade-offs

### Streaming
- **What users feel is time to first token (TTFT), not total latency.** Streaming doesn't make the answer faster; it makes the wait visible and short.
- **SSE vs WebSocket.** Server-Sent Events are one-way (server → client), plain HTTP, and pass through proxies and load balancers. They reconnect automatically and suit LLM output. Use WebSockets when the client must also send mid-stream (voice, collaborative editing).
- **Streaming conflicts with output guardrails.** A check on the whole answer can only run at the end. The options:
  - **post mode:** stream, then send a verdict event (the UI flags or collapses the answer). TTFT stays low, but the user may read an unverified number.
  - **buffer mode:** hold the answer, check it, then send. Safe, but TTFT equals the full latency.
  - **incremental checks:** check sentence by sentence and stop the stream on a violation. This needs checks that work on partial text.
- **Client disconnects.** Close the generator, which closes the upstream LLM stream, so you stop paying for tokens nobody reads, and still end the trace.
- **Errors mid-stream.** The HTTP status is already 200, so failures become `error` events, and a refusal after streaming has started can't become a 403.

### Caching layers (cheapest to riskiest)
1. Embedding cache: hash(model + text). Always safe.
2. Structured LLM calls (planner): hash(model + messages + schema). Safe, but a **volatile value in the key kills it**. Our planner prompt contains today's date, so its cache resets daily; that's measured in this phase.
3. Retrieval results, keyed by (sub-query, filters, index version). Safe-ish: no generation is reused.
4. Exact answer cache: normalized question + scope + version. Safe if the key is complete.
5. Semantic answer cache: unsafe for precise numeric questions (§1). Fine for FAQ-like traffic with few distinct answers, ideally with a verification step.

What goes in every key: **access scope** (otherwise a cross-tenant leak through the cache) and **version** (index, models, prompt version; otherwise stale answers after a re-index). Only cache answers that passed the guardrails and weren't degraded: caching a bad answer makes a one-off failure persistent. Invalidate on re-index by changing the version string, not by deleting entries one by one.

### Graceful degradation
| Dependency down | Fallback | Cost |
|---|---|---|
| embeddings API | BM25-only retrieval | worse recall on paraphrased questions |
| planner LLM | rule-based plan (regex companies/years, one sub-question each) | can't detect unsupported companies ("Google") or out-of-scope questions |
| generation LLM | extractive answer: best passage's lines, quoted and cited | no synthesis or comparison, but no hallucination either |
| moderation | fail open (log it) | right for a low-harm domain; fail closed for health or minors |
| agent | route to the pipeline | multi-hop questions get worse |

- **Circuit breaker:** after N failures, stop calling for a cooldown (so you don't pay N retries × timeout per request), then try one half-open trial.
- **Retries:** batch jobs want many retries with backoff; a user-facing request wants one retry and a short timeout. Our client had `max_retries=8` for evaluation; the service uses 1 retry and a 20 s timeout.
- **Rule:** decide the fallbacks in advance, label degraded answers, never cache them, and show the degraded rate on the dashboard (fallbacks hide outages from users, so something else must reveal them).

### Rate limits and budgets
- **Algorithms:** token bucket (burst + steady rate) vs fixed window (allows 2× at the boundary) vs sliding log (exact, memory-heavy) vs sliding window counter (a good approximation).
- **Where to keep the state:** in-process limits multiply by the number of workers and replicas. Use Redis (atomic Lua script) or the API gateway.
- **Limit what costs money:** cap requests per minute for abuse and **dollars per day** for spend. The cost is known only after the call, so reserve a pre-call estimate (exact prompt tokens + max_tokens) and settle with the actual cost.
- **Agents:** the same estimate before every agent step closes Phase 6's "overshoots by one call" gap.

### Access control in RAG
- **Pre-filter in the index** (metadata filter inside the vector DB and BM25) is the access control. **Post-filtering** leaks through counts and timing and returns fewer than k results. **Prompt instructions are never access control.**
- **Every path to a document needs the filter:** retriever, graph lookups, "get chunk by id" tools, parent expansion, caches. We found the graph tool bypassing it.
- **Defense in depth:** assert after retrieval. In a pipeline, drop the offending chunk before generation. In an agent, evidence is already in the context, so **withhold the answer**.
- **At scale:** per-tenant namespaces or collections (strong isolation, many small indexes) vs a shared index with tenant-id filters (efficient, but filtered-HNSW recall drops for small tenants). Document-level ACLs that change often argue for filter-at-query over baking ACLs into embeddings.

### PII
- **Redact user input before it reaches logs, traces, evaluation queues and feedback.** These stores are widely read and long-lived.
- **Redact in the exporter too:** the exporter is the last place PII can be stopped before it leaves the process.
- **Don't redact a public corpus.** 14 chunks have HQ phone numbers. Redacting them would only damage answers.
- **Regex vs NER:** regexes catch structured identifiers (email, phone, SSN, card with Luhn, IBAN, IP). NER (Presidio, spaCy) catches names and addresses, but flags every executive name in a 10-K, a false positive for public filings.

### Observability in production
- **Traces** (one request, every stage) for debugging; **metrics** (rates, percentiles) for dashboards and alerts; **logs** for events.
- **OpenTelemetry** is vendor-neutral: one SDK and exporter config sends to Phoenix, Jaeger, Tempo, Datadog or Langfuse (all accept OTLP).
- **Semantic conventions:** OTel GenAI `gen_ai.*` (usage, model) and OpenInference (`openinference.span.kind`: LLM, RETRIEVER, RERANKER, GUARDRAIL…).
- **Metric labels must have low cardinality:** route, stage, outcome. **Never a user id or a question:** each distinct label set is a separate time series.
- **What to alert on:** p95 latency, error rate, refusal-rate spikes (a broken router or index looks "safe"), guardrail failure rate, degraded rate, spend per hour, score drift.
- **Drift:** compare the distribution of retrieval scores, query length or embedding centroids with a baseline window. PSI below 0.1 is stable; above 0.25 is a large shift.
- **Online evaluation:** judge a sample of real traffic with reference-free metrics (faithfulness to the cited passages), offline and in batches. Thumbs-down traces become golden-set candidates.

### Guardrail frameworks vs hand-rolled
| | What it is | Good at | Watch out |
|---|---|---|---|
| **Hand-rolled** (this repo) | regex + rules + small LLM classifiers, in your code | exact control, measurable FP/FN, zero latency for rules | you maintain every rule; no coverage beyond what you thought of |
| **NeMo Guardrails** (NVIDIA) | a runtime with "rails" written in Colang: input/output/dialog/retrieval rails, LLM self-checks | dialog flows, topical rails, plugging many checks into one config | extra LLM calls per rail (latency and cost); a DSL to learn |
| **Guardrails AI** | validators (a hub of them) wrapped around LLM output, with re-ask on failure | structured-output validation, PII, toxicity, competitor mentions | quality varies per validator; re-asking costs tokens |
| **Llama Guard 3** (Meta) | an open-weights classifier for prompts and responses over a hazard taxonomy (1B / 8B) | local moderation; no data leaves; a custom taxonomy in the prompt | needs a GPU for latency; English-centric; not a grounding checker |
| **Moderation APIs** (OpenAI omni-moderation, Azure Content Safety) | hosted classifiers | free or cheap, fast, broad categories | a dependency (fail open or closed?); generic categories |

For this domain the risky failures are **wrong numbers, wrong entity and leaked data**, not toxicity. No framework checks numeric grounding against 10-K tables, so the domain checks stay hand-rolled. A framework would add moderation and topical rails around them.

---

## 4. System design mock: "Design RAG for 10M documents"

How to run it in the interview (about 45 minutes): clarify (5) → estimate (5) → high-level design (10) → deep dives (15) → evaluation and operations (5) → trade-offs (5).

**Clarify.** What documents (10-Ks? PDFs with tables? tickets?), how many queries per second, latency target (p95 TTFT < 1.5 s?), freshness (new documents searchable within minutes or days?), multi-tenant with ACLs?, languages, what an error costs (a wrong number vs a vague answer), budget per query.

**Estimate** (computed in this repo: `chunks_structctx350` averages 257 tokens/chunk and about 300 chunks per 10-K):
- 10M docs × 50 chunks = **500M chunks**; at 300 tokens, **150B tokens to embed ≈ $3,000** once with text-embedding-3-small. Re-embedding on a model change costs the same: version the index, never mix models.
- Vectors at 1536 dims: fp32 **3.1 TB**; int8 768 GB; binary 96 GB; PQ with 64 bytes **32 GB**. HNSW links (M=16) add about 64 GB. → quantize, shard, and keep only the compressed vectors in RAM, with fp32 on disk for re-scoring.
- 100 QPS × (1 embedding + 2 searches + rerank 50 + 1 LLM call of ~2k tokens) → LLM cost dominates (at gpt-4o-mini prices about $0.0003/query, so $2.6k/day at 100 QPS), then reranker GPUs.

**High-level design.**
- **Ingestion (async, queue-driven):** fetch → parse (layout-aware: tables keep their headers, Phase 5) → chunk (structure-aware + contextual header) → embed (batched, GPU or API, rate-limited) → upsert to the vector DB + BM25/inverted index → metadata store (doc id, tenant, ACL, version). Idempotent by content hash, so re-runs and updates only touch changed chunks.
- **Query path:** gateway (auth, rate limit) → planner (cached) → hybrid retrieval with tenant/ACL pre-filters over a **sharded** index → fusion → cross-encoder rerank (GPU, batched) → generation (streamed) → guardrails → caches.
- **Stores:** a vector DB (Qdrant, Milvus, Vespa or pgvector up to ~tens of millions per node; sharded HNSW or IVF-PQ), a search engine for BM25 (OpenSearch/Elasticsearch, or one engine that does both: Vespa, Elasticsearch kNN), a document store for parent text, Redis for caches and limits.

**Deep dives to offer.**
- **ANN at scale:** HNSW (fast, memory-hungry, filtered recall drops) vs IVF-PQ (compact, needs training and re-scoring) vs DiskANN (SSD-resident). Shard by document hash for even load or by tenant for isolation; query all shards and merge the top-k. Re-score with full-precision vectors.
- **Filters:** pre-filter with payload indexes. With very selective filters (a small tenant), HNSW's graph breaks up, so use per-tenant shards or fall back to brute force on small subsets (Qdrant's planner does this).
- **Freshness:** a write-ahead buffer index for new documents searched alongside the main index; periodic compaction; deletions as tombstones plus a filter.
- **Index versions:** build the new index alongside the old one, shadow traffic, compare offline metrics, switch an alias. Caches key on the version.
- **Latency budget (p95 TTFT 1.5 s):** gateway 10 ms, plan 300 ms (or cached), embedding 50 ms, search 30 ms, rerank 100 ms on a GPU, LLM TTFT 500 ms, so about 1 s. The agent path is opt-in for multi-hop questions only (Phase 6: 2.7× cost).
- **Multi-tenancy and ACLs:** tenant id on every chunk; filters enforced in the retrieval service, never by the LLM; caches keyed by scope; per-tenant rate limits and budgets.

**Evaluation and operations.** Golden sets per document type with frozen test splits (Phase 4); a regression gate in CI on every prompt, model or index change; online faithfulness sampling; thumbs-down → triage → golden set; dashboards (latency, cost, refusals, degraded rate, drift); canary releases for prompt and model changes.

**Trade-offs to name.** Recall vs latency (candidate count, rerank depth); cost vs quality (gpt-4o-mini + a strong reranker vs a bigger model); freshness vs index quality; isolation vs efficiency (per-tenant vs shared index); semantic cache savings vs wrong answers (§1).

---

## 5. Interview Q&A

**Q: How do you take a RAG prototype to production?**
I put it behind a service that streams, authenticates and rate-limits. Each dependency gets a timeout, a circuit breaker and a planned fallback. Retrieval is filtered by the user's entitlements. Spend is checked before each LLM call. Every request is traced with OpenTelemetry and exposed as metrics, and real traffic is sampled for online evaluation. In my project the OpenAI key died during this phase, so the service ran on its fallbacks: BM25 instead of dense search, rule-based planning and extractive answers. It kept answering, labeled those answers degraded and never cached them.

**Q: Would you add a semantic cache?**
Not for numeric questions. I measured it with bge-small on labeled pairs. True paraphrases had a median cosine of 0.968, and questions with different answers had 0.950. "Total gross margin" vs "automotive gross margin" scored 0.975. At a 0.95 threshold, a third of the hits were wrong answers even with an entity guard that blocks company and year swaps. The guard can't catch metric changes. I keep an exact cache keyed by scope and version, cache the lower layers (embeddings, plans, retrieval), and would use a semantic cache only for FAQ-style traffic or with a verification step.

**Q: How do you stream and still run output guardrails?**
I choose per risk. Post mode streams the answer and then sends a verdict event the UI uses to flag or collapse it, so time to first token stays low. Buffer mode holds the answer until the checks pass, so time to first token equals total latency: 1.84 s vs 0.47 s in my load test (§6). Incremental checking stops the stream at the first violating sentence. For financial numbers I'd use buffer mode on the high-risk route, or stream with a visible "verifying…" state.

**Q: How do you enforce document permissions in RAG?**
With a metadata pre-filter inside the index, never a prompt instruction. Every path that can fetch a document must apply it, then an assertion after retrieval catches bugs. In my agent, a graph tool fetched chunks by id and skipped the retriever's filter: an AAPL-only user got 12 chunks from other companies. The tool now has its own filter. For an agent, the post-check withholds the answer, because the evidence is already in its context. Caches are keyed by scope, otherwise a cached answer leaks across tenants.

**Q: What breaks under load?**
In my case, the first thing that broke was the model server, not the LLM. At 4 concurrent requests, the cross-encoder on Apple's GPU crashed the process: two threads used the same Metal command buffer. A lock fixed the crash, but now the reranker is the throughput ceiling: 2.8 req/s, and p95 time to first token went from 0.7 s at 1 concurrent request to 5 s at 16 (§6). The real fixes are dynamic batching or a separate inference server (Triton, TEI) with its own scaling, and more processes, each with its own model copy.

**Q: What do you put on the dashboard?**
p50/p95 latency and time to first token, cost per request and per stage, outcomes (answered, refused by reason, error), guardrail pass rate, degraded rate by dependency, cache hit rate, and drift: the population stability index of the best rerank score against the evaluated baseline. I also show thumbs-down rate with links to the traces, and online faithfulness. Refusal spikes and degraded rate are the RAG-specific alerts: a broken index looks "safe" because it refuses more.

**Q: How do you estimate and control LLM cost per user?**
Before the call I count prompt tokens with tiktoken (exact: 0.0% error on 124 prompts in my runs) and add max_tokens as the output bound. That estimate is reserved against the user's daily budget and settled with the actual cost. Answers in my runs used a median of 26 output tokens against 600 reserved, so the reservation is conservative by design. For agents the same estimate runs before every step.

**Q: Chroma, pgvector or Qdrant?**
Chroma is embedded and easy, good for prototyping. pgvector is the choice if you already run Postgres: transactions, joins with your metadata and ACL tables, HNSW or IVFFlat. It's fine to tens of millions of vectors per node. Qdrant (or Milvus, Weaviate) is a dedicated server with payload indexes, filtered HNSW, quantization, sharding and replication. Retrieval only talked to the store through two methods, so I swapped Chroma for Qdrant with an adapter. I copied the stored vectors rather than re-embedding, and got the same top-6 on 40/40 questions.

---

## 6. Load test (stub LLM, 15 ms/word)

| Scenario (48 requests) | req/s | total p50 / p95 | first token p50 / p95 | cache hits |
|---|---|---|---|---|
| concurrency 1 | 0.58 | 1.76 / 2.42 s | 0.44 / 0.73 s | 0 |
| concurrency 4 | 2.11 | 1.82 / 2.78 s | 0.47 / 1.26 s | 0 |
| concurrency 16 | 2.79 | 5.19 / 6.26 s | 3.99 / 4.97 s | 0 |
| c4, repeats arrive later | 3.70 | 1.30 / 2.35 s | 0.37 / 1.12 s | 22/24 |
| c4, repeats in flight together | 2.06 | 1.79 / 2.76 s | 0.55 / 1.19 s | 0/24 (stampede) |
| c4, buffer mode | 2.09 | 1.85 / 2.67 s | 1.84 / 2.67 s | 0 |
| c16, 2 workers | 2.86 | 5.37 / 6.67 s | 4.31 / 5.58 s | 0 |

- **Bottleneck:** the median rerank span goes 383 ms (c1) → 3,810 ms (c16) while generation stays ~1.25 s: requests queue for the reranker lock (one GPU). A second worker doesn't help (it shares the GPU).
- **Cache:** +75% throughput when repeats arrive after the first answer, 0 hits when they arrive together, so coalesce in-flight requests ("single-flight").
- **Buffer mode:** time to first token = total latency.

---

## 7. Experiments to try

- [ ] **Dynamic batching for the reranker:** collect concurrent rerank requests for ≤10 ms and score them in one `predict()`. How do throughput and p95 change at concurrency 16?
- [ ] **Two workers vs the lock:** each process has its own model copy (memory ×2). Does throughput double?
- [ ] **Buffer mode with withhold:** with the real LLM, how many answers would be withheld on dev (the numeric check's false positives, Phase 1: 2/10)?
- [ ] **Semantic cache with verification:** on a semantic hit, rerank the cached answer's cited passages against the NEW question with the cross-encoder, and serve only if the top score clears the old one's. Rerun `03_cache`.
- [ ] **Pre-call budget inside the agent:** call `estimate_cost` before every `agent_node` LLM call and route to finalize when the estimate would cross `max_cost_usd`. Rerun Phase 6's `04_limits` max_cost case.
- [ ] **Planner cache across days:** take today's date out of the analyzer prompt (pass it only when the question has relative time). How much does the plan cache hit rate rise?
- [ ] **Llama Guard 3 1B** locally as the moderation fallback instead of fail-open: latency on CPU/MPS, and agreement with omni-moderation.
