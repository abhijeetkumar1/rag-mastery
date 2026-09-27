# Phase 3: Query intelligence

```
01_rewrite       query rewriting (generic vs domain vocabulary) + multi-query / RAG-Fusion
02_hyde          HyDE: search with an LLM-written hypothetical 10-K passage
03_self_query    LLM filter extraction (companies, which filings) vs Phase 2's regex rules
04_decompose     per-company sub-questions; entity coverage; the comparison answer end to end; tolerant guardrail
05_route         router + input guardrails, measured per prompt version (v1/v2/v3/plan) on 22 labeled cases
06_ask           full pipeline: plan → per-sub-question retrieval → floor → generate → guardrails, traced (ask_v3)
cases.py         labeled analyzer cases + comparison probes (regression tests for prompts)
```
Start with `WALKTHROUGH.md` for the step-by-step code explanation and the deep dives.

```bash
uv run python -m phase3_query_intelligence.01_rewrite
uv run python -m phase3_query_intelligence.02_hyde
uv run python -m phase3_query_intelligence.03_self_query
uv run python -m phase3_query_intelligence.04_decompose
uv run python -m phase3_query_intelligence.05_route
uv run python -m phase3_query_intelligence.06_ask
uv run python -m phase1_naive_rag.06_traces --name ask_v3
```

## 1. Change the question, not the index

Phases 1–2 improved *how* we search. Phase 3 improves *what* we search for:

| Technique | Fixes | Measured here |
|---|---|---|
| Rewriting (domain vocabulary) | Vocabulary mismatch ("revenue" vs "net sales") | = baseline on the probes (10/12); the misses are reranker/chunk failures |
| Multi-query (RAG-Fusion) | Recall: different phrasings reach different chunks | hit@50 11 → **12/12**; hit@5 unchanged |
| HyDE | Query/document asymmetry for dense search | Dense-only 8 → 9/12; no gain after reranking |
| Self-query | Filters from indirect names and relative time | Companies and filings **10/10** vs regex 5/10 and 6/10 |
| Decomposition | Comparisons: coverage, per-company vocabulary and filters | Entity coverage 1/8 → **3/8**; the comparison answer is **correct** for the first time |
| Routing / input guardrails | Unsupported companies, advice, off-topic | Refusal precision **8/8**, recall 8/9, **0 false refusals** |

## 2. Lessons that transfer to any RAG system

1. **Rewrite for retrieval, rerank against the user's question.** Reranking against the rewrite dropped hit@5 from 10 to 7.
2. **A rewrite may add words; it must never be the only query.** The planner turned the 10-K's exact phrase "purchases of property and equipment" into "capital expenditures", and the answer was lost until the original question was kept as a second query.
3. **Query techniques raise recall; they can't fix ordering or dilution.** The same two probes failed under every query variant (mixed-topic chunks, reranker), so the fix belongs in chunking (Phase 5).
4. **Give the model today's date and a catalog of what exists.** Without them, "last year" and "latest fiscal year" resolved wrongly.
5. **Compute in code what code can compute.** A prompt edit flipped Microsoft's "latest" from FY2026 to FY2025; now it's computed from the catalog.
6. **An LLM router guesses about content it hasn't seen.** v3's reasons said "AWS operating income is not a term used in the 10-K filings" (false). Never let it refuse on those grounds: a code rule overrides `out_of_scope` when a covered company is named, and the relevance floor decides with evidence.
7. **Prompts need versioning and regression tests.** v2 fixed 3 cases and broke 1 that v1 got right. `PROMPT_VERSION` goes into every trace, and old versions stay runnable.
8. **Pick the cheaper error direction.** A false refusal is permanent; a false accept still meets the floor, the LLM's refusal rule and the output checks ("How do I reset my iPhone?" → router lets it through → floor refuses at −9.46).
9. **Test the guardrail on bad inputs too.** The first "tolerant" numeric check accepted the tampered numbers 15.3% and $736.9B (~10,000 number pairs → coincidences). Requiring the unit word and a single table row fixed it.
10. **Incomplete labels make good systems look bad.** The Microsoft label accepted only the MD&A sentence, not the income statement that the (correct) answer used. Coverage looked like 2/8 until the label was fixed (3/8).

## 3. The comparison question across three phases

| Phase | Microsoft (truth: +18%, +$50.1B) | Amazon (truth: +12.4%) | Why |
|---|---|---|---|
| 1 | not retrieved | not retrieved | One query vector → 5/5 Microsoft → "I don't know" |
| 2 | "+16%, $19.2B" ❌ (a **segment**) | "+15%, $36.6B" ❌ (**guidance** + an invented figure) | Fan-out gave coverage, but with the same wording for both → wrong Amazon chunks; the segment heading was lost |
| 3 | **+17.8%** ✅ ($281.7B → $331.8B) | **+12.4%** ✅ ($638.0B → $716.9B) | Per-company sub-questions, Amazon's "net sales" vocabulary, per-company "latest" filings |

The Phase 2 Microsoft error was found **here**, while verifying ground truth for the comparison probes. The Phase 2 docs have been corrected.

## 4. Still wrong: right number, wrong label

Three answers used a real number from the filing, attached to a different metric:
- **Microsoft (Phase 2):** a segment's revenue growth instead of the total.
- **Amazon capex:** $128.3B, which is "net of proceeds from sales and incentives" (non-GAAP). The cash-flow line asked about is **$131.8B**.
- **Tesla gross margin:** segment margins (16.2% automotive, 29.8% energy). The total is **18.0%**.

The numeric guardrail passes all three (the numbers exist). Fixes: a **semantic faithfulness check** (Phase 4) and **chunks that carry segment and metric context** (Phase 5).

## 5. Cost of intelligence (measured, `06_traces --name ask_v3`)

- **Planning is a second LLM call:** ~950 input / 80–107 output tokens, **1.3–1.8 s** uncached, ≈ $0.0002.
- **Answered question:** median **3.7 s** and $0.00054 with an uncached plan, 2.8 s and $0.00036 cached, against Phase 2's 2.1 s and $0.00031.
- **Router refusals cost only the plan**: no retrieval, no generation.
- **Levers:** plan caching, a rules-based fast path for simple questions, a smaller planning model, parallel speculative retrieval, streaming.

---

## Interview Q&A

**Q: What is query rewriting and when does it help or hurt?**
An LLM rewrites the user's question into a better search query: domain vocabulary, resolved references, no filler. It helps with vocabulary mismatch ("revenue" vs Amazon's "net sales") and in conversations ("what about their margins?"). It hurts when it drifts: I saw "deliveries" turned into "revenues", and the document's own phrase replaced by a synonym, which lost the answer. So I keep the original query alongside the rewrite, rerank against the user's question, and regression-test the rewriting prompt.

**Q: Explain HyDE. When would you use it?**
Generate a hypothetical answer passage with an LLM and embed that instead of the question, because answer-shaped text lands closer to real answer passages than a short question does. The fake facts don't matter; only the shape and vocabulary do. It helps dense-only retrieval in zero-shot domains (my dense-only score went 8 → 9/12) but added nothing on top of hybrid search plus a cross-encoder. Its risks are invented dates and figures biasing the search, and an extra LLM call on the critical path.

**Q: How do you handle comparison or multi-part questions?**
Decompose into per-entity sub-questions, each with its own metadata filter and its own vocabulary, retrieve them in parallel, and generate from the combined context. Measure with entity coverage, because hit@k can't see that one company's evidence is missing. In my system that fixed a comparison that had been wrong in two different ways. Multi-hop questions, where one answer determines the next query, need sequential decomposition: an agent loop.

**Q: What's self-query retrieval?**
An LLM turns the question into a semantic search string plus a structured metadata filter. Give it a catalog of what's indexed and today's date so it can resolve "latest fiscal year" or "last year" correctly, compute anything computable in code, and remember that a wrong filter silently removes the answer. So: constrain it with enums, test it on labeled cases, and fall back to an unfiltered search when filtered results are weak.

**Q: How do you design routing and input guardrails for a RAG assistant?**
One structured-output call classifies the question (answer / unsupported entity / policy violation / out of scope) and plans the retrieval. Treat it as a classifier: measure precision and recall of refusals on labeled cases, per prompt version. Decide which error is cheaper. A false refusal can't be recovered, so I override the router's out-of-scope refusals when a covered entity is named and let retrieval evidence (a relevance floor) decide. Then keep layers: router → relevance floor → the LLM's refusal instruction → output checks.

**Q: How do you manage prompt changes in production?**
Version the prompt, record the version in every trace, keep a labeled regression set, and run it on every change. My v2 prompt fixed three router errors and broke a case v1 handled. Cache LLM outputs by a hash of the full prompt, so versions never mix and experiments rerun deterministically. Prefer structural fixes (schema field order, code-level policy) over piling rules into the prompt.

**Q: Query understanding added 1.5 seconds. Is it worth it, and how would you reduce it?**
It depends on the traffic. For comparisons, date-relative and out-of-scope questions it changed wrong answers into right ones and made refusals cheap. For simple single-company questions it's mostly overhead. So: a fast path (rules detect exactly one company and no relative time → skip the planner), plan caching (exact, then semantic), a smaller planning model, running the planner in parallel with a speculative plain retrieval, and streaming the answer so time-to-first-token hides the rest.

## Experiments to try
- [ ] Add 10 new questions to `ANALYZER_CASES` *without* looking at the analyzer's output first (a held-out mini set). Is plan() still ~90%?
- [ ] Remove `VOCAB` from the `analyze()` prompt (keep the code-level `expand_terms`). Which cases break? Is the code rule enough?
- [ ] Implement the fast path: skip `plan()` when `extract_filters` finds exactly one ticker and the question has no relative-time words. Compare accuracy and latency in `06_traces`.
- [ ] Run `06_ask --multi-query` and `--hyde` on the demo questions. Does either change an answer? What does it cost?
- [ ] Map-reduce generation: answer each sub-question separately, then synthesize. Does it fix Tesla's segment-vs-total margin?
- [ ] Put a *wrong* number into the comparison answer by hand in two ways: one that is a real number from another table row, and one that is invented. Which does the tolerant guardrail catch?
