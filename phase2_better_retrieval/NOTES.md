# Phase 2: Better retrieval

```
01_bm25          BM25 from scratch; BM25 vs dense side by side
02_hybrid_rrf    RRF fusion; dense vs BM25 vs score fusion vs RRF on 12 probes
03_filters       metadata filters (ticker, fiscal year), fan-out, the filing-year pitfall
04_rerank        cross-encoder rerank: gain, N vs latency, reranker comparison, relevance-floor calibration
05_ask           full pipeline (filters → hybrid → RRF → rerank → floor → Phase 1 generator → guardrails), traced
probes.py        12 queries with text-verified relevant chunks (sanity checks, not an eval set)
```
Start with `WALKTHROUGH.md` for the step-by-step code explanation and the deep dives.

```bash
uv run python -m phase2_better_retrieval.01_bm25
uv run python -m phase2_better_retrieval.02_hybrid_rrf
uv run python -m phase2_better_retrieval.03_filters
uv run python -m phase2_better_retrieval.04_rerank
uv run python -m phase2_better_retrieval.05_ask
uv run python -m phase1_naive_rag.06_traces --name ask_v2
```

## 1. Sparse vs dense: complementary failures

| Query type | BM25 | Dense | Example |
|---|---|---|---|
| Exact names, IDs, codes | ✅ | ❌ | "Vision Pro": BM25 → Apple, dense → NVIDIA |
| Paraphrase / synonyms | ❌ | ✅ | "Can the company lose key people?" |
| Keywords that appear in boilerplate | ❌ (matches the TOC) | ✅ | "Item 1C cybersecurity" |
| A name that isn't in the corpus | Matches the other words | Drifts to the topic | "Dojo supercomputer": neither can say "not here" |

Probe hit@5: dense 8/12, BM25 8/12, but **BM25 4/4 on keyword probes and 1/4 on semantic ones**. They fail on different queries, which is the case for hybrid.

## 2. Fusion: RRF is a recall tool

- **RRF** = Σ 1/(60 + rank). It uses ranks, because BM25 (0–30) and cosine (0.4–0.8) scores aren't comparable.
- **Measured:** RRF has the **best candidate pool** (hit@50 11/12, recall@50 0.88 vs 0.75 for dense or BM25 alone) but the **worst hit@5 (7/12)**. It rewards agreement between the lists, so a chunk that dense ranked #1 and BM25 ranked #39 (the Tesla/Musk chunk) falls to #6.
- **Lesson:** hybrid without a reranker can be worse than dense alone. Fusion widens the candidate pool; something else must order it.

## 3. Filters: restrict, but don't guarantee

- **Extraction here is rule-based:** company aliases, plus only *fiscal-qualified* years.
- **Apply the same filter to both retrievers** (Chroma `where=` and a BM25 mask). Otherwise fusion brings excluded chunks back.
- **Measured:** the FY2025 filter made Apple's top 5 all FY2025.
- **`ticker ∈ {MSFT, AMZN}` still gave 6/6 Microsoft.** A filter limits what's *allowed*; it doesn't balance the results. **Fan-out** (one retrieval per company) is what gave Amazon slots.
- **The filing-year pitfall:** a "fiscal 2023" filter returns nothing, but FY2023 numbers sit inside the FY2024 and FY2025 filings. The metadata describes the document, not the facts.
- **Filtered vector search on Chroma was exact** (recall 1.00 against brute force on filtered subsets). Unfiltered HNSW was 0.99 (min 0.90) even at 2,344 vectors.

## 4. Reranking: where the precision comes from

| | hit@5 | MRR@5 | ms/query |
|---|---|---|---|
| hybrid, no rerank | 7/12 | 0.52 | 8 |
| + MiniLM-L-6 rerank (N = 10) | 10/12 | 0.69 | 69 |
| + MiniLM-L-6 rerank (N = 50) | 10/12 | 0.69 | 306 |
| + bge-reranker-base (N = 50) | 9/12 | 0.75 | 2,193 |

- **The cross-encoder fixed Phase 1's wrong-year table:** 2025 table 5.68 against 2023 table −0.36. It reads "fiscal 2025" and the column headers together.
- **The remaining misses are mixed-topic chunks and vocabulary gaps**, not truncation (the chunks are under 512 tokens).
- **Reranker scores are model-specific,** so a threshold must be recalibrated per model.

## 5. The relevance floor (a guardrail before generation)

- **Floor −3.0 (MiniLM):** keeps 12/12 answerable probes and blocks 3/5 unanswerable questions, **without an LLM call** ("Dojo" refused in 367 ms).
- **It can't block wrong-entity questions** (Google 1.83, Meta 1.99), because on-topic chunks from *other* companies exist. Those need input entity validation (Phase 3), the LLM's refusal rule, and the output checks.

## 6. The Amazon case: every layer in one question

"Compare Microsoft and Amazon revenue growth":
1. **Fan-out** gave Amazon slots, and the question got answered (Phase 1 said "I don't know").
2. **But** the question says "revenue" and Amazon says "net sales", so every Amazon candidate scored below −2.5, and the top Amazon chunk was **Q1 2026 guidance**.
3. **The LLM wrote "+15%, $36.6B".** The truth is **+12.4%** (637,959 → 716,924).
4. **The numeric guardrail flagged 36.6** (invented) ✅. It passed "15", which is real but taken from a forecast: the known false-negative type.
5. **Asked in Amazon's vocabulary**, the top hit is the right table (score 3.06). **This is the motivation for Phase 3:** per-entity query rewriting.

## 7. Observability findings

- **Latency budget:** rerank about 330 ms (warm), generation 1.3–1.5 s, dense about 60–85 ms, BM25 3 ms.
- **Cold start:** the first request took about 12 s to load torch and the reranker, and **p95 went to 11.9 s**. Fixed with `Retriever.warmup()` at startup. A classic production issue that's only visible in the traces.
- **Per-stage `top` lists in the trace** show, for every question, which retriever found the right chunk and which stage lost it (the Tesla trace: dense #1 → fused #6 → reranked below the compensation notes).

---

## Interview Q&A

**Q: Why hybrid search? Isn't dense retrieval enough?**
Dense retrieval misses exact identifiers, rare names and codes; embeddings blur "Vision Pro" into a topic. BM25 misses paraphrases and synonyms. Their failures barely overlap, so production systems run both and fuse the results. In my probes, BM25 hit 4/4 keyword queries and 1/4 semantic ones, and dense did the opposite on names.

**Q: How do you combine BM25 and vector scores?**
Not by adding raw scores: they're on unrelated scales (BM25 is unbounded, cosine sits in a narrow band). Reciprocal Rank Fusion uses ranks, Σ 1/(k + rank) with k = 60, so it needs no calibration and works for any number of retrievers. The alternative is min-max normalization plus weights, which is sensitive to outliers. And RRF is a recall step: in my measurements it improved recall@50 but hurt top-5 until a reranker was added.

**Q: What's the difference between a bi-encoder and a cross-encoder, and where does each go?**
A bi-encoder embeds query and document separately, so documents are precomputed and search is fast: the first stage. A cross-encoder reads query and document together, with attention across both, and outputs one relevance score. It's much more accurate (it caught the fiscal-2025 vs 2023 table mismatch that cosine missed), but it costs one forward pass per pair and nothing can be precomputed. So it reranks a shortlist of 20–100 candidates.

**Q: How many candidates should you rerank?**
It's a latency/recall trade-off, and latency is linear in N (about 6 ms per pair for MiniLM on CPU: N = 50 took ~300 ms, N = 100 ~600 ms). You tune it on an eval set: the smallest N where recall@N stops improving. On my probes N = 10 matched N = 100, but harder queries benefit from a larger pool.

**Q: How do metadata filters work in a vector DB, and what can go wrong?**
Pre-filter (exact, but can be slow), post-filter (fast, but can return fewer than k results), or filtering inside the ANN walk (efficient, but a very selective filter can break HNSW's graph connectivity). Good DBs pick a strategy per query based on selectivity. Also: a filter on a *set* of entities doesn't guarantee each one appears (fan-out fixes that), filters must be applied to every retriever in a hybrid setup, and metadata often describes the document rather than the fact (filing year vs fiscal year of the data).

**Q: How do you stop a RAG system from answering when it has nothing relevant?**
Layers. Before generation: a relevance floor on calibrated reranker scores (cosine is too compressed for this) blocks off-topic questions with no LLM call. On the input: entity validation catches "a company we don't cover". In generation: an explicit refusal instruction. After generation: groundedness checks. Each layer is measured; my floor blocked 3/5 unanswerable questions and lost none of the answerable ones.

**Q: Your hybrid search improved recall but answers didn't improve. Why might that be?**
Fusion without reranking can reduce top-k precision (it did here: hit@5 went from 8 to 7). Also check whether the new chunks survive to the prompt (k, reranking), whether the reranker handles your domain (web-trained models struggle with financial tables), and whether the failure is actually vocabulary ("revenue" vs "net sales"), which retrieval tuning can't fix but query rewriting can.

## Experiments to try
- [ ] Tune BM25: `k1 ∈ {0.9, 1.2, 2.0}`, `b ∈ {0.3, 0.75, 1.0}`. Rerun `02`. Do the keyword and semantic probes move in opposite directions?
- [ ] Add a simple suffix stemmer to `tokenize()` (e.g. strip "s", "ed", "ing"). Does the Apple buybacks probe ("buybacks" vs "repurchased") improve? What breaks?
- [ ] Weighted RRF: give dense weight 2. Does the Tesla/Musk chunk recover in the fused ranking without a reranker?
- [ ] Rerun `04` with `RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2` (twice as deep). Where does it sit between L-6 and bge on quality and latency?
- [ ] Ask the Amazon comparison in Amazon's vocabulary ("... Microsoft revenue and Amazon net sales ...") with `05_ask`. Is the answer now correct, and does the guardrail pass?
- [ ] Calibrate your own floor: add 5 unanswerable questions to `OUT_OF_CORPUS` in `04`. Does −3.0 still hold?
- [ ] Run `05_ask --mode dense --no-rerank --filters none` on the demo questions to reproduce Phase 1 from the Phase 2 code (an ablation), then turn on one stage at a time.
