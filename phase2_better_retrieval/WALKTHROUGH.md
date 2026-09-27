# Phase 2 walkthrough: better retrieval

Read this top to bottom and run each command as you reach it. The theory, the interview Q&A and the experiments are in `NOTES.md`. All numbers are from real runs on the current index (`text-embedding-3-small`, recursive chunks of 350 tokens, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`).

**What this phase fixes from Phase 1:**
- exact names that dense search misses ("Vision Pro"),
- the wrong-year table ranked #1,
- the comparison question that returned only Microsoft,
- paraphrases whose right chunk sits at rank 6+,
- and "the retriever always returns *something*".

## Architecture

### The retrieval pipeline

```
question
   │
   ├─► filters (rule-based: company aliases, "fiscal 2025") ─────────────────────────────┐
   │        │ >1 company? ──► fan-out: run the pipeline below once per company            │
   ▼        ▼                                                                             │
┌──────────────────────┐    ┌──────────────────────┐                                      │
│ dense search         │    │ BM25 search          │   both see the same filter:          │
│ Chroma HNSW, top-50  │    │ own inverted index,  │   Chroma where= / boolean mask ◄─────┘
│ (semantic matches)   │    │ top-50 (exact terms) │
└──────────┬───────────┘    └──────────┬───────────┘
           └────────────┬──────────────┘
                        ▼
              ┌───────────────────┐
              │ RRF fusion        │  Σ 1/(60 + rank): combines RANKS, ignores incomparable scores
              │ → 50 candidates   │  job: RECALL (get the right chunk into the pool)
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ cross-encoder     │  reads (question, chunk) together, 50 forward passes, ~330 ms
              │ rerank → top-5    │  job: PRECISION (put the right chunk first)
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ relevance floor   │  best rerank score < −3.0 → refuse, no LLM call
              └─────────┬─────────┘
                        ▼
      Phase 1 generator (same prompt, unchanged) → output guardrails → answer
```

Every box is a **trace span** (`filters`, `dense_search`, `bm25_search`, `fusion`, `rerank`, `relevance_floor`, `generate`, `guardrails`). In `06_traces` you can see, for any question, which stage brought the right chunk in and which one dropped it.

**The design idea: recall first, then precision.**
- **Cheap, broad stages** (BM25 and dense, milliseconds each) cast a wide net of 50 candidates.
- **An expensive, accurate stage** (the cross-encoder, one model call per candidate) orders them.
- This **two-stage retrieve → rerank** pattern is how production search engines work.

### Components

| Layer | Module | Key functions | Tech |
|---|---|---|---|
| Sparse retrieval | `common/bm25.py` | `tokenize`, `BM25Index.scores`, `BM25Index.search` | Plain Python + NumPy (from scratch) |
| Dense retrieval | `common/store.py` (Phase 1) | `search_by_vector(where=...)` | Chroma HNSW |
| Fusion | `common/fusion.py` | `rrf`, `score_fusion` | Plain Python (from scratch) |
| Filters | `common/filters.py` | `extract_filters`, `to_chroma_where`, `to_mask` | Regex + alias table |
| Reranking | `common/rerank.py` | `rerank_scores` | sentence-transformers `CrossEncoder` |
| Orchestration | `common/retriever.py` | `Retriever.retrieve`, `retrieve_fanout`, `warmup` | Traces each stage |
| Probes | `phase2_better_retrieval/probes.py` | `PROBES`, `relevant_ids`, `hit_and_rr` | 12 queries with text-verified relevant chunks |
| Generation | `phase1_naive_rag/04_ask.py` (reused) | `SYSTEM`, `format_context` | Unchanged from Phase 1: a controlled experiment |

### What a hit looks like now

Phase 1 hits were `{id, text, meta, score}`. Phase 2 hits carry the **history of every stage**:
```
{id, text, meta,
 dense_rank, dense_score,        # position/cosine in the dense list (absent if dense didn't return it)
 bm25_rank, bm25_score,          # position/BM25 score in the sparse list
 rrf_score, fused_rank,          # after fusion
 rerank_score,                   # cross-encoder logit
 score}                          # final: rerank_score if reranked, else rrf_score
```
`05_ask` prints them as `dense#2 bm25#11 fused#4`, so every answer shows how each chunk got there.

### Scripts

| Script | API calls? | What it shows |
|---|---|---|
| `01_bm25.py` | query embeddings only | BM25 internals; BM25 vs dense side by side |
| `02_hybrid_rrf.py` | query embeddings only | RRF worked example; dense vs BM25 vs score fusion vs RRF on the probes |
| `03_filters.py` | query embeddings only | Filter extraction, the Apple and comparison questions, the filing-year pitfall |
| `04_rerank.py` | query embeddings only | Rank movement, gain from reranking, N vs latency, score calibration for the floor |
| `05_ask.py` | yes (chat) | Full pipeline with tracing and both guardrails |

Queries embedded before come from the cache, so re-running is nearly free.

---

## Step 1: BM25, the keyword retriever (`01_bm25.py` + `common/bm25.py`)

**Goal:** a retriever that matches **exact terms**, the thing Phase 0 showed dense embeddings are blind to.

```bash
uv run python -m phase2_better_retrieval.01_bm25
uv run python -m phase2_better_retrieval.01_bm25 "Blackwell"     # your own query
```

### 1a. Tokenization: `tokenize()` (`bm25.py:25`)

```
"Apple's total net sales were $416,161 million per Form 10-K, E-4471"
   → lowercase, curly quote → straight
   → TOKEN_RE (bm25.py:17): [a-z0-9]+ joined by . , - '     keeps "416,161", "10-k", "e-4471", "u.s" whole
   → strip possessive "'s"                                   "apple's" → "apple"
   → drop STOPWORDS (bm25.py:18)                             "the", "were", "per"...
= ['apple', 'total', 'net', 'sales', '416,161', 'million', 'form', '10-k', 'e-4471']
```
- **Why keep punctuation inside tokens:** identifiers are exactly what BM25 is for. Splitting `E-4471` into `e` + `4471` would let it match every "4471".
- **The possessive fix came from a real bug.** The first version kept `apple's` as one token, so queries about "Apple's buybacks" never matched text saying "Apple".
- **No stemming:** "repurchase" and "repurchased" are different tokens. Stemming (a Porter stemmer) would raise recall and lower precision. It's an experiment to try.

### 1b. Indexing: `BM25Index.__init__()` (`bm25.py:31`)

```
for each chunk i:  tokens → Counter → for each (term, tf): postings[term].append((i, tf))
idf[term]  = ln(1 + (N − df + 0.5) / (df + 0.5))          df = len(postings[term])
_norm[i]   = k1 · (1 − b + b · len_i / avgdl)             precomputed per chunk
```
Built in **0.13 s** for 2,344 chunks: vocabulary 13,709 terms, average chunk length 133 tokens (after stopword removal).

### 1c. Scoring: `scores()` (`bm25.py:46`) and `search()` (`bm25.py:57`)

- **Only the postings of the query's terms are touched.** For "Vision Pro" that's 8 + 23 chunks, not all 2,344. That's why inverted indexes scale to billions of documents.
- **A metadata filter** becomes a boolean `mask`: filtered-out chunks get −∞ before top-k (the same idea as Chroma's `where=`).
- **Chunks sharing no term score 0** and are never returned. That's why BM25 returns *fewer* results for paraphrases.

### 1d. What `01_bm25.py` shows

**Explaining one score** (`explain()`, `01_bm25.py:25`):
```
query tokens: ['vision', 'pro']
  vision       df=    8/2344  idf= 5.62
  pro          df=   23/2344  idf= 4.60
top-1: AAPL_FY2025_Item1_000  score=14.37  (doc length 149 tokens, avg 133)
  contribution of 'vision': 5.33
  contribution of 'pro': 9.04
```
"pro" contributes **more** than the rarer "vision", because it appears several times in that chunk (MacBook Pro, iPhone Pro, Vision Pro). That's **term frequency**, with saturation controlled by k1.

**BM25 vs dense, top-3:**

| Query | BM25 | Dense | Winner and why |
|---|---|---|---|
| Vision Pro | AAPL Item 1 ✅ | NVDA Item 1 ❌ | **BM25:** product name; dense reads "vision/pro" as a topic |
| Dojo supercomputer | TSLA (matches "supercomputer") | NVDA | **Neither:** "Dojo" isn't in the corpus (see Step 4) |
| Item 1C cybersecurity | Cover (table of contents) ❌ | Item 1C sections ✅ | **Dense:** keywords also match the TOC text |
| Can the company lose key people? | noise ❌ | AMZN/TSLA key-personnel risks ✅ | **Dense:** paraphrase with no shared words |
| Single supplier for chips | Item 1C ❌ | NVDA/AAPL supply-chain risks ✅ | **Dense:** meaning, not words |

### 📘 Deep dive: BM25 and sparse retrieval

**BM25** ("Best Matching 25", Robertson et al., 1990s) is the default ranking function of Lucene, Elasticsearch and OpenSearch. After 30 years it's still a **strong baseline**. On the BEIR benchmark it beats many dense models on out-of-domain data.

**The formula, piece by piece:**
```
score(q, d) = Σ_{t ∈ q}  idf(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )
```

| Part | Intuition | Knob (ours) |
|---|---|---|
| **idf(t)** = ln(1 + (N − df + 0.5)/(df + 0.5)) | Rare terms carry information ("vision" in 8 of 2,344 chunks: idf 5.62). Common terms carry little ("company" in 523: idf 1.50; "net" in 606: idf 1.35) | none |
| **tf saturation** tf·(k1+1)/(tf + k1·…) | The 1st mention matters most. The 10th adds little. It approaches (k1+1) as tf → ∞, unlike raw TF-IDF, which grows linearly | **k1 = 1.5** (typical 1.2–2.0). 0 means "presence only" |
| **Length normalization** (1 − b + b·\|d\|/avgdl) | A long chunk mentions everything once, so don't reward length itself | **b = 0.75** (0 = off, 1 = full) |

**Sparse vs dense:**

| | BM25 (sparse) | Embeddings (dense) |
|---|---|---|
| Representation | Vocabulary-sized vector, mostly zeros (13,709 dims here) | 1,536 floats, all non-zero |
| Matches | **Exact tokens**: names, IDs, codes, numbers | **Meaning**: paraphrases, synonyms |
| Fails on | Synonyms and paraphrases (vocabulary mismatch) | Exact identifiers, negation, rare names |
| Index | Inverted index (term → postings) | ANN index (HNSW) |
| Training | None | A trained neural model |
| Explainable | Yes: per-term contributions (above) | Not really |
| Cost | Microseconds, CPU | An embedding call per query, plus vector memory |

**Beyond BM25:**
- **Learned sparse retrieval** (SPLADE, ELSER) trains a model to produce sparse term weights, including terms that *aren't in the text* (document expansion). It closes much of the paraphrase gap while keeping an inverted index.
- **In production**, BM25 comes built in: Elasticsearch/OpenSearch, Postgres full-text search (`tsvector`), and hybrid search in Qdrant, Weaviate and Vespa. We wrote it by hand to see the internals.
- **Chroma:** version 1.5.9 (ours) ships sparse-vector types and a BM25 sparse embedding function (`chromadb.utils.embedding_functions.bm25_embedding_function`), so hybrid search can live inside the DB. We wrote BM25 ourselves to see the internals and to control tokenization.

**Interview angle:** *"Dense retrieval misses exact identifiers and rare names, and BM25 misses paraphrases. Their failures don't overlap much, so production RAG runs both (hybrid) and fuses the results."*

---

## Step 2: Hybrid search with RRF (`02_hybrid_rrf.py` + `common/fusion.py`)

**Goal:** combine the two ranked lists into one that keeps the strengths of both.

```bash
uv run python -m phase2_better_retrieval.02_hybrid_rrf
```

### 2a. `rrf()` (`fusion.py:12`)

```
for each ranked list:  for rank, id in enumerate(list, 1):  score[id] += 1 / (60 + rank)
sort by score, descending
```
**Worked example** (from `02`), "Is Tesla too reliant on its CEO?":
```
fused  dense  bm25   rrf score  chunk
    1      5     8     0.03009  TSLA_FY2025_Item1A_054   = 1/(60+5) + 1/(60+8)
    2     16     6     0.02831  TSLA_FY2025_Item8_005
    ...
    6      1    39     0.02649  TSLA_FY2025_Item1A_024   = 1/(60+1) + 1/(60+39)   ← the Musk chunk
```
**The dense retriever had the right answer at #1**, but BM25 put it at #39, so RRF dropped it to **#6**. Chunks that both lists ranked moderately well beat a chunk that only one list ranked first. RRF rewards **agreement**, which is usually good, but not when one retriever is simply wrong for this query.

### 2b. `score_fusion()` (`fusion.py:22`), the alternative

Min-max normalize each list's raw scores to [0,1], then take a weighted sum. It uses score *magnitudes* (a confident #1 counts for more), but it depends on each list's min and max, so one outlier squashes everything else.

### 2c. The numbers (`evaluate()`, `02_hybrid_rrf.py:28`)

```
mode                  hit@5  MRR@5  hit@50  recall@50   keyword/semantic/numeric hit@5
dense                   8/12   0.60   10/12       0.75   3/4  2/4  3/4
bm25                    8/12   0.57   10/12       0.75   4/4  1/4  3/4
score fusion 50/50      9/12   0.59   10/12       0.79   4/4  2/4  3/4
rrf (k=60)              7/12   0.52   11/12       0.88   3/4  1/4  3/4
```
**RRF has the worst top-5 but the best candidate pool:** hit@50 11/12, and **recall@50 0.88** against 0.75 for dense or BM25 alone. hit@50 asks "is *any* relevant chunk in the pool?"; recall@50 asks "what *share* of all relevant chunks is?". The gap between them (11/12 vs 0.88) means some relevant chunks still don't make it in. Fusion's job is to get the right chunk *into the candidate pool*, and ordering is the reranker's job (Step 4). Hybrid without a reranker can be **worse** than dense alone. That's a common real-world surprise, and it's why "just add BM25" isn't a complete answer.

### 📘 Deep dive: Reciprocal Rank Fusion

**Origin:** Cormack, Clarke and Büttcher, SIGIR 2009. It beat learned fusion methods despite having no training at all.

**Why ranks and not scores:**
- BM25 scores run from 0 to ~30 and depend on the query length and vocabulary.
- Cosine similarities sit in a narrow band (0.4–0.8 here) that depends on the model.
- Adding them is meaningless, and normalizing them is fragile (2b).
- **Ranks are comparable across any retrievers.**

**Why k = 60:**
- k damps the top ranks. With k = 60, rank 1 gives 1/61 = 0.0164 and rank 2 gives 1/62 = 0.0161, nearly equal. So **no single list can dominate** with its #1, and a document ranked well by *several* lists wins.
- A small k (e.g. 1) makes #1 worth far more than #2: 1/2 vs 1/3.
- 60 is the paper's empirically robust value. It's rarely tuned.

**Properties:**
- ✅ No training and no score calibration. It works for any number of retrievers: dense + BM25 + multi-query variants (Phase 3).
- ✅ It's the default fusion in Elasticsearch, OpenSearch, Weaviate, Qdrant and Azure AI Search.
- ❌ It throws away confidence. A dense hit at 0.9 and one at 0.5 count the same if they have the same rank.
- ❌ It rewards agreement: a document found by only one retriever (the Musk chunk) gets demoted.
- **Weighted RRF** (w_r / (k + rank)) lets you trust one retriever more.

**Interview angle:** *"I fuse with RRF because BM25 and cosine scores aren't comparable, and rank fusion needs no calibration. But fusion alone can hurt top-k precision, so I treat it as a recall stage and put a cross-encoder reranker after it."*

### 📘 Deep dive: probes, hit@k, MRR and recall@k (`probes.py`)

**Probes:** 12 queries in three kinds (keyword, semantic, numeric). Each lists its relevant chunks as an **id prefix + a regex on the text** (`probes.py`).
- **Every pattern was checked against the chunks.** `relevant_ids()` (`probes.py:33`) raises an error if a probe has no relevant chunk.
- That check matters. I first wanted a "Dojo" probe, then found that "Dojo" isn't in the corpus at all. **Verify ground truth before measuring against it.**

**The metrics** (`hit_and_rr()`, `probes.py:43`):

| Metric | Definition | Answers |
|---|---|---|
| **hit@k** | 1 if *any* relevant chunk is in the top k | "Can the LLM see the answer?" |
| **MRR@k** | Mean of 1/rank of the first relevant chunk (0 if none in the top k) | "How high is it?" Rank 1 = 1.0, rank 2 = 0.5, rank 5 = 0.2 |
| **hit@50** | Any relevant chunk among the 50 candidates? | "Can the reranker possibly fix it?" |
| **recall@k** (`recall_at()`) | Share of *all* relevant chunks in the top k | "Is *all* the evidence there?" Matters for comparisons and multi-part questions |
| **precision@k** | Share of the top k that is relevant | "How much noise goes into the prompt?" Capped at (#relevant / k) when few chunks are relevant |

**Example:** 3 relevant chunks, top 5 = `X R1 Y R2 Z` → hit@5 = 1, RR = 1/2, recall@5 = 2/3, precision@5 = 2/5.

**Caveats:**
- **Twelve probes is a sanity check, not an evaluation.** One query flipping moves hit@5 by 8 percentage points.
- **Relevance is binary** and was judged by one person.

Phase 4 builds a proper golden set, with more queries, graded relevance and nDCG.

---

## Step 3: Metadata filters and fan-out (`03_filters.py` + `common/filters.py`)

**Goal:** use what we *know* about the question (company, fiscal year) to restrict the search, instead of hoping the embeddings figure it out.

```bash
uv run python -m phase2_better_retrieval.03_filters
```

### 3a. Extraction: `extract_filters()` (`filters.py:27`)

```
"What was Apple's total net sales in fiscal 2025?"                  → {'ticker': ['AAPL'], 'fiscal_year': [2025]}
"Compare ... Microsoft and Amazon in their latest fiscal year."     → {'ticker': ['MSFT', 'AMZN']}
"What was AWS operating income in 2025?"                            → {'ticker': ['AMZN']}     ("aws" is an alias)
"Which companies mention export controls?"                          → {}
```
- **`COMPANY_ALIASES`** (`filters.py:15`) maps names and products to tickers ("azure" → MSFT, "iphone" → AAPL).
- **`FY_RE`** (`filters.py:24`) only accepts **fiscal-qualified** years. A bare "in 2025" is ambiguous: it could be the calendar year, the fiscal year, or a year mentioned inside a filing that covers three years.
- **This is rule-based on purpose:** transparent and free, but brittle (it misses "Cupertino company", "the EV maker", or typos). Phase 3 replaces it with LLM-based query understanding.

### 3b. Applying a filter to both retrievers

- **Dense:** `to_chroma_where()` (`filters.py:39`) turns `{'ticker': ['AAPL'], 'fiscal_year': [2025]}` into `{'$and': [{'ticker': {'$in': ['AAPL']}}, {'fiscal_year': {'$in': [2025]}}]}`, which is passed to `col.query(where=...)`.
- **BM25:** `to_mask()` (`filters.py:46`) builds a boolean array over the rows, and `BM25Index.search(mask=...)` sets the rest to −∞.

**The same filter is applied to both sides.** If only one side were filtered, fusion would reintroduce the chunks the filter was meant to exclude.

### 3c. Results

**Apple FY2025 net sales:**
- **Without a filter:** 2 of the top 5 are from FY2024.
- **With the filter:** all 5 are FY2025, and #1 is the right table.

**The comparison question:**
```
no filter               MSFT, MSFT, MSFT, MSFT, MSFT, MSFT
ticker ∈ {MSFT, AMZN}   MSFT, MSFT, MSFT, MSFT, MSFT, MSFT        ← the filter didn't help
fan-out (3 per ticker)  MSFT, AMZN, MSFT, AMZN, MSFT, AMZN
```
**A filter restricts what *may* come back; it doesn't guarantee each value appears.** Microsoft's chunks simply score higher for this wording.

**`retrieve_fanout()`** (`retriever.py:114`) runs one retrieval **per filter value** and interleaves the results, which guarantees coverage. This is the simplest form of **query decomposition**, which Phase 3 generalizes (LLM-generated sub-questions).

**The filing-year pitfall:** "What was Apple's total net sales in fiscal 2023?"
- **With the extracted filter {AAPL, 2023}:** *nothing*. We have no FY2023 filing.
- **With {AAPL} only:** it finds FY2024 and FY2025 filings, and those **contain** the FY2023 column.

The metadata says what year the *document* is, not what years the *facts* cover. The fix is at indexing time: tag each chunk with the periods it covers (Phase 5, table handling). The Apple wrong-year table in Phase 1 was the same problem.

### 📘 Deep dive: metadata filtering in vector search

**Three strategies**, and the trade-off every vector DB has to make:

| Strategy | How | Problem |
|---|---|---|
| **Post-filter** | ANN top-k first, then drop what doesn't match | A restrictive filter can leave **fewer than k** results, or zero, even though matches exist |
| **Pre-filter** | Find the matching IDs first, then brute-force search only those | Exact, but slow if the allowed set is large |
| **Filtered ANN** | Check the filter *during* the graph walk | Efficient, but a very restrictive filter can disconnect the HNSW graph, so the walk misses matches |

**Measured on our Chroma** (`chromadb 1.5.9`), comparing filtered queries with brute force over the same subset:
```
AAPL FY2025 (165 chunks, ~7%)   recall@10 = 1.00
TSLA Item 1C (8 chunks)         recall@8  = 1.00
no filter (2,344 chunks)        recall@10 = 0.99  (one query 0.90)
```
- **With filters we got exact results.** At this scale the filtered sets are small enough to search exactly.
- **Without a filter, HNSW was approximate even at 2,344 vectors** (with ef_search = 100). "Approximate" is real, not theoretical.

**Production vector DBs handle this with query planning.** Qdrant and Weaviate estimate how selective a filter is and switch between filtered HNSW and brute force. pgvector relies on Postgres's planner, and its classic failure is a post-filter on an HNSW index returning too few rows, which is what `hnsw.iterative_scan` was added to address.

**Filters are also a guardrail.** Per-user or per-tenant **access control** is a filter (`where={"tenant": ...}`). It must be enforced *inside retrieval*, never by asking the LLM to ignore documents.

**Interview angle:** *"I extract structured constraints (entity, date) and filter in the vector DB, on both retrievers. I watch for three failure modes: post-filtering starving the top-k, filters that don't guarantee coverage across entities (use fan-out or decomposition), and metadata that describes the document instead of the fact."*

---

## Step 4: Cross-encoder reranking (`04_rerank.py` + `common/rerank.py`)

**Goal:** reorder the ~50 fused candidates by reading each (question, chunk) pair together.

```bash
uv run python -m phase2_better_retrieval.04_rerank
RERANK_MODEL=BAAI/bge-reranker-base uv run python -m phase2_better_retrieval.04_rerank   # bigger model (~1.1 GB)
```

### 4a. `rerank_scores()` (`rerank.py:21`)

```
CrossEncoder(RERANK_MODEL)                              lazy-loaded once (rerank.py:14), ~90 MB for MiniLM-L-6
predict([(query, chunk_1), ..., (query, chunk_50)])     one forward pass per pair, batches of 32
→ 50 raw logits (higher = more relevant)                ms-marco models: no sigmoid, range ≈ −11 … +10
```
In `Retriever.retrieve()` (`retriever.py:83`) every candidate gets a `rerank_score`, the list is re-sorted, and the top k are kept.

### 4b. What it fixes: the Apple question

```
rerank fused   score  chunk
     1     1    5.68  AAPL_FY2025_Item8_049   2025 | 2024 | 2023 Net sales... Total net sales | 416,161   ← correct
     2    12    5.52  AAPL_FY2025_Item8_016   Services net sales ... (net sales by category, FY2025)
     3    37    5.51  AAPL_FY2024_Item8_015   ← fused rank 37 → 3
     ...
```
The Phase 1 problem, the **2023 segment table ranked #1**, is gone. Scored directly, the cross-encoder gives that table −0.36 and the correct table 5.68. **It sees that "fiscal 2025" doesn't fit a table of 2023 figures**, which a single embedding vector can't.

### 4c. Probes: gain and cost

```
                     hit@5   MRR@5   ms/query
hybrid, no rerank     7/12    0.52          8
rerank top-10        10/12    0.69         69
rerank top-50        10/12    0.69        306
rerank top-100       10/12    0.69        594
```
- **The reranker lifts hit@5 from 7 to 10 out of 12, and MRR from 0.52 to 0.69.**
- **Latency grows linearly with N**, the number of candidates reranked. On these probes N = 10 was already enough. On harder queries a larger N gives the reranker more chances. It's a knob you tune on an eval set (Phase 4), trading latency against recall.

**The two remaining misses have different causes** (found by looking at every stage's rank):

| Probe | Right chunk's path | Cause |
|---|---|---|
| "How big is Amazon's workforce?" | In the candidates (fused #15–39), but rerank score **−10.97** | Reranker: the answer sentence comes at the end of a **mixed-topic chunk** (competition, then headcount), and "workforce / how big" vs "employed approximately 1,576,000" is a vocabulary gap for a 22M-parameter model |
| "Apple buybacks in fiscal 2025" | **Not in the top 100** without a filter; fused #26 with one, rerank −5.05 | **Recall** first, then the reranker. The chunk *opens* with interest-rate swaps, and "repurchased 402 million shares for $89.3 billion" comes at the end |

**Truncation isn't the cause:** both chunks are 323–360 word-piece tokens, under the 512 limit. Scoring only the answer sentence raises the scores (−10.97 → −6.46), which confirms that the mixed topics are diluting the chunk. The fix is at indexing time (Phase 5: smaller or parent-child chunks) or in the query (Phase 3: rewriting).

### 4d. Choosing a reranker (measured)

| Model | Params | hit@5 | MRR@5 | ms/query (N = 50) | Score range |
|---|---|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (default) | 22M | **10/12** | 0.69 | **309** | Raw logits, −11 … +10 |
| `BAAI/bge-reranker-base` | 278M | 9/12 | **0.75** | 2,193 | 0 … 1 |

- **bge ranks better on average** (MRR), but it's **7× slower** on a laptop CPU. The hit@5 difference is one query, which is noise at n = 12.
- **Scores aren't comparable across models.** A threshold tuned on one reranker means nothing for another. That's why `05_ask` only applies the floor when `RERANK_MODEL` is the model it was calibrated for.

### 4e. Calibration: a relevance floor

```
in-corpus probes : min= -2.76  median=  4.43  max=  9.38
out-of-corpus    :   1.83  What was Google's advertising revenue in 2025?
out-of-corpus    :  -4.99  Dojo supercomputer
out-of-corpus    : -10.14  What is the capital of France?
out-of-corpus    :  -3.57  How many employees does Netflix have?
out-of-corpus    :   1.99  What was Meta's capital expenditure in 2025?
floor -3.0: keeps 12/12 answerable, blocks 3/5 unanswerable
floor +0.0: keeps 10/12 answerable, blocks 3/5 unanswerable
```
**A floor of −3.0 is free precision**: it doesn't lose a single answerable probe. It **can't** catch the Google or Meta questions:
- The Google question's best chunk (1.83) is from **Apple's** filing and mentions Google (in litigation affecting Apple's revenue). It's on-topic, but about the wrong entity.
- A reranker judges *topical* relevance, not "is this about the company you asked for?".

That's the job of **entity checks on the input** (Phase 3). The LLM's refusal rule still catches these two.

### 📘 Deep dive: cross-encoders and reranking

**Architecture:**
```
bi-encoder:     enc(query) → v_q        enc(chunk) → v_c        score = cos(v_q, v_c)       chunks precomputed
cross-encoder:  enc([CLS] query [SEP] chunk [SEP]) → classification head → one logit        nothing precomputable
```
In a cross-encoder, **attention runs across query and chunk tokens together**. Each query word can attend to each chunk word, so the model can check that "fiscal 2025" matches the column header, that "not" flips the meaning, or which entity a number belongs to. A bi-encoder has to squeeze the whole chunk into one vector *before* it sees the query.

**Training:** our model is fine-tuned on **MS MARCO**, about 500k real Bing queries with human-labeled relevant passages. It's a binary relevant/not-relevant task, which is why it outputs a logit (a sigmoid gives a probability-like number). Being trained on web search explains its weaknesses here: domain vocabulary (10-K tables) and entity awareness.

**Cost model:**
- A bi-encoder does **one** model call per query, plus an ANN search.
- A cross-encoder does **N** calls per query (N = candidates), and nothing can be cached across queries. Our MiniLM: about 6 ms per pair on a laptop CPU, so ≈ 330 ms for 50 pairs.
- That's why it's always the **second stage** over a shortlist.

**The model landscape:**

| Kind | Examples | Notes |
|---|---|---|
| Small cross-encoders | ms-marco-MiniLM-L-6/L-12, TinyBERT | Fast on CPU, English, web-trained |
| Strong open cross-encoders | bge-reranker-base/large, bge-reranker-v2-m3, mxbai-rerank | Multilingual, better quality, GPU-friendly |
| API rerankers | Cohere Rerank, Voyage rerank, Jina | No hosting; data leaves your network |
| LLM rerankers | RankGPT-style listwise ranking, LLM relevance judgments | The most capable, but the slowest and most expensive. Useful in Phase 6 agents |
| Late interaction | ColBERT, ColPali | Per-token vectors, precomputable. Between bi- and cross-encoders in cost and quality |

**Interview angle:** *"Two-stage retrieval: a cheap, high-recall first stage (hybrid, top-50), then a cross-encoder for precision. It reads query and passage jointly, so it handles dates, negation and entity attachment that embeddings blur. It costs one forward pass per candidate, so N is a latency/recall knob I tune on an eval set. Reranker scores are also better calibrated than cosine, which makes a relevance floor possible."*

---

## Step 5: The full pipeline (`05_ask.py`)

```bash
uv run python -m phase2_better_retrieval.05_ask                              # Phase 1 demo questions + 2 more
uv run python -m phase2_better_retrieval.05_ask "your question" --show-context
uv run python -m phase2_better_retrieval.05_ask --mode dense --no-rerank --filters none "..."    # ablation
```

### 5a. `ask()` (`05_ask.py:40`)

```
f = extract_filters(q) if filters == "auto"
if f has >1 ticker:  hits = R.retrieve_fanout(q, "ticker", tickers, k_each=max(2, k//n+1))   (:47)
else:                hits = R.retrieve(q, k, mode, rerank, n_candidates, filters)
relevance floor:     if best rerank score < RELEVANCE_FLOOR (−3.0): refuse, return   (:57)
generate:            phase1.SYSTEM + phase1.format_context(hits) → chat_completion   (:67)
guardrails:          check_answer(answer, passages)   (Phase 1 output checks)        (:78)
```
- **Phase 1's prompt and formatter are imported unchanged** (`importlib`, because the module name starts with a digit). Only retrieval changed between phases, so any difference in answers comes from retrieval. That's how to run an **ablation**.
- **`R.warmup()`** loads the reranker at startup (`retriever.py:33`). See 6b for why.
- **Flags:** `--mode`, `--no-rerank`, `--filters none` and `-n` let you turn each stage off and compare.

### 5b. Results against Phase 1

| Question | Phase 1 | Phase 2 | What made the difference |
|---|---|---|---|
| Apple FY2025 net sales | ✅ (after the footer fix) | ✅ all top 5 from FY2025 | Filter plus reranker |
| NVIDIA export controls | ✅ | ✅ | Both fine |
| MSFT vs AMZN growth | ❌ "I don't know" (5/5 MSFT) | ❌ **answered, but both halves wrong**: MSFT "+16%, $19.2B" (a *segment*; the total is +18%, $50.1B), AMZN "+15%, $36.6B" (the truth is +12%) | Fan-out gave Amazon slots, but see 5c |
| Google ad revenue | ✅ refused by the LLM | ✅ refused by the LLM | The floor can't catch it (on-topic chunks, 1.83) |
| Dojo supercomputer | n/a | ✅ **refused before the LLM** (−4.99 < −3.0) | Relevance floor: no tokens spent |
| Tesla reliant on CEO? | Musk chunk was dense #1 | ✅ answered from the Musk chunks [4][5] | Hybrid kept it in the candidates. The reranker preferred CEO-compensation notes (−2.76), just above the floor |

### 5c. The comparison answer: two wrong numbers, traced to their root causes

**Microsoft (found in Phase 3; the guardrail passed it):** "total revenue increased by $19.2 billion or 16% [1]". [1] says:
```
Reportable Segments / Fiscal Year 2026 Compared with Fiscal Year 2025 / Productivity and Business Processes
Revenue increased $19.2 billion or 16%.
```
That's **one segment's** growth. Microsoft's total is "Revenue increased **$50.1 billion or 18%**" (`MSFT_FY2026_Item7_009`).
- The sentence says just "Revenue increased", and the segment name is a heading a line above it. The LLM attached the number to the wrong entity: the segment instead of the company.
- **The numeric guardrail passed it**, because 19.2 and 16 really are in the passage. This is the "real number, wrong label" false negative from the Phase 1 demo, now in a real answer.
- **Fixes:** contextual chunk headers that say which segment a chunk covers (Phase 5), and a semantic faithfulness check (Phase 4).

**Amazon:**

The comparison answer claimed Amazon's "total revenue increased by $36.6 billion or 15% [6]". The truth, from Amazon's FY2025 income statement: **$637,959M → $716,924M, +12.4%**.

1. **What [6] actually is:** Amazon's **Q1 2026 guidance** ("net sales expected to grow between 11% and 15%"). The LLM took a forecast's upper bound as last year's growth, and **made up** $36.6B.
2. **The guardrail caught half of it:** `ungrounded numbers ['36.6']` ✅. The "15" *does* appear in [6], with a different meaning, so it passed. That's the **false negative the Phase 1 demo predicted** (a real number attached to the wrong fact).
3. **Why retrieval brought guidance instead of the income statement:** vocabulary. The question says "**revenue**"; Amazon says "**net sales**". Every Amazon candidate scored below −2.5. Asked in Amazon's own terms ("Amazon total net sales growth"), the top hit (3.06) is exactly the right table: `Consolidated | 637,959 | 716,924`.
4. **Fixes:**
   - **Phase 3:** rewrite and decompose the question per company, in that company's vocabulary.
   - **Phase 4:** a semantic faithfulness check that would catch the misused 15%.
   - **Phase 5:** route numeric questions to structured data.

This one question passes through **every layer**: fan-out solved coverage, vocabulary broke relevance, the LLM filled the gap, and the guardrail and tracing showed what happened.

### 📘 Deep dive: the relevance floor, a guardrail before generation

**What it is:** if the best reranked chunk scores below a threshold, refuse **before** calling the LLM.

**Why it's worth having:**
1. **Cheaper and faster:** Dojo was refused in 367 ms, with no generation tokens.
2. **Safer:** the LLM never sees irrelevant context it might build a confident answer from.
3. **Measurable:** it's a classifier (answerable / unanswerable) with a threshold you can tune on labeled data.

**Why cross-encoder scores and not cosine:**
- The reranker judges each chunk **against the query**, and its training objective (relevant / not relevant) pushes irrelevant pairs far down (−5 to −10).
- Cosine is a geometric closeness score in a narrow band: Phase 1's off-corpus 0.50 against real hits at 0.65–0.75 was a gap of 0.15, and with bge-small there was no gap at all.

**Limits (measured):** it blocks 3/5 of the unanswerable questions (off-topic ones). It can't block **wrong-entity** questions (Google, Meta), because on-topic chunks exist. Defense in depth covers the rest:
- input entity validation (Phase 3),
- the LLM's refusal rule,
- the output guardrails.

**Calibration hygiene:**
- The floor belongs to **one reranker model**. `05_ask` disables it when `RERANK_MODEL` changes (`FLOOR_MODEL`, `05_ask.py:31`).
- **Re-calibrate** after changing the model, the chunking or the corpus.
- **Pick the threshold from the answerable side** (don't lose real answers), then measure what it blocks. Our −3.0 sits just below the lowest answerable probe (−2.76). The Tesla question in 5b scored −2.76 too, which shows how close to the edge this is.

---

## Step 6: Observability in Phase 2

```bash
uv run python -m phase1_naive_rag.06_traces --name ask_v2            # summary + last 5 timelines
uv run python -m phase1_naive_rag.06_traces --name ask_v2 --last 1
```
The trace viewer from Phase 1 now takes `--name`, and understands the new spans (`top` lists, the floor decision, `refused_by`).

### 6a. Reading a Phase 2 trace

"Is Tesla too reliant on its CEO?":
```
dense_search      61.7 ms   top: TSLA_FY2025_Item1A_024 (0.677) ...       ← dense found the Musk chunk at #1
bm25_search        3.0 ms   top: TSLA_FY2025_Item8_120 (13.43) ...        ← BM25 matched "CEO" in compensation notes
fusion             0.0 ms   top: TSLA_FY2025_Item1A_054 (0.03009) ...     ← agreement wins: Musk chunk now #6
rerank           313.6 ms   top: TSLA_FY2024_Item8_102 (-2.76) ...        ← reranker preferred CEO-award notes
relevance_floor    0.0 ms   best rerank score -2.76 vs floor -3.0 -> blocked=False    ← barely passed
generate        1283.9 ms   tokens in/out: 1868/62
```
**Every stage's opinion is visible.** Without the per-stage spans you'd only know "the answer came from chunks 4 and 5". With them, you know that dense was right, BM25 pulled in noise, and the reranker nearly caused a refusal.

### 6b. Two lessons from the aggregate metrics

**1. Cold start.** The first traced run showed `rerank mean = 2157 ms` and **p95 = 11.9 s**. The first request had paid about 12 s to load torch and the model; steady-state reranking is about 330 ms.
- **Fix:** `Retriever.warmup()` at startup (`retriever.py:33`). The first request's rerank became 379 ms.
- **In production:** load models when the process starts, run a health check that exercises them, and keep the model in memory.

**2. The latency budget moved.** In Phase 1, generation was 77% of the time. Now:

| Stage | Warm latency | Notes |
|---|---|---|
| dense_search | ~60–85 ms | Includes the query embedding (API, or 0.5 ms from cache) |
| bm25_search | ~3 ms | Pure Python + NumPy over 2,344 chunks |
| fusion + filters + floor | ~0 ms | |
| **rerank (N = 50)** | **~330 ms** | CPU-bound; linear in N; 7× more with bge-reranker-base |
| generate | ~1.3–1.5 s | Unchanged |

**Levers:** lower N (10 was enough on the probes, at 69 ms), a GPU, a smaller or quantized reranker, or caching reranks for repeated queries.

---

## What to rerun after a change

| You changed... | Rerun |
|---|---|
| Phase 1 chunks or index | Phase 1 `02` → `03`, then any Phase 2 script (the BM25 index is rebuilt in memory on every start, in 0.13 s) |
| BM25 tokenizer, k1 or b (`common/bm25.py`) | `01`, `02` (probe table) |
| Fusion (`common/fusion.py`) | `02` |
| Filter aliases or regex (`common/filters.py`) | `03`, then `05` |
| `RERANK_MODEL` in `.env` | `04` (re-calibrate, then update `RELEVANCE_FLOOR` and `FLOOR_MODEL` in `05_ask.py`), then `05` |
| Probes (`probes.py`) | `02`, `04` (they raise an error if a probe has no relevant chunk) |

## Debugging a Phase 2 answer

1. **Trace:** `06_traces --name ask_v2 --last 1`.
2. **Right chunk in `dense_search` or `bm25_search`?** If neither has it, it's a **recall** problem: query wording (Phase 3), the filter (too strict?), or chunking.
3. **Survived `fusion`?** It's in the top 50 if both lists had it, or one ranked it highly.
4. **Where did `rerank` put it?** A low score means a reranker limitation: a mixed-topic chunk, vocabulary, or model size.
5. **`relevance_floor` blocked it?** Look at the best score against the floor.
6. **Retrieved, but the answer is wrong?** Generation. Check the guardrail output and the cited passages (the Amazon guidance case).

## What's still broken, and the phase that fixes it

| Failure | Seen in | Fix | Phase |
|---|---|---|---|
| Question vocabulary ≠ document vocabulary ("revenue" vs "net sales") | Amazon comparison, Amazon workforce | Query rewriting, per-entity decomposition, HyDE | 3 |
| Rule-based filters miss paraphrased entities | `filters.py` aliases | LLM query understanding (self-query) | 3 |
| Wrong-entity questions pass the floor | Google, Meta | Input entity validation | 3 |
| Mixed-topic chunks score low in the reranker | Workforce, buybacks | Smaller or parent-child chunks, contextual headers | 5 |
| Metadata = filing year, not fact year | Fiscal 2023 filter | Period tags on chunks, table handling | 5 |
| Misused real numbers pass the numeric guardrail | Amazon "15%" | Semantic faithfulness check | 4 |
| Are these improvements real, or 12-probe noise? | All tables above | A golden set with more queries, nDCG, significance | 4 |

## Glossary (quick reference)

| Term | One-line meaning |
|---|---|
| **Sparse / dense retrieval** | Term-based vectors with an inverted index (BM25) / learned embeddings with an ANN index |
| **Inverted index / postings** | term → list of (doc, tf); query time touches only the query's terms |
| **tf / df / idf** | Term count in a doc / number of docs with the term / log-scaled rarity |
| **k1 / b** | BM25 term-frequency saturation / length-normalization strength |
| **Hybrid search** | Run sparse + dense, then fuse the rankings |
| **RRF** | Reciprocal Rank Fusion: Σ 1/(k + rank). Combines ranks, not scores |
| **Score fusion** | Normalize each list's scores, then take a weighted sum |
| **Metadata filter** | Restrict search to chunks whose fields match (`where=`, mask) |
| **Pre / post / filtered ANN** | Filter before / after / during the vector search |
| **Fan-out** | One retrieval per filter value (per company), results merged, guaranteeing coverage |
| **Cross-encoder** | Model scoring (query, passage) jointly; accurate, one call per pair |
| **Two-stage retrieval** | Cheap high-recall stage → expensive high-precision rerank |
| **N (candidates)** | How many fused results the reranker sees; a latency/recall knob |
| **Relevance floor** | Refuse before generation if the best rerank score < threshold |
| **Calibration** | Choosing a threshold from labeled score distributions, per model |
| **hit@k / MRR** | Any relevant chunk in the top k / mean of 1/rank of the first relevant chunk |
| **recall@k / precision@k** | Share of all relevant chunks found in the top k / share of the top k that is relevant |
| **Ablation** | Turn one component off, keep everything else fixed, measure the difference |
| **Cold start** | The first request paying one-time setup costs (model loading) |

## Self-check before moving on
- [ ] Why does "pro" contribute more than "vision" to the Vision Pro score, even though "vision" is rarer?
- [ ] Hybrid without reranking scored *worse* than dense alone. Explain why using the RRF worked example.
- [ ] Why can't you add a BM25 score to a cosine similarity? What does RRF do instead?
- [ ] The `ticker ∈ {MSFT, AMZN}` filter didn't fix the comparison question. Why, and what did?
- [ ] Why does "fiscal 2023" return nothing with the "correct" filter? Where should the fix go?
- [ ] What can a cross-encoder see that a bi-encoder can't? Use the Apple 2023 table.
- [ ] Why is the relevance floor tied to one reranker model? Why can't it catch the Google question?
- [ ] Walk the Amazon "15%" error from question to answer: which stage failed, which guardrail caught what, and which phase fixes it?
- [ ] p95 was 11.9 s in the first run and fine afterwards. What happened, and what's the production fix?
