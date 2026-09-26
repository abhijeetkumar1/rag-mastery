# Phase 0 walkthrough: foundations

Read this top to bottom and run each command as you reach it. The theory and interview Q&A are in `NOTES.md`. This file covers **how the code works, step by step**.

## Architecture

Phase 0 has no pipeline. It's three standalone experiments that build the mental model every later phase relies on: *text → vector → similarity → nearest-neighbour search*.

```
 01_similarity            02_embeddings                   03_tiny_vector_search
 ─────────────            ─────────────                   ─────────────────────
 hand-made vectors        7 sentences                     7 FAQ docs + 3 queries
       │                       │                                │
       ▼                       ▼                                ▼
 dot / cos / L2         common.llm.embed()  ───────────►  TinyVectorStore
 normalization          (OpenAI or local HF,               add(): embed + stack into matrix
 curse of dimensionality cached on disk)                   search(): matrix @ q → top-k
       │                       │                                │
       ▼                       ▼                                ▼
 "which metric?"        "where do embeddings fail?"       "why do vector DBs exist?"
                                                          (brute force vs scale → ANN)
```

The one piece of shared infrastructure is `common/llm.py → embed()`. Every later phase uses it.

| File | Needs API/model? | Teaches |
|---|---|---|
| `01_similarity.py` | No, pure NumPy | Metrics, normalization, high-dimensional geometry |
| `02_embeddings.py` | Yes (`embed`) | What embeddings capture, and what they miss |
| `03_tiny_vector_search.py` | Yes (`embed`) | Retrieval = nearest neighbours; brute-force cost |

---

## Shared infrastructure: config and `embed()`

**`common/config.py`** loads `.env` once and exposes:
- `EMBED_PROVIDER`: `openai` or `hf`. The repo default is `hf`; your `.env` sets `openai`.
- `EMBED_MODEL`: resolves to `OPENAI_EMBED_MODEL` (`text-embedding-3-small`) or `HF_EMBED_MODEL` (`BAAI/bge-small-en-v1.5`), depending on the provider.
- `CHAT_MODEL`, `OPENAI_API_KEY`, `HF_TOKEN`, `CACHE_DIR`, `DATA_DIR`.

Rule: **no model name is hardcoded in scripts.** Change `.env`, not code.

**`common/llm.py → embed(texts)`** (`llm.py:50`):

```
texts ──► for each: key = sha256(f"{model}::{text}")
            ├─ .cache/embeddings/<key>.json exists  → load it (cache hit)
            └─ else → add to misses
misses ──► batches of 256 ──► _embed_batch()
                                ├─ provider "openai": client().embeddings.create(model, input=batch)
                                └─ provider "hf":     SentenceTransformer(model).encode(batch, normalize_embeddings=True)
         ──► write each new vector to the cache
return np.float32 matrix of shape (n, d)     d = 1536 (OpenAI) or 384 (bge-small)
```

Why it's built this way:
- **Cache key = model + text.** The same text under a different model gets a different key, so vectors from different models never mix.
- **Batching:** one API call per 256 texts instead of one per text. That means far fewer round trips and fewer rate-limit problems.
- **Normalization:** OpenAI vectors come unit-length, and HF vectors are normalized explicitly, so **dot product = cosine** everywhere downstream.
- **Lazy clients:** the OpenAI client is created only on first use (and checks for the key then). The HF model is loaded once and kept in memory (`_hf_models`), because loading torch is slow.

---

## Step 1: Similarity metrics (`01_similarity.py`)

```bash
uv run python -m phase0_foundations.01_similarity
```

### Part A: three metrics on raw vectors (`:8-31`)

Functions: `dot(a,b)`, `cosine(a,b) = a·b / (|a||b|)`, `euclidean(a,b) = |a−b|`.

Vectors: `a = [1,2,3]`, `b = 2a` (same direction, twice as long), `c` (a different direction).

```
a vs b (same dir)    dot= 28.000  cos= 1.000  euclid= 3.742
a vs c (diff dir)    dot=  2.500  cos= 0.209  euclid= 4.387
```
**Read it:** a and b point the same way, so cosine is exactly 1.0. Dot and Euclidean are both affected by b being twice as long. **Cosine measures only direction; dot and L2 also depend on length.**

### Part B: after L2-normalizing (`:36-41`)

`normalize(v) = v / |v|` gives every vector length 1.
```
a vs b   dot= 1.000  cos= 1.000  euclid= 0.000  check: euclid^2 = 2 - 2*cos -> 0.000 == 0.000
a vs c   dot= 0.209  cos= 0.209  euclid= 1.258  check: euclid^2 = 2 - 2*cos -> 1.583 == 1.583
```
**Read it:** on unit vectors, **dot = cosine**, and **‖a−b‖² = 2 − 2·cos**. All three metrics therefore give the **same ranking**. That's why vector DBs can use plain dot product, the cheapest operation, on normalized embeddings.

### Part C: the curse of dimensionality (`:47-53`)

Generate 500 random unit vectors in d dimensions and measure the cosine between every pair:
```
dim=    2  mean cos=+0.002  std=0.707
dim=   10  mean cos=-0.000  std=0.316
dim=  100  mean cos=+0.000  std=0.100
dim= 1536  mean cos=-0.000  std=0.026
```
**Read it:** as d grows, random vectors become almost exactly perpendicular (cos ≈ 0), with a spread of about 1/√d. In 1536 dimensions, *unrelated* texts cluster tightly around a base level, and *related* texts are only somewhat higher. **Scores sit in a narrow band, so fixed thresholds break easily. Rank with top-k instead.** Phase 1 confirmed this with real numbers: real hits scored 0.65–0.75, and an off-corpus question scored 0.50.

---

## Step 2: Real embeddings and their failure modes (`02_embeddings.py`)

```bash
uv run python -m phase0_foundations.02_embeddings
EMBED_PROVIDER=hf uv run python -m phase0_foundations.02_embeddings    # compare with local bge-small
```

### Flow

```
7 hand-picked sentences ──► embed() ──► (7, 1536) matrix ──► sim = vecs @ vecs.T (7×7 cosine matrix)
                                                          └─► show(i, j) prints chosen pairs
```

Each sentence pair is chosen to test one property:

| Pair (sentences) | Tests | text-embedding-3-small | bge-small |
|---|---|---|---|
| "revenue grew 12%" vs "sales increased twelve percent" | Paraphrase with no shared words | **0.763** | 0.807 |
| "revenue grew 12%" vs "revenue declined 12%" | Opposite meaning, same words | **0.702** | 0.822 |
| "Apple released iPhone" vs "ate an apple" | Word-sense ambiguity | **0.281** | 0.505 |
| "E-4471 disk full" vs "E-4417 network timeout" | Exact identifiers | **0.630** | 0.747 |

(The printed list numbers sentences from 1. The code indexes from 0, so `show(0, 1)` means sentences 1 and 2.)

**Read it:**
1. **Paraphrases work.** Both models score them high with no shared keywords. This is the reason dense retrieval exists.
2. **Negation blindness.** "Grew" and "declined" score almost as high as the true paraphrase. With bge-small they score **higher** (0.822 against 0.807). Embeddings encode topic much more strongly than polarity or numbers.
3. **Context disambiguates word sense.** Apple the company and apple the fruit score low, especially with OpenAI.
4. **Exact-ID blindness.** Two different error codes score 0.63–0.75. To an embedding model `E-4471` and `E-4417` are both "some error code". **Fix: BM25 keyword search in a hybrid setup (Phase 2).**

### Part 2: Matryoshka truncation (`:39-46`)

Take the first d dimensions, re-normalize, and recompute the similarities:
```
dim=1536  paraphrase=0.763  opposite=0.702  fruit-vs-company=0.281
dim= 256  paraphrase=0.783  opposite=0.753  fruit-vs-company=0.311
dim=  64  paraphrase=0.779  opposite=0.795  fruit-vs-company=0.227
```
**Read it:** text-embedding-3 models are trained so that the **first dimensions of the vector are a usable embedding on their own** (Matryoshka representation learning). At 256 dimensions the index is 6× smaller with only a small quality change. At 64 dimensions the *opposite* pair overtakes the paraphrase, so real distinctions are being lost. bge-small wasn't trained this way, so it degrades faster. With the API you'd pass `dimensions=256` to get this directly.

---

## Step 3: A vector database in 30 lines (`03_tiny_vector_search.py`)

```bash
uv run python -m phase0_foundations.03_tiny_vector_search
```

### `TinyVectorStore` (`:12-27`)

```
add(texts):     vecs = embed(texts)
                self.texts += texts
                self.matrix = vstack([matrix, vecs])       # (N, d), one row per document

search(q, k):   qv = embed([q])[0]                          # (d,)
                scores = matrix @ qv                        # (N,): N dot products = cosines
                top = argpartition(-scores, k)[:k]          # O(N): unordered top-k
                top = top[argsort(-scores[top])]            # sort only those k
                return [(score, text), ...]
```
This is **exact k-nearest-neighbour search**. `argpartition` is a small but real optimization: it finds the top-k in O(N) instead of sorting all N scores in O(N log N).

**Note:** this script calls `embed([query])` directly. It doesn't use `embed_query()`, which was added in Phase 1 to put BGE's query prefix in front. With OpenAI that makes no difference; with bge-small the queries would score a bit lower. Phase 1 always uses `embed_query()`.

### Part A: semantic search works
```
Q: Can I get my money back?          0.357  Our refund policy allows returns within 30 days...
Q: How long until my package ships?  0.443  Orders are usually dispatched within 2 business days.
Q: Is my data safe?                  0.433  We store customer data in encrypted form in AWS...
```
**Read it:** every top-1 is correct, even though "money back" never appears next to "refund" and "safe" never appears next to "encrypted". The absolute scores are low (0.36–0.44), but the **ranking** is right. That's Step 1 Part C in practice.

### Part B: brute force at scale (`:49-57`)
```
N= 10,000  memory=0.06 GB  search=4.2 ms
N=100,000  memory=0.61 GB  search=142.5 ms
N=300,000  memory=1.84 GB  search=6722.0 ms
```
**Read it:**
- **Memory = N × d × 4 bytes.** 10M chunks × 1536 dimensions is about 61 GB of raw floats.
- **Time grows at least linearly with N.** Going from 100k to 300k was about 47× slower for 3× the data. Once the matrix no longer fits in the CPU caches, and your machine starts swapping memory, it's much worse than linear.
- **Conclusion:** ANN indexes exist because of this. **HNSW** (graph based, the default in Chroma, Qdrant and pgvector), **IVF** (clustering) and **PQ** (compression) give up a little recall to get sub-linear search. Phase 1 uses Chroma's HNSW. See NOTES.md §5 for the comparison table.

---

## How Phase 0 feeds the later phases

| Phase 0 finding | Where it comes back |
|---|---|
| Normalized vectors → dot = cosine | Chroma collection uses `hnsw:space=cosine` (Phase 1) |
| Scores in a narrow band → no fixed thresholds | Off-corpus Google question: 0.50 vs 0.65–0.75 (Phase 1). Reranker scores (Phase 2) |
| Exact-ID blindness | BM25 + dense hybrid search (Phase 2) |
| Negation/number blindness | Cross-encoder reranking (Phase 2). The Apple wrong-year table (Phase 1) |
| Paraphrase strength, query/doc mismatch | Query rewriting, HyDE (Phase 3) |
| Brute force is O(N) | HNSW in Chroma (Phase 1). Index tuning and pgvector/Qdrant (Phase 7) |
| Cache by model + text | Re-index after parser fixes took 5 s instead of 21 s (Phase 1) |

## Self-check before moving on
- [ ] Why does cosine give 1.0 for `a` and `2a` while Euclidean distance doesn't?
- [ ] On unit vectors, derive ‖a−b‖² = 2 − 2cos.
- [ ] Why is "similarity > 0.8" a bad retrieval rule? Use the dimensionality numbers.
- [ ] Name two queries where dense retrieval would fail on your 10-K corpus, and say why.
- [ ] Estimate the brute-force memory for 5M chunks with 1536-d float32 vectors (≈ 30.7 GB), and say what you'd do about it.
