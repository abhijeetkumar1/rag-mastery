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

### 📘 Deep dive: what an embedding model is and how it's trained

An **embedding** is a fixed-length list of numbers (a vector) that represents a text's meaning. Similar meanings → nearby vectors.

**Architecture:** a Transformer **encoder** (BERT-style) reads the tokens and produces one vector per token. **Pooling** turns them into one vector per text:
- **CLS pooling** takes the output of a special first token (BGE).
- **Mean pooling** averages all the token vectors (MiniLM).

The vector is then usually **L2-normalized** to length 1.

**Training (contrastive learning):**
```
batch of pairs:  (q1, p1)  (q2, p2)  (q3, p3) ...      q = query/sentence, p = its matching passage
loss (InfoNCE):  make cos(q1, p1) high, and cos(q1, p2), cos(q1, p3), ... low   ← in-batch negatives
```
- **Training data:** millions of pairs (search queries with clicked results, question/answer pairs, title/body, paraphrases).
- **Hard negatives:** passages that look similar but are wrong. They teach fine distinctions.
- **What this explains** about the failure modes in Step 2:
  - Models learn **topic** extremely well, because most training pairs share a topic.
  - They learn **negation, numbers and exact IDs** poorly, because few training pairs hinge on them.

**Bi-encoder vs cross-encoder:**
- An embedding model is a **bi-encoder**: query and document are encoded *separately*, so documents can be embedded once, ahead of time.
- A **cross-encoder** reads the query and document *together* and outputs a relevance score. It's more accurate, but nothing can be precomputed, so it's used only to rerank a shortlist (Phase 2).

### 📘 Deep dive: the two embedding backends

**OpenAI Embeddings API**:
- **Call:** `client.embeddings.create(model="text-embedding-3-small", input=[...up to 2048 texts...])`. It returns `data[i].embedding`, a list of 1536 floats, already normalized.
- **Limits:** max 8,191 tokens per input; price $0.02 per 1M tokens.
- **`dimensions=`:** asks the server for a shorter vector (Matryoshka, Step 2).
- **Batching:** one call carries many texts, which is why `embed()` sends batches of 256.

**sentence-transformers (local)**:
- **Setup:** `SentenceTransformer("BAAI/bge-small-en-v1.5")` downloads the weights (about 130 MB) from the Hugging Face Hub to `~/.cache/huggingface`, then runs on CPU or Apple's MPS GPU.
- **`encode(texts, normalize_embeddings=True)`:** tokenize, then Transformer forward pass, then pooling, then normalization.
- **Trade-off:** free and private, but slower on a laptop. bge-small has 33M parameters, 384 dimensions and a 512-token limit.

**The cache** (`.cache/embeddings/<sha256>.json`):
- **SHA-256** turns `model::text` into a fixed 64-character hex key: the same input always gives the same key, and different inputs practically never collide.
- **Why it's safe:** an embedding is a pure function of (model, text), so a cached vector is always valid. Switching models changes the key.
- **Production version:** the same idea, stored in Redis, a KV store, or alongside the chunk record.

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

### 📘 Deep dive: NumPy vectorization

- **NumPy stores arrays as contiguous typed memory** (float32 = 4 bytes per number), so a (N, d) matrix is one block of N·d·4 bytes.
- **`a @ b` / `np.dot`** hand the work to **BLAS**, a highly optimized linear-algebra library (Apple Accelerate on macOS, OpenBLAS elsewhere). It uses SIMD instructions and multiple cores.
- **The payoff:** `matrix @ q` over 10,000 × 1536 floats takes a few milliseconds. A Python loop over the same numbers would take seconds.
- **`np.linalg.norm(v)`** = √(Σvᵢ²), the L2 length. `v / norm(v)` gives it length 1.
- **Why float32:** embeddings don't need float64 precision, and float32 halves memory and bandwidth. Vector DBs go further with float16, int8 or binary quantization (Phase 7).

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

### 📘 Deep dive: why random vectors become orthogonal (concentration of measure)

The cosine of two random unit vectors is an average of d small independent products. By the law of large numbers it concentrates around 0, with **standard deviation ≈ 1/√d**. Your output matches this exactly: d=10 → 0.316 (1/√10), d=100 → 0.100, d=1536 → 0.026 (1/√1536 = 0.0255).

**What it means for RAG:**
- **Real embeddings aren't random.** They share common directions (anisotropy), so even unrelated texts score around 0.1–0.5, depending on the model.
- **The usable range is narrow.** Unrelated texts cluster in a tight band, related texts sit only somewhat above it, and **each model has its own band**. In Phase 1, bge-small put real hits at 0.72–0.83, while OpenAI put them at 0.65–0.75.
- **So:** rank by top-k, and calibrate any threshold per model on labeled data.

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

### 📘 Deep dive: Matryoshka representation learning (MRL)

A normal embedding spreads information across all dimensions, so cutting it in half destroys it. **MRL** changes the training objective: the loss is computed on several **prefixes** of the vector at once (e.g. the first 64, 128, 256, 512 and 1536 dimensions), so the most important information is packed into the first dimensions. It's named after nested Russian dolls.

- **Supported:** OpenAI text-embedding-3 (pass `dimensions=256`), nomic-embed-v1.5, and some newer open models.
- **Not supported:** bge-small and MiniLM, so truncating them degrades faster.
- **Why use it:** a 256-d index is **6× smaller** and faster to search. A common pattern is **"funnel search"**: shortlist with short vectors, then re-score with the full vectors.
- **Rule:** always **re-normalize** after truncating, because the prefix of a unit vector isn't a unit vector.

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

### 📘 Deep dive: `argpartition` vs sort

- `np.argsort(scores)` fully sorts N scores: O(N log N).
- `np.argpartition(-scores, k)` uses **introselect** to put the k largest in the first k positions, *unordered*, in O(N).
- Then only those k get sorted: O(k log k).
- For N = 1M and k = 5, that's about 1M operations instead of about 20M. Real vector DBs do the equivalent with a **bounded min-heap** of size k while scanning.

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

### 📘 Deep dive: why brute force is slow at scale

**Bytes:** each query must read the **whole matrix**: N × d × 4 bytes. At 300k × 1536 that's 1.84 GB *per query*.

**Memory bandwidth, not arithmetic, sets the speed:**
- A laptop reads RAM at about 50–100 GB/s, which is about 20–40 ms per query at 1.84 GB.
- Your 300k run took 6.7 s. The script creates the random data in float64 first (3.7 GB) and then converts it, so memory pressure and swapping are the likely cause.
- At that scale you're limited by memory, not CPU.

**Fixes, from simplest to most involved:**
1. **Smaller vectors:** Matryoshka truncation, float16/int8 quantization.
2. **Don't read everything:** ANN indexes (HNSW, IVF). See the HNSW deep dive in `phase1_naive_rag/WALKTHROUGH.md`.
3. **Compress:** product quantization (PQ).
4. **Spread the index:** sharding across machines.

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

## Glossary (quick reference)

| Term | One-line meaning |
|---|---|
| **Embedding** | Fixed-length vector representing a text's meaning |
| **Encoder / pooling** | Transformer that produces per-token vectors / combining them into one (CLS or mean) |
| **Contrastive learning / InfoNCE** | Training that pulls matching pairs together and pushes others apart |
| **Bi-encoder / cross-encoder** | Encode query and document separately (fast, indexable) / together (accurate, rerank only) |
| **L2 norm / normalization** | Vector length √Σv² / scaling to length 1 |
| **Dot / cosine / Euclidean** | Similarity metrics. Identical ranking on unit vectors |
| **Curse of dimensionality** | Random high-dimensional vectors are nearly orthogonal; scores concentrate |
| **Matryoshka (MRL)** | Training so vector prefixes are usable embeddings |
| **Exact k-NN / ANN** | Brute-force nearest neighbours / approximate but sub-linear (HNSW, IVF, PQ) |
| **BLAS** | Optimized linear-algebra library behind NumPy's `@` |
| **argpartition** | O(N) selection of the top-k without a full sort |

## Self-check before moving on
- [ ] Why does cosine give 1.0 for `a` and `2a` while Euclidean distance doesn't?
- [ ] On unit vectors, derive ‖a−b‖² = 2 − 2cos.
- [ ] Why is "similarity > 0.8" a bad retrieval rule? Use the dimensionality numbers.
- [ ] Name two queries where dense retrieval would fail on your 10-K corpus, and say why.
- [ ] Estimate the brute-force memory for 5M chunks with 1536-d float32 vectors (≈ 30.7 GB), and say what you'd do about it.
