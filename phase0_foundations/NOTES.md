# Phase 0: Foundations

## 1. What is RAG and why does it exist?

An LLM only knows what was in its training data, frozen at a cutoff date, and it has no access to your private data. RAG (Retrieval-Augmented Generation) works around this at inference time:

```
query ──► retrieve relevant chunks from YOUR corpus ──► stuff them into the prompt ──► LLM answers grounded in them
```

**RAG compared with fine-tuning** (this comes up in every interview):

| | RAG | Fine-tuning |
|---|---|---|
| Adds new *knowledge* | ✅ Instantly, by updating the index | ❌ Poorly; models memorize facts unreliably |
| Changes *behavior*/style/format | ❌ | ✅ |
| Cites sources | ✅ | ❌ |
| Data freshness | Minutes | Needs retraining |
| Access control per user | ✅ Filter at retrieval time | ❌ Baked into weights |
| Cost | Index + longer prompts | Training runs |

**One-line answer:** "Fine-tuning teaches the model *how* to respond, and RAG gives it *what* to know. They are complementary."

**RAG compared with long context** (a 1M-token window): long context is simpler, but it costs more per query, it's slower, and quality degrades in the middle of the window ("lost in the middle"). It also doesn't scale to a 10 GB corpus. RAG is essentially a relevance filter in front of the context window.

## 2. Embeddings

An embedding model maps text to a fixed-length vector (1536 dimensions for `text-embedding-3-small`) so that **semantically similar texts land close together**. The models are trained contrastively: pairs such as (question, answer) or (paraphrase, paraphrase) are pulled together, and random pairs are pushed apart.

- **Bi-encoder:** the query and the document are embedded *independently*. That's why documents can be pre-computed and indexed. It's fast but coarse.
- **Cross-encoder:** the query and the document are fed *together* into one model, which outputs a relevance score. It's much more accurate, but you can't pre-compute anything, so it's O(N) model calls. That's why it's used only to **rerank** the top-50 results (Phase 2).

## 3. Similarity metrics (see `01_similarity.py`)

- **Cosine** = angle between vectors. It ignores length.
- **Dot product** = cosine × the two lengths.
- **Euclidean (L2)** = straight-line distance.
- **On unit-normalized vectors all three give the same ranking.** Dot equals cosine, and ‖a−b‖² = 2 − 2·cos. OpenAI embeddings are normalized, so use dot product because it's the cheapest.

**The curse of dimensionality:** random high-dimensional vectors are nearly orthogonal (cos ≈ 0). Real similarity scores therefore sit in a narrow band. **Prefer top-k over hard thresholds**, or calibrate a threshold on your own data.

## 4. Where dense retrieval fails (see `02_embeddings.py`)

1. **Exact identifiers** such as error codes, SKUs, names and IDs. `E-4471` and `E-4417` look nearly the same to an embedding model. The fix is BM25/keyword search, combined in hybrid search (Phase 2).
2. **Negation and numbers.** "grew 12%" and "declined 12%" score as similar. The fix is a reranker, or letting the LLM read carefully.
3. **Domain vocabulary** the model never saw. The fix is hybrid search or a fine-tuned embedding model.
4. **Query/document asymmetry.** A short question doesn't look like a long passage. The fix is HyDE or query rewriting (Phase 3).

**Matryoshka embeddings:** `text-embedding-3-*` vectors can be truncated to 256 or 512 dimensions with a small quality loss, for a 3–6× cheaper index. You pass `dimensions=` to the API or slice and re-normalize yourself.

## 5. Vector search at scale (see `03_tiny_vector_search.py`)

Brute force is **exact** but costs O(N·d) per query, and memory is N × d × 4 bytes (10M × 1536 floats ≈ 61 GB). **ANN (Approximate Nearest Neighbor)** indexes trade a little recall for speed:

| Index | Idea | Trade-off |
|---|---|---|
| **HNSW** | Multi-layer proximity graph. Greedy-walk from coarse to fine layers. | Best recall/latency. RAM-heavy. Slow inserts. Tuned with `M` and `ef_search`. It's the default in Chroma, Qdrant, pgvector and Weaviate. |
| **IVF** | k-means clusters. Only the `nprobe` nearest clusters are searched. | Less memory. Recall depends on `nprobe`. Needs training. |
| **PQ** (product quantization) | Compress vectors into codes (roughly 1536×4 bytes down to ~64 bytes). | Huge memory savings. Lossy. Often combined as IVF-PQ. |
| **Flat** | Brute force | Exact. Fine up to about 100k vectors. |

**Interview framing:** "ANN recall@10 is typically 95–99%. You tune `ef_search` or `nprobe` to trade latency for recall, and you measure it against a brute-force baseline."

---

## Interview Q&A

**Q: Why not just fine-tune the LLM on our documents?**
Fine-tuning is unreliable for injecting facts: the model hallucinates details it half-learned. It can't cite sources, it goes stale, and it can't enforce per-user permissions. RAG solves all four. Fine-tune for style or format, and use RAG for knowledge.

**Q: Cosine or dot product?**
For normalized embeddings they are identical, so use dot because it's cheaper. If the vectors aren't normalized, dot product rewards long vectors, which can bias results toward long documents. Normalize first.

**Q: How do you pick an embedding model?**
Check the MTEB leaderboard for your task type (retrieval), then **evaluate on your own data**, because leaderboard rank doesn't transfer reliably. Also weigh dimension size (storage cost), max input tokens, multilingual support, latency, cost, and whether you can self-host for data privacy.

**Q: What happens if you change embedding models?**
You have to re-embed the whole corpus, because vectors from different models live in incompatible spaces. Store the model name alongside each vector. That's why the cache in `common/llm.py` keys on `model + text`.

**Q: What is HNSW, in simple terms?**
It's a layered graph like a skip list. The top layers hold a few nodes with long-range links, and the bottom layer holds every node with local links. A search greedily hops toward the query at each layer and drops down a level. That gives roughly O(log N) search.

**Q: Why is "similarity score > 0.8" a bad retrieval rule?**
Score distributions depend on the model and the domain, and in high dimensions they are compressed into a narrow range. Use top-k plus a reranker, or calibrate the threshold on labeled data.

## Experiments to try
- [ ] Add your own sentence pairs to `02_embeddings.py` and find a case that beats your intuition.
- [ ] Rerun `03` with `k=1` and a deliberately vague query. When does top-1 go wrong?
- [ ] Pass `dimensions=256` to the embeddings API and compare against the manual truncation in `02`.
