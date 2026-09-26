# Phase 1: Naive RAG

```
00_download  EDGAR ──► data/raw/*.html (10 filings, 5 companies × 2 years) + manifest.json
01_parse     HTML ──► clean text, split by SEC "Item" sections ──► data/processed/*.json
02_chunk     sections ──► ~350-token chunks + metadata ──► data/processed/chunks_*.jsonl
03_index     chunks ──► embed (bge-small, cached) ──► Chroma (HNSW, cosine) at data/chroma/
04_ask       question ──► embed query ──► top-k ──► prompt with numbered passages ──► answer with [n] citations
```

Run them in order:
```bash
uv run python -m phase1_naive_rag.00_download_10k   # once
uv run python -m phase1_naive_rag.01_parse
uv run python -m phase1_naive_rag.02_chunk
uv run python -m phase1_naive_rag.03_index
uv run python -m phase1_naive_rag.04_ask                # demo questions
uv run python -m phase1_naive_rag.04_ask "your question" -k 8 --show-context
```

"Naive" means one embedding query, top-k, stuff the results into the prompt, and generate. There is no query rewriting, keyword search, reranking or filtering. Every later phase fixes one of the failures listed in section 6.

## 1. Ingestion: garbage in, garbage out (see `01_parse.py`)

Most real RAG quality problems start **before** the vector DB. A 10-K's HTML is 1.5–8.6 MB, but only about 200–400K characters of it are text. What we had to deal with:

| Problem | What happens if you ignore it | Fix |
|---|---|---|
| `<ix:header>` inline-XBRL block | Thousands of hidden machine tags get chunked and retrieved as noise | Drop it before extracting text |
| Headings split across `<span>`s (`ITEM 1. B` + `USINESS`) | Section detection fails | Join inline text and break lines only on block tags |
| Table of contents and running page headers look like headings | Every "Item 1" page header opens a new section | Require a title after the item number, and treat a row ending in a page number as a TOC row |
| Different filer layouts (AMZN puts headings in tables) | 0 sections found for AMZN | Handle the `Item 1. \| Business` row form |
| Financial tables | `get_text()` spreads each cell onto its own line, so numbers lose their labels | Flatten each row to `label \| v1 \| v2` |

**Takeaway:** always *look at* your parsed text. The AMZN bug was found by printing section counts per file, not by any metric. In production you would use a proper parser (Unstructured, Docling, LlamaParse, or the XBRL data itself for numbers) and still spot-check the output.

**Layout quirk:** NVDA has no Item 8. Its financial statements live under Item 15, so an "Item 8 only" filter would silently lose NVIDIA's financials. Metadata is only as good as your understanding of the documents.

## 2. Chunking (see `02_chunk.py`, `common/chunking.py`)

Why chunk at all? Embedding models have input limits, one vector for a 100-page document is a blurry average, and the LLM context should hold only the relevant parts.

| Strategy | How | Pros | Cons |
|---|---|---|---|
| **Fixed-size tokens** | Slide an N-token window with overlap | Simple, predictable sizes | Cuts mid-sentence and mid-table-row |
| **Recursive** (our default) | Split on the coarsest separator that fits (paragraph → line → sentence → word), then pack pieces up to N | Respects natural boundaries | Uneven sizes |
| **Structure-aware** | Split on document structure (headings, Items) first | Chunks never mix sections, and section metadata comes free | Needs a reliable parser |
| **Semantic** | Break where the embedding similarity of adjacent sentences drops | Topic-coherent | Costly, and on real data often no better than recursive |

We combine **structure-aware** (never cross an Item) with **recursive** (within an Item). Results on this corpus:

| | chunks | mean tokens | max |
|---|---|---|---|
| recursive 350/50 | 2,366 | 286 | 350 |
| fixed 350/50 | 2,201 | 338 | 350 |

**Chunk-size trade-off:** small chunks give precise vectors and match specific questions, but each carries little context (a table row without its header). Large chunks give rich context but a diluted vector, and fewer of them fit in the prompt. The usual range is 200–800 tokens. **Tune it with an eval set (Phase 4), not by intuition.**

**Overlap** protects a fact that straddles a boundary. The cost is duplicate content in the index and in the prompt. Typical overlap is 10–20% of the chunk size.

**The embedding model's token limit, a silent bug:** bge-small has a limit of **512 of its own tokens**, and anything longer is **silently truncated**. The model doesn't raise an error; the tail of the chunk simply isn't in the vector. `02_chunk.py` checks with the embedder's own tokenizer: 0 of our chunks are truncated at 350 tiktoken tokens. OpenAI's text-embedding-3 accepts 8,191 tokens.

**Metadata:** each chunk carries `ticker, fiscal_year, item, section, url`. We don't use it for retrieval yet. Phase 2 uses it for filters, and we already use it for citations.

## 3. The vector store (see `03_index.py`, `common/store.py`)

We compute embeddings ourselves and hand Chroma raw vectors. Chroma is then just **storage + an ANN index (HNSW) + metadata filtering**. Keeping it that thin means:
- we control batching and caching (a re-index costs 0 embedding calls);
- the DB can be swapped (pgvector or Qdrant in Phase 7) without touching the embedding code.

Things worth saying in an interview:
- **One collection per (embedding model, chunking config).** The collection name encodes both, e.g. `10k_bge-small-en-v1-5_recursive350`. Never mix vectors from different models.
- **Distance metric:** set `hnsw:space = cosine`. Chroma returns *distance* (1 − cos), so higher is not better until you convert it.
- **Asymmetric query prefix:** BGE wants `"Represent this sentence for searching relevant passages: "` on the **query only** (`common.llm.embed_query`). Forgetting it is a common quiet quality loss. OpenAI embeddings are symmetric and need no prefix.
- Indexing 2,366 chunks with bge-small locally took about 38 s the first time and seconds afterwards because of the cache.

## 4. Generation with citations (see `04_ask.py`)

The prompt pattern is **numbered passages with source headers**, plus rules:
1. Answer **only** from the passages.
2. Cite every claim as `[n]`.
3. If the passages don't contain the answer, say "I don't know". This is the refusal path.
4. State units, because 10-K tables are usually in millions.

Each passage header (`[3] AAPL 10-K FY2025, Item 8: Financial Statements...`) tells the LLM **which company and year** a chunk belongs to. The chunk text alone often doesn't say, because risk-factor prose says "we" and never names the company.

Citations are *claims* by the model, not proof. The model can cite [3] for something [3] doesn't say. Checking that is **faithfulness evaluation** (Phase 4).

## 5. What we observed (demo in `04_ask.py`)

| Question | Result | Lesson |
|---|---|---|
| Apple FY2025 total net sales | ✅ $416,161M [3] | Works when the wording is distinctive and the fact sits in one chunk |
| NVIDIA export-control risk | ✅ but cites all 5 | FY2025 and FY2026 risk factors are near-duplicates, so **the top-k fills with duplicates** and the answer blends the two years |
| MSFT vs AMZN revenue growth | ❌ "I don't know" | **All 5 hits are Microsoft.** A single query vector lands near one company, and top-k can't guarantee coverage |
| Google ad revenue (not in the corpus) | ✅ refuses | Refusal works, **but** the retrieval scores (0.67–0.69) were almost as high as for real hits (0.70–0.83). The retriever always returns *something*, and scores can't tell you the answer is missing |
| Tesla, reliance on Elon Musk (`03_index.py`) | ⚠️ rank 2, score 0.717 | The right chunk *starts* with customer credit risk and mentions Musk halfway through. **Mixed-topic chunks dilute the vector** |

## 6. Where naive RAG breaks, and which phase fixes it

| Failure | Fix | Phase |
|---|---|---|
| Multi-entity or comparison questions get one-sided context | Query decomposition, multi-query, metadata filters per entity | 3, 2 |
| Exact terms (tickers, product names, "Item 1A") | BM25 + dense hybrid search | 2 |
| Near-duplicate chunks crowd the top-k | Reranking, MMR diversity, year filters | 2 |
| Right chunk at rank 7 instead of rank 1 | Cross-encoder rerank of the top 50 | 2 |
| Vague or abbreviated questions | Query rewriting, HyDE | 3 |
| Chunks lose context ("we", tables without headers) | Contextual retrieval, parent-child chunks | 5 |
| "Does it actually work?" | Golden set, recall@k, faithfulness | 4 |

---

## Interview Q&A

**Q: Walk me through a basic RAG pipeline.**
Offline: load the documents, parse and clean them, chunk them, embed the chunks, and store the vectors with metadata in a vector index. Online: embed the query (with the same model), run ANN top-k search, build a prompt from the retrieved chunks with source labels, generate an answer with citations, and include a refusal path. Then name the failure points (parsing, chunking, recall, faithfulness) and say how you would measure each.

**Q: How do you choose a chunk size?**
Start around 300–500 tokens with 10–20% overlap and respect document structure. Then tune on an eval set: measure recall@k and answer quality for, say, 200/400/800. The trade-off is precise vectors (small chunks) against enough context per chunk (large chunks), and prompt budget is k × size. Stay under the embedding model's token limit, or the text is silently truncated.

**Q: Why do chunks need overlap? What does it cost?**
Overlap keeps a sentence or fact that falls on a boundary whole in at least one chunk. It costs index size and near-duplicate retrieval results. Structure-aware splitting (on paragraphs) reduces the need for it.

**Q: Your RAG answers "I don't know" but the answer is in the corpus. How do you debug it?**
Separate retrieval from generation. First check whether the right chunk is in the top-k (log retrieved IDs; recall@k on a golden set). If it isn't, it's a retrieval problem: parsing (was the text even extracted?), chunking (split or diluted?), a query/document vocabulary mismatch (try hybrid search or query rewriting), or k too small. If the chunk *was* retrieved, it's a generation problem: prompt instructions, context ordering ("lost in the middle"), or the chunk missing context such as which company "we" refers to.

**Q: Why not just use a similarity threshold to detect "no answer"?**
In our run, an out-of-corpus question (Google) scored 0.69 against real hits at 0.70–0.83. Similarity scores are compressed and query-dependent. Handle it in generation (an explicit refusal instruction), with a reranker whose scores are better calibrated, or with a threshold calibrated on labeled data.

**Q: How do you get citations, and can you trust them?**
Number the passages in the prompt, require a `[n]` citation per claim, and map the numbers back to chunk IDs and source URLs. The model can still cite wrongly, so verify with faithfulness checks: an NLI model or an LLM judge asks whether passage n supports the claim (Phase 4).

**Q: What metadata do you store with each chunk, and why?**
Source document, entity (ticker), date or fiscal year, section, and a URL or page. It's used for citations, for filters ("only FY2025", "only this tenant"), for access control, for dedup and deletes when a document changes, and for debugging.

**Q: The same question gets a different answer after you re-index. Why?**
Possible causes: a different embedding model or version (vectors aren't comparable), changed chunking, ANN non-determinism or index parameters, ties in near-duplicate chunks, or LLM sampling (use temperature 0 for evaluation). Version your index config (model, chunker, size) the way we name collections.

**Q: Why parse into sections instead of chunking the raw text?**
Chunks never mix unrelated sections, every chunk gets section metadata for free (for filters and citations), and we can drop boilerplate sections. It's cheap structure that makes both retrieval and debugging better.

## Experiments to try
- [ ] Build and compare `--chunker fixed` vs `recursive`, and `--size 150` vs `800` (`02_chunk.py`, then `03_index.py` with the same flags, then `04_ask.py --chunker/--size`). Which demo questions change?
- [ ] Ask the comparison question with `-k 15`. Does Amazon show up? What does that do to the prompt size?
- [ ] Remove the BGE query prefix (`query_prefix` in `common/llm.py`) and compare the scores and ranks in `03_index.py`.
- [ ] Switch to `EMBED_PROVIDER=openai`, rerun 02 → 04, and compare. (The 02 truncation check is skipped for OpenAI because its 8K limit is far above our chunk sizes.)
- [ ] Ask "What was Microsoft's revenue?" with no year given. Which year does it pick, and does the answer say so?
- [ ] Delete the "I don't know" rule from `SYSTEM` and ask the Google question again.
