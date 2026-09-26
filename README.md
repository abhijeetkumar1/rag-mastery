# RAG Mastery: from naive to agentic RAG in 7 days

**Domain:** a financial-filings Q&A assistant over public SEC 10-K reports (see "Dataset" below).

## Setup
```bash
uv sync                      # installs deps into .venv  (or: pip install -r requirements.txt)
cp .env.example .env         # then fill in OPENAI_API_KEY and SEC_USER_AGENT
uv run python -m phase0_foundations.01_similarity
```

Embeddings run locally by default (`EMBED_PROVIDER=hf`, `BAAI/bge-small-en-v1.5`); set `EMBED_PROVIDER=openai` to use the API instead. Chat always uses OpenAI.

## 7-day plan

| Day | Phase | Build | Key interview topics |
|---|---|---|---|
| 1 | 0 Foundations + 1 Naive RAG | Embeddings/similarity from scratch → ingest, chunk, embed, Chroma, generate with citations | RAG vs fine-tuning, embeddings, ANN/HNSW, chunking |
| 2 | 2 Better retrieval | BM25 + dense hybrid, RRF fusion, metadata filters, cross-encoder reranking | Sparse vs dense, bi- vs cross-encoder |
| 3 | 3 Query intelligence | Query rewriting, multi-query, HyDE, decomposition, routing | Query–doc mismatch, recall vs precision |
| 4 | 4 Evaluation | Golden Q&A set, retrieval metrics (recall@k, MRR, nDCG), faithfulness/relevance with LLM-as-judge | "How do you know it works?" |
| 5 | 5 Advanced indexing | Parent-child chunks, contextual retrieval, table handling, GraphRAG intro | Lost-in-the-middle, chunk-size trade-offs |
| 6 | 6 Agentic RAG | LangGraph agent: tool-based retrieval, self-correction (CRAG), multi-hop comparison | ReAct, when agents hurt |
| 7 | 7 Production + review | FastAPI + streaming, caching, tracing, guardrails, cost/latency; system-design mock | "Design RAG for 10M docs" |

Every phase folder contains runnable scripts plus a `NOTES.md` with theory, trade-offs, and interview Q&A.

## Dataset
10-K annual reports for AAPL, MSFT, NVDA, TSLA and AMZN (two most recent fiscal years each), downloaded from SEC EDGAR by `phase1_naive_rag/00_download_10k.py`. Not committed; re-download with that script.

## Structure
```
common/        config (.env), embed/chat wrappers with cache, chunkers, Chroma store
phase0_*/      one folder per phase: numbered scripts + NOTES.md
data/          raw/ filings, processed/ text + chunks, chroma/ index (all git-ignored)
```
