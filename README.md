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

| Day | Phase | Build | Observability | Guardrails | Key interview topics |
|---|---|---|---|---|---|
| 1 | 0 Foundations + 1 Naive RAG | Embeddings/similarity from scratch → ingest, chunk, embed, Chroma, generate with citations | Hand-rolled JSONL tracer: spans per stage, latency, tokens, cost; trace viewer with p50/p95 | Output: citation validator, numeric grounding check, refusal path | RAG vs fine-tuning, embeddings, ANN/HNSW, chunking, traces and spans |
| 2 | 2 Better retrieval | BM25 + dense hybrid, RRF fusion, metadata filters, cross-encoder reranking | A span per retrieval stage (BM25, dense, fused, reranked) to see where the right chunk enters or drops out | Relevance floor on calibrated rerank scores (refuse before the LLM call) | Sparse vs dense, bi- vs cross-encoder |
| 3 | 3 Query intelligence | Query rewriting, multi-query, HyDE, decomposition, routing | Log rewritten queries and sub-queries with their per-query hits | Input: scope classifier, company/ticker validation, "investment advice" detection | Query–doc mismatch, recall vs precision |
| 4 | 4 Evaluation | Golden Q&A set, retrieval metrics (recall@k, MRR, nDCG), faithfulness/relevance with LLM-as-judge | Offline eval = observability offline; traces become regression data | Faithfulness judge (LLM or NLI) as an optional runtime output check | "How do you know it works?" |
| 5 | 5 Advanced indexing | Parent-child chunks, contextual retrieval, table handling, GraphRAG intro | Log parent expansion and contextual headers | Indirect prompt injection: retrieved text is untrusted (test with a planted chunk, then add delimiting and a detector) | Lost-in-the-middle, chunk-size trade-offs |
| 6 | 6 Agentic RAG | LangGraph agent: tool-based retrieval, self-correction (CRAG), multi-hop comparison | Langfuse (or LangSmith/Phoenix): agent step and tool-call traces | Agent limits: max steps, tool allowlist, token/cost budget, loop detection | ReAct, when agents hurt |
| 7 | 7 Production + review | FastAPI + streaming, caching, cost/latency; system-design mock | OpenTelemetry export, dashboards (latency, cost, refusal rate, score drift), online eval sampling, user feedback | Full layer: PII redaction, moderation, rate limits, per-user access filters; frameworks (NeMo Guardrails, Guardrails AI, Llama Guard) vs hand-rolled | "Design RAG for 10M docs" |

Every phase folder contains runnable scripts plus:
- `WALKTHROUGH.md`: the architecture and a step-by-step explanation of the code. **Start here.**
- `NOTES.md`: theory, trade-offs, interview Q&A and experiments.

## Dataset
10-K annual reports for AAPL, MSFT, NVDA, TSLA and AMZN (two most recent fiscal years each), downloaded from SEC EDGAR by `phase1_naive_rag/00_download_10k.py`. Not committed; re-download with that script.

## Structure
```
common/        config (.env), embed/chat wrappers with cache, chunkers, Chroma store, tracer, guardrails,
               BM25, fusion (RRF), filters, reranker, hybrid Retriever
phase0_*/      one folder per phase: numbered scripts + NOTES.md
data/          raw/ filings, processed/ text + chunks, chroma/ index, traces/ request traces (all git-ignored)
```
