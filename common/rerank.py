"""Cross-encoder reranking: re-score a SHORTLIST of candidates by reading query + chunk together.

Bi-encoder (embeddings): query and chunk encoded separately -> fast, precomputable, coarse.
Cross-encoder:           [CLS] query [SEP] chunk [SEP] -> one relevance score. Attention runs ACROSS
                         the query and chunk tokens, so it can see "fiscal 2025" vs a 2023 column,
                         negation, which entity a number belongs to... but it costs one model forward
                         pass per (query, chunk) pair, so only the top ~20-100 candidates get reranked.
"""
import threading

from common.config import HF_TOKEN, RERANK_MODEL

_models: dict = {}
# Phase 7: one inference at a time. On Apple's GPU (MPS), two threads calling predict() at once crash the process
# (Metal: "A command encoder is already encoding to this command buffer"); the server's thread pool does exactly that.
# Production would batch concurrent requests into one forward pass, or run the model in its own inference server.
_lock = threading.Lock()


def _model(name: str):
    if name not in _models:
        from sentence_transformers import CrossEncoder  # lazy: pulls in torch
        _models[name] = CrossEncoder(name, token=HF_TOKEN)
    return _models[name]


def rerank_scores(query: str, texts: list[str], model: str = RERANK_MODEL) -> list[float]:
    """Relevance score per text (higher = more relevant). ms-marco cross-encoders output a raw logit."""
    if not texts:
        return []
    m = _model(model)
    with _lock:
        scores = m.predict([(query, t) for t in texts], batch_size=32)
    return [float(s) for s in scores]
