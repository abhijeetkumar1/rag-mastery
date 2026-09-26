"""Thin OpenAI / HuggingFace wrappers with a disk cache for embeddings.

Why cache? Embedding the same text twice is pure waste — same input, same model => same vector.
Production RAG systems cache by hash(model + text). Interviewers like hearing this.
"""
import hashlib
import json

import numpy as np
from openai import OpenAI

from common.config import CACHE_DIR, CHAT_MODEL, EMBED_MODEL, EMBED_PROVIDER, HF_TOKEN, require_key

_client: OpenAI | None = None
_hf_models: dict = {}


def client() -> OpenAI:
    global _client
    if _client is None:
        require_key()
        _client = OpenAI()
    return _client


def _hf_model(name: str):
    """Lazy-load a sentence-transformers model (import is slow and pulls in torch)."""
    if name not in _hf_models:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise SystemExit("sentence-transformers missing: run uv sync")
        _hf_models[name] = SentenceTransformer(name, token=HF_TOKEN)
    return _hf_models[name]


def _embed_batch(texts: list[str], model: str, provider: str) -> list[list[float]]:
    if provider == "hf":
        # normalize so dot == cosine, same as OpenAI's pre-normalized vectors
        return _hf_model(model).encode(texts, normalize_embeddings=True).tolist()
    resp = client().embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def _cache_path(text: str, model: str):
    key = hashlib.sha256(f"{model}::{text}".encode()).hexdigest()
    return CACHE_DIR / "embeddings" / f"{key}.json"


def embed(
    texts: list[str],
    model: str = EMBED_MODEL,
    provider: str = EMBED_PROVIDER,
    batch_size: int = 256,
) -> np.ndarray:
    """Return an (n, d) float32 matrix. Cached per text; misses are batched into one call.

    The cache key includes the model name, so switching provider/model never mixes vectors.
    """
    vectors: list[list[float] | None] = [None] * len(texts)
    misses = []
    for i, t in enumerate(texts):
        p = _cache_path(t, model)
        if p.exists():
            vectors[i] = json.loads(p.read_text())
        else:
            misses.append(i)

    for start in range(0, len(misses), batch_size):
        idx = misses[start : start + batch_size]
        batch = _embed_batch([texts[i] for i in idx], model, provider)
        for i, vec in zip(idx, batch):
            vectors[i] = vec
            p = _cache_path(texts[i], model)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(vec))

    return np.array(vectors, dtype=np.float32)


# Asymmetric models want an instruction on the QUERY side only (docs are embedded as-is).
# BGE v1.5: https://huggingface.co/BAAI/bge-small-en-v1.5 ; OpenAI / MiniLM are symmetric -> no prefix.
def query_prefix(model: str) -> str:
    if model.startswith("BAAI/bge-") and "-en" in model:
        return "Represent this sentence for searching relevant passages: "
    return ""


def embed_query(query: str, model: str = EMBED_MODEL, provider: str = EMBED_PROVIDER) -> np.ndarray:
    return embed([query_prefix(model) + query], model=model, provider=provider)[0]


def chat_completion(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0.0, **kw):
    """Full API response: use when you need token usage (tracing, cost)."""
    return client().chat.completions.create(model=model, messages=messages, temperature=temperature, **kw)


def chat(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0.0, **kw) -> str:
    return chat_completion(messages, model, temperature, **kw).choices[0].message.content
