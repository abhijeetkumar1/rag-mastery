"""Chroma wrapper. We compute embeddings ourselves (common.llm.embed) and hand Chroma raw vectors,
so the vector DB is just storage + ANN index (HNSW) + metadata filtering, not a black box.
"""
import re

import chromadb

from common.config import DATA_DIR, EMBED_MODEL
from common.llm import embed, embed_query

CHROMA_DIR = DATA_DIR / "chroma"


def collection_name(chunker: str, size: int, model: str = EMBED_MODEL) -> str:
    # One collection per (model, chunking) config: vectors from different models must never mix
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model.split("/")[-1]).strip("-").lower()
    return f"10k_{slug}_{chunker}{size}"


def get_collection(name: str, reset: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset and name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    # cosine space: our vectors are unit-normalized, so this matches dot-product ranking
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def add_chunks(col, ids: list[str], texts: list[str], metadatas: list[dict], batch: int = 1000) -> None:
    vecs = embed(texts)
    for s in range(0, len(ids), batch):
        e = s + batch
        col.add(ids=ids[s:e], embeddings=vecs[s:e].tolist(), documents=texts[s:e], metadatas=metadatas[s:e])


def search(col, query: str, k: int = 5, where: dict | None = None) -> list[dict]:
    return search_by_vector(col, embed_query(query), k, where)


def search_by_vector(col, q, k: int = 5, where: dict | None = None) -> list[dict]:
    res = col.query(query_embeddings=[q.tolist()], n_results=k, where=where)
    return [
        {"id": i, "text": d, "meta": m, "score": 1 - dist}  # chroma returns cosine DISTANCE
        for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
