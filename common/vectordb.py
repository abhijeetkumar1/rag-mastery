"""A second vector database behind the same interface (Phase 7): Qdrant, as a drop-in for Chroma.

Retriever only calls two methods on its collection, Chroma's col.query(query_embeddings, n_results, where) and
col.count(). QdrantCollection implements exactly those, translating Chroma's `where` into a Qdrant Filter, so
`Retriever` runs unchanged on either store (VECTOR_DB=qdrant). That's the adapter pattern, and it's why the
retrieval code never imported chromadb directly.

Migration copies the stored VECTORS out of Chroma (no re-embedding: the vectors are the expensive part, and
re-embedding with a different model version would silently change retrieval). Qdrant point ids must be unsigned
ints or UUIDs, so the chunk id becomes a deterministic UUID5 and is kept in the payload.

Local mode (QdrantClient(path=...)) is an embedded, pure-Python store for development: it does exact (brute-force)
search and has no HNSW. The server (Docker image qdrant/qdrant, or Qdrant Cloud) has the same API with HNSW,
payload indexes, quantization and sharding: QdrantClient(url="http://localhost:6333").
"""
import uuid

from common.config import DATA_DIR

QDRANT_DIR = DATA_DIR / "qdrant"
_NS = uuid.UUID("6f1c2b9e-0000-4000-8000-000000000007")
_clients: dict = {}


def qdrant_client(path=QDRANT_DIR):
    # local mode takes a file lock on the directory: one client per process
    from qdrant_client import QdrantClient
    key = str(path)
    if key not in _clients:
        _clients[key] = QdrantClient(path=key)
    return _clients[key]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


def where_to_filter(where: dict | None):
    """Chroma where ({"ticker": {"$in": [...]}} or {"$and": [...]}) -> Qdrant Filter(must=[...])."""
    from qdrant_client import models
    if not where:
        return None
    clauses = where["$and"] if "$and" in where else [where]
    must = []
    for c in clauses:
        (key, cond), = c.items()
        values = cond["$in"] if isinstance(cond, dict) else [cond]
        must.append(models.FieldCondition(key=key, match=models.MatchAny(any=values)))
    return models.Filter(must=must)


class QdrantCollection:
    """The two methods of a Chroma collection that Retriever uses."""

    def __init__(self, name: str, client=None):
        self.name = name
        self.client = client or qdrant_client()

    def count(self) -> int:
        return self.client.count(self.name, exact=True).count

    def query(self, query_embeddings: list[list[float]], n_results: int, where: dict | None = None) -> dict:
        res = self.client.query_points(self.name, query=query_embeddings[0], limit=n_results,
                                       query_filter=where_to_filter(where), with_payload=True)
        pts = res.points
        # Chroma returns cosine DISTANCE (1 - similarity); store.search_by_vector converts back
        return {"ids": [[p.payload["chunk_id"] for p in pts]], "documents": [[p.payload["document"] for p in pts]],
                "metadatas": [[{k: v for k, v in p.payload.items() if k not in ("chunk_id", "document")} for p in pts]],
                "distances": [[1 - p.score for p in pts]]}


def migrate_from_chroma(chroma_col, name: str, client=None, batch: int = 500) -> int:
    """Copy ids, vectors, documents and metadata from a Chroma collection into a new Qdrant collection."""
    from qdrant_client import models
    client = client or qdrant_client()
    total = chroma_col.count()
    first = chroma_col.get(limit=1, include=["embeddings"])
    dim = len(first["embeddings"][0])
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    for offset in range(0, total, batch):
        got = chroma_col.get(offset=offset, limit=batch, include=["embeddings", "documents", "metadatas"])
        client.upsert(name, points=[
            models.PointStruct(id=point_id(i), vector=list(map(float, v)), payload={"chunk_id": i, "document": d, **m})
            for i, v, d, m in zip(got["ids"], got["embeddings"], got["documents"], got["metadatas"])])
    return client.count(name, exact=True).count
