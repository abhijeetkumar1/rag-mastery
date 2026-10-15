"""Second vector DB: migrate the default index from Chroma to Qdrant, then check the two give the same results.

Run: uv run python -m phase7_production.05_vector_db --migrate     # copy vectors Chroma -> Qdrant (data/qdrant/), then compare
     uv run python -m phase7_production.05_vector_db               # compare only
Serve on it: VECTOR_DB=qdrant uv run python -m phase7_production.01_serve   (or 01_serve --qdrant)

Comparisons, on the golden dev + test questions whose query embeddings are already cached (no API calls):
  exact top-10 (NumPy brute force over all 3,043 vectors) = ground truth
  Chroma (HNSW, approximate) recall@10 vs exact;  Qdrant local (exact search) recall@10 vs exact
  the same with a metadata filter (one ticker), where HNSW graphs are known to lose recall
  the full hybrid Retriever (dense + BM25 + RRF + rerank) top-6 on each backend
  latency per dense query
"""
import argparse
import importlib
import json
import time

import chromadb
import numpy as np

from common.config import EMBED_MODEL
from common.filters import extract_filters, to_chroma_where
from common.llm import _cache_path, embed_query, query_prefix
from common.store import CHROMA_DIR, collection_name, search_by_vector
from common.vectordb import QDRANT_DIR, QdrantCollection, migrate_from_chroma, qdrant_client

ask5 = importlib.import_module("phase5_advanced_indexing.06_ask")
retrieval5 = importlib.import_module("phase5_advanced_indexing.05_retrieval_eval")
K = 10


def cached_questions() -> list[str]:
    qs = [x["question"] for s in ("dev", "test", "test2") for x in retrieval5.golden(s)]
    return [q for q in dict.fromkeys(qs) if _cache_path(query_prefix(EMBED_MODEL) + q, EMBED_MODEL).exists()]


def dir_mb(path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--migrate", action="store_true")
    args = ap.parse_args()
    name = collection_name("structctx", 350)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(name)
    if args.migrate:
        t = time.perf_counter()
        n = migrate_from_chroma(chroma, name)
        print(f"migrated {n} points to Qdrant ({QDRANT_DIR}) in {time.perf_counter() - t:.1f} s, no re-embedding")
    qd = QdrantCollection(name)
    print(f"collection {name}: Chroma {chroma.count()} vectors, Qdrant {qd.count()} points   "
          f"on disk: Chroma dir {dir_mb(CHROMA_DIR):.0f} MB (all collections), Qdrant {dir_mb(QDRANT_DIR):.0f} MB")

    got = chroma.get(include=["embeddings", "metadatas"])
    ids, M = got["ids"], np.array(got["embeddings"], dtype=np.float32)
    tick = np.array([m["ticker"] for m in got["metadatas"]])
    qs = cached_questions()
    print(f"{len(qs)} golden questions with cached query embeddings\n")

    rec = {"chroma": [], "qdrant": [], "chroma_f": [], "qdrant_f": []}
    lat = {"chroma": [], "qdrant": [], "numpy": []}
    for q in qs:
        v = embed_query(q)
        t = time.perf_counter()
        exact = [ids[i] for i in np.argsort(-(M @ v))[:K]]
        lat["numpy"].append(time.perf_counter() - t)
        for backend, col in (("chroma", chroma), ("qdrant", qd)):
            t = time.perf_counter()
            hits = search_by_vector(col, v, k=K)
            lat[backend].append(time.perf_counter() - t)
            rec[backend].append(len({h["id"] for h in hits} & set(exact)) / K)
        f = extract_filters(q)
        if f.get("ticker"):
            f = {"ticker": f["ticker"][:1]}
            mask = tick == f["ticker"][0]
            sims = np.where(mask, M @ v, -np.inf)
            exact_f = [ids[i] for i in np.argsort(-sims)[:K]]
            for backend, col in (("chroma_f", chroma), ("qdrant_f", qd)):
                hits = search_by_vector(col, v, k=K, where=to_chroma_where(f))
                rec[backend].append(len({h["id"] for h in hits} & set(exact_f)) / K)
    print(f"recall@{K} vs exact brute force      unfiltered (n={len(rec['chroma'])})   ticker-filtered (n={len(rec['chroma_f'])})")
    for b in ("chroma", "qdrant"):
        r, rf = np.array(rec[b]), np.array(rec[b + "_f"])
        print(f"  {b:7s} {'HNSW' if b == 'chroma' else 'exact'}   mean {r.mean():.3f}  min {r.min():.1f}  "
              f"perfect {int((r == 1).sum())}/{len(r)}       mean {rf.mean():.3f}  min {rf.min():.1f}")
    print("\ndense query latency (ms): " + "   ".join(
        f"{b} p50 {np.median(lat[b]) * 1000:.1f} / p95 {np.percentile(lat[b], 95) * 1000:.1f}" for b in lat))

    ix = ask5.Index("structctx350")
    R = ix.R
    chroma_col = R.col
    same, n = 0, 0
    for q in qs[:40]:
        R.col = chroma_col
        a = [h["id"] for h in R.retrieve(q, k=6)]
        R.col = qd
        b = [h["id"] for h in R.retrieve(q, k=6)]
        same += len(set(a) & set(b)) / 6
        n += 1
    R.col = chroma_col
    print(f"\nfull hybrid Retriever top-6 (dense+BM25+RRF+rerank), Chroma vs Qdrant backend: overlap {same / n:.3f} on {n} questions")
    print(json.dumps({"qdrant_collections": [c.name for c in qdrant_client().get_collections().collections]}))
    qdrant_client().close()  # release the local-mode file lock cleanly (its __del__ fails during interpreter shutdown)


if __name__ == "__main__":
    main()
