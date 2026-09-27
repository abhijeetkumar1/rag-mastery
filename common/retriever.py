"""The Phase 2 retrieval pipeline, reusable by later phases:

    query ─┬─► dense (Chroma, top-N, where=filter) ─┐
           │                                         ├─► RRF fusion ─► top-N candidates ─► cross-encoder rerank ─► top-k
           └─► BM25  (own index, top-N, mask=filter) ┘

Every stage is optional (mode="dense"|"bm25"|"hybrid", rerank=True/False) so scripts can compare
them, and every stage gets its own span when a Trace is passed in.
"""
import json
from contextlib import nullcontext

from common.bm25 import BM25Index
from common.config import DATA_DIR, RERANK_MODEL
from common.filters import extract_filters, to_chroma_where, to_mask
from common.fusion import rrf
from common.llm import embed, embed_query
from common.rerank import rerank_scores
from common.store import collection_name, get_collection, search_by_vector


class Retriever:
    def __init__(self, chunker: str = "recursive", size: int = 350):
        path = DATA_DIR / "processed" / f"chunks_{chunker}{size}.jsonl"
        self.rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.by_id = {r["id"]: r for r in self.rows}
        self.bm25 = BM25Index([r["text"] for r in self.rows])
        self.col = get_collection(collection_name(chunker, size))
        if self.col.count() != len(self.rows):
            raise SystemExit(f"Index has {self.col.count()} chunks but {path.name} has {len(self.rows)}: "
                             f"rerun phase1_naive_rag.03_index")

    def warmup(self) -> None:
        """Load the reranker (and torch) now, not inside the first user request. A cold start showed up in
        the traces as a ~12 s first request, pushing p95 from ~2 s to ~12 s."""
        rerank_scores("warm up", ["warm up"])

    def dense(self, query: str, n: int, f: dict | None, as_document: bool = False) -> list[tuple[str, float]]:
        # as_document: embed the text like a CHUNK (no query instruction prefix). Used for HyDE passages.
        vec = embed([query])[0] if as_document else embed_query(query)
        hits = search_by_vector(self.col, vec, k=n, where=to_chroma_where(f))
        return [(h["id"], h["score"]) for h in hits]

    def sparse(self, query: str, n: int, f: dict | None) -> list[tuple[str, float]]:
        return [(self.rows[i]["id"], s) for i, s in self.bm25.search(query, n, mask=to_mask(f, self.rows))]

    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid", rerank: bool = True,
                 n_candidates: int = 50, filters: dict | str | None = None, trace=None,
                 hyde_passage: str | None = None) -> list[dict]:
        """filters: None (no filter), "auto" (rule-based extraction from the query), or an explicit dict.
        hyde_passage: if given, the DENSE side searches with this hypothetical passage (embedded as a document)
        instead of the question; BM25 and the reranker still use the question."""
        span = trace.span if trace else (lambda *a, **kw: nullcontext({}))

        with span("filters") as sp:
            f = extract_filters(query) if filters == "auto" else filters
            sp["filters"] = f

        ranked: dict[str, dict] = {}  # id -> stage info, kept for inspection / tracing
        lists = []
        if mode in ("dense", "hybrid"):
            with span("dense_search", n=n_candidates, hyde=hyde_passage is not None) as sp:
                d = self.dense(hyde_passage or query, n_candidates, f, as_document=hyde_passage is not None)
                sp["top"] = [[i, round(s, 4)] for i, s in d[:10]]
            lists.append([i for i, _ in d])
            for rank, (i, s) in enumerate(d, 1):
                ranked.setdefault(i, {})["dense_rank"], ranked[i]["dense_score"] = rank, s
        if mode in ("bm25", "hybrid"):
            with span("bm25_search", n=n_candidates) as sp:
                b = self.sparse(query, n_candidates, f)
                sp["top"] = [[i, round(s, 2)] for i, s in b[:10]]
            lists.append([i for i, _ in b])
            for rank, (i, s) in enumerate(b, 1):
                ranked.setdefault(i, {})["bm25_rank"], ranked[i]["bm25_score"] = rank, s

        with span("fusion", method="rrf" if len(lists) > 1 else "none") as sp:
            fused = rrf(lists) if len(lists) > 1 else [(i, 1.0 / (60 + r)) for r, i in enumerate(lists[0], 1)]
            fused = fused[:n_candidates]
            sp["top"] = [[i, round(s, 5)] for i, s in fused[:10]]
        hits = [{"id": i, "text": self.by_id[i]["text"], "meta": self._meta(i), "rrf_score": s,
                 "fused_rank": r, **ranked[i]} for r, (i, s) in enumerate(fused, 1)]

        if rerank and hits:
            with span("rerank", model=RERANK_MODEL, n=len(hits)) as sp:
                for h, s in zip(hits, rerank_scores(query, [h["text"] for h in hits])):
                    h["rerank_score"] = s
                hits.sort(key=lambda h: -h["rerank_score"])
                sp["top"] = [[h["id"], round(h["rerank_score"], 3), h["fused_rank"]] for h in hits[:10]]

        out = hits[:k]
        for h in out:
            h["score"] = h.get("rerank_score", h["rrf_score"])
        return out

    def retrieve_multi(self, queries: list[str], rerank_query: str, k: int = 5, n_candidates: int = 50,
                       filters: dict | None = None, trace=None) -> list[dict]:
        """Multi-query / RAG-Fusion: hybrid candidates for EACH query, RRF across all of them, then ONE rerank
        against the user's original question (the reranker judges relevance to what the user asked)."""
        span = trace.span if trace else (lambda *a, **kw: nullcontext({}))
        per_query = [self.retrieve(q, k=n_candidates, rerank=False, n_candidates=n_candidates, filters=filters, trace=trace)
                     for q in queries]
        with span("multi_query_fusion", n_queries=len(queries)) as sp:
            fused = rrf([[h["id"] for h in hits] for hits in per_query])[:n_candidates]
            sp["top"] = [[i, round(s, 5)] for i, s in fused[:10]]
        hits = [{"id": i, "text": self.by_id[i]["text"], "meta": self._meta(i), "rrf_score": s, "fused_rank": r}
                for r, (i, s) in enumerate(fused, 1)]
        with span("rerank", model=RERANK_MODEL, n=len(hits)) as sp:
            for h, s in zip(hits, rerank_scores(rerank_query, [h["text"] for h in hits])):
                h["rerank_score"] = h["score"] = s
            hits.sort(key=lambda h: -h["rerank_score"])
            sp["top"] = [[h["id"], round(h["rerank_score"], 3), h["fused_rank"]] for h in hits[:10]]
        return hits[:k]

    def retrieve_fanout(self, query: str, key: str, values: list, k_each: int = 3, trace=None, **kw) -> list[dict]:
        """One retrieval PER filter value (e.g. per ticker), results interleaved. A single filter like
        ticker ∈ {MSFT, AMZN} only restricts WHAT can come back; it doesn't guarantee each value shows up.
        Fan-out guarantees coverage: the simplest form of query decomposition (Phase 3 generalizes it)."""
        base = kw.pop("filters", None) or {}
        per_value = [self.retrieve(query, k=k_each, filters={**base, key: [v]}, trace=trace, **kw) for v in values]
        return [h for group in zip(*per_value) for h in group] + [h for g in per_value for h in g[len(min(per_value, key=len)):]]

    def _meta(self, i: str) -> dict:
        r = self.by_id[i]
        return {k: r[k] for k in ("ticker", "fiscal_year", "item", "section", "url")}
