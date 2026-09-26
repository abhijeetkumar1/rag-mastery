"""Cross-encoder reranking: reorder the hybrid candidates, measure the gain, calibrate a relevance floor.

Run: uv run python -m phase2_better_retrieval.04_rerank
     RERANK_MODEL=BAAI/bge-reranker-base uv run python -m phase2_better_retrieval.04_rerank   # bigger model

1. Rank movement on the Apple question (fused rank -> reranked rank)
2. Probes: hybrid vs hybrid+rerank, and the effect of N (candidates reranked) on quality and latency
3. Score calibration: top-1 rerank score for in-corpus probes vs out-of-corpus questions
   -> the relevance floor used by 05_ask
"""
import time

import numpy as np

from common.config import RERANK_MODEL
from common.retriever import Retriever
from phase2_better_retrieval.probes import PROBES, hit_and_rr, relevant_ids

OUT_OF_CORPUS = [
    "What was Google's advertising revenue in 2025?",
    "Dojo supercomputer",
    "What is the capital of France?",
    "How many employees does Netflix have?",
    "What was Meta's capital expenditure in 2025?",
]


def main() -> None:
    R = Retriever()
    print(f"reranker: {RERANK_MODEL}\n")

    q = "What was Apple's total net sales in fiscal 2025?"
    print(f"1. {q}\n   {'rerank':>6} {'fused':>5} {'score':>7}  chunk")
    for r, h in enumerate(R.retrieve(q, k=6, rerank=True), 1):
        print(f"   {r:>6} {h['fused_rank']:>5} {h['rerank_score']:7.2f}  {h['id']}  {h['text'][:70]!r}")
    print()

    rel = relevant_ids(R.rows)
    print("2. Probes (12)            hit@5   MRR@5   ms/query")
    for label, kw in [("hybrid, no rerank", dict(rerank=False)), ("rerank top-10", dict(rerank=True, n_candidates=10)),
                      ("rerank top-50", dict(rerank=True, n_candidates=50)), ("rerank top-100", dict(rerank=True, n_candidates=100))]:
        R.retrieve("warm up", k=1, **kw)
        h5 = rr = 0
        t = time.perf_counter()
        for (pq, *_), r in zip(PROBES, rel):
            h, x = hit_and_rr([h["id"] for h in R.retrieve(pq, k=5, **kw)], r, 5)
            h5, rr = h5 + h, rr + x
        ms = (time.perf_counter() - t) * 1000 / len(PROBES)
        print(f"   {label:20s} {h5:>4}/12   {rr / 12:5.2f}   {ms:7.0f}")
    print("   -> more candidates = more chances to find the right chunk, but latency grows linearly with N.\n")

    print("3. Calibration: top-1 rerank score")
    inside = [R.retrieve(pq, k=1)[0]["rerank_score"] for pq, *_ in PROBES]
    outside = [R.retrieve(oq, k=1)[0]["rerank_score"] for oq in OUT_OF_CORPUS]
    print(f"   in-corpus probes : min={min(inside):6.2f}  median={np.median(inside):6.2f}  max={max(inside):6.2f}")
    for oq, s in zip(OUT_OF_CORPUS, outside):
        print(f"   out-of-corpus    : {s:6.2f}  {oq}")
    for floor in [-3.0, 0.0]:
        kept = sum(s >= floor for s in inside)
        blocked = sum(s < floor for s in outside)
        print(f"   floor {floor:+.1f}: keeps {kept}/{len(inside)} answerable, blocks {blocked}/{len(outside)} unanswerable")
    print("   -> no floor separates them perfectly: questions about the WRONG ENTITY (Google, Meta) still match")
    print("      on-topic chunks. A floor catches off-topic questions; entity checks (Phase 3) catch the rest.")


if __name__ == "__main__":
    main()
