"""Combine ranked lists from different retrievers (BM25 + dense) into one ranking.

Problem: BM25 scores are unbounded (0..~30), cosine scores live in a narrow band (~0.4..0.8).
You can't just add them. Two standard answers:

  rrf()          Reciprocal Rank Fusion: ignore scores, use RANKS.  score(d) = Σ_r 1 / (k + rank_r(d))
  score_fusion() normalize each list's scores to [0,1] (min-max), then weighted sum. Sensitive to
                 outliers and to what "the min" happens to be in each candidate list.
"""


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """rankings: lists of doc ids, best first. k=60 is the constant from the original paper
    (Cormack et al., 2009): it damps the gap between rank 1 and rank 2 so no single list dominates."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def score_fusion(scored: list[dict[str, float]], weights: list[float]) -> list[tuple[str, float]]:
    """scored: one {doc_id: raw_score} per retriever. Missing from a list = contributes 0."""
    fused: dict[str, float] = {}
    for s, w in zip(scored, weights):
        if not s:
            continue
        lo, hi = min(s.values()), max(s.values())
        for doc_id, v in s.items():
            norm = (v - lo) / (hi - lo) if hi > lo else 1.0
            fused[doc_id] = fused.get(doc_id, 0.0) + w * norm
    return sorted(fused.items(), key=lambda x: -x[1])
