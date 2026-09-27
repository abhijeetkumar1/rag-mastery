"""Hybrid retrieval: fuse BM25 + dense with Reciprocal Rank Fusion, and compare against each alone.

Run: uv run python -m phase2_better_retrieval.02_hybrid_rrf

Part 1: RRF worked example on one query (ranks in, fused ranking out)
Part 2: probe table: hit@5 / MRR@5 / hit@50 / recall@50 for dense, BM25, score fusion, RRF
        hit@50    = "is ANY relevant chunk among the 50 candidates the reranker will see?"
        recall@50 = "what SHARE of all relevant chunks is among them?" (mean over probes)
"""
from common.fusion import rrf, score_fusion
from common.retriever import Retriever
from phase2_better_retrieval.probes import PROBES, hit_and_rr, recall_at, relevant_ids


def worked_example(R: Retriever, q: str) -> None:
    d = R.dense(q, 50, None)
    b = R.sparse(q, 50, None)
    d_rank = {i: r for r, (i, _) in enumerate(d, 1)}
    b_rank = {i: r for r, (i, _) in enumerate(b, 1)}
    fused = rrf([[i for i, _ in d], [i for i, _ in b]])
    print(f"Q: {q}\n{'fused':>5} {'dense':>6} {'bm25':>5}  {'rrf score':>10}  chunk")
    for r, (i, s) in enumerate(fused[:8], 1):
        dr, br = d_rank.get(i, "-"), b_rank.get(i, "-")
        parts = " + ".join(f"1/(60+{x})" for x in (dr, br) if x != "-")
        print(f"{r:>5} {dr!s:>6} {br!s:>5}  {s:10.5f}  {i}   = {parts}")
    print("-> a chunk ranked well by BOTH lists beats one ranked #1 by only one of them.\n")


def evaluate(R: Retriever) -> None:
    rel = relevant_ids(R.rows)
    modes = {
        "dense": lambda q: [i for i, _ in R.dense(q, 50, None)],
        "bm25": lambda q: [i for i, _ in R.sparse(q, 50, None)],
        "score fusion 50/50": lambda q: [i for i, _ in score_fusion([dict(R.dense(q, 50, None)), dict(R.sparse(q, 50, None))], [0.5, 0.5])],
        "rrf (k=60)": lambda q: [i for i, _ in rrf([[i for i, _ in R.dense(q, 50, None)], [i for i, _ in R.sparse(q, 50, None)]])],
    }
    print(f"{'mode':20s} {'hit@5':>6} {'MRR@5':>6} {'hit@50':>7} {'recall@50':>10}   keyword/semantic/numeric hit@5")
    for name, fn in modes.items():
        h5 = rr = h50 = rec50 = 0
        per_kind = {"keyword": 0, "semantic": 0, "numeric": 0}
        for (q, kind, *_), r in zip(PROBES, rel):
            ids = fn(q)
            h, x = hit_and_rr(ids, r, 5)
            h5, rr, h50 = h5 + h, rr + x, h50 + hit_and_rr(ids, r, 50)[0]
            rec50 += recall_at(ids, r, 50)
            per_kind[kind] += h
        print(f"{name:20s} {h5:>4}/12 {rr / 12:6.2f} {h50:>4}/12 {rec50 / 12:10.2f}   {per_kind['keyword']}/4  {per_kind['semantic']}/4  {per_kind['numeric']}/4")
    print("\nFusion's job is RECALL: getting the right chunk into the candidate pool. Its top-5 can be")
    print("worse than the best single retriever; the reranker (04_rerank) fixes the ORDER.")


def main() -> None:
    R = Retriever()
    worked_example(R, "Is Tesla too reliant on its CEO?")
    evaluate(R)


if __name__ == "__main__":
    main()
