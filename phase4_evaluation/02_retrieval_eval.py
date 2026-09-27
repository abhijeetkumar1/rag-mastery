"""Retrieval evaluation on the held-out golden set: every retrieval configuration from Phases 1-3.

Run: uv run python -m phase4_evaluation.02_retrieval_eval               # dev split
     uv run python -m phase4_evaluation.02_retrieval_eval --split test  # held-out test split (report once)

Configurations (all return a top-6 list; Phase 3 splits the 6 across sub-questions):
  P1 dense                  Phase 1: embedding search only
  P2 bm25                   keyword only
  P2 hybrid (RRF)           dense + BM25 fused, no rerank
  P2 hybrid+rerank          + cross-encoder
  P2 +auto filters          + rule-based filters and per-company fan-out (Phase 2's 05_ask)
  P3 plan                   LLM plan: sub-questions with their own filters, original question kept as a query

Metrics @6 on the 54 answerable questions: hit, recall, precision, MRR, nDCG (graded: 2 = the evidence in the
filing the question is about, 1 = the same text in the other filing), with 95% bootstrap CIs, plus
paired differences vs Phase 1 and entity coverage on the 6 comparison questions.
"""
import importlib
import json
from pathlib import Path

from common.evaluation import bootstrap_ci, hit_at, mrr_at, ndcg_at, paired_bootstrap, precision_at, recall_at
from common.filters import extract_filters
from common.query import plan
from common.retriever import Retriever

ask3 = importlib.import_module("phase3_query_intelligence.06_ask")
K = 6
METRICS = {"hit": hit_at, "recall": recall_at, "precision": precision_at, "mrr": mrr_at, "ndcg": ndcg_at}


def interleave(hits: list[dict]) -> list[str]:
    """Phase 3 returns sub-question blocks [MSFT..., AMZN...]; interleave so rank positions are comparable."""
    blocks: dict[str, list[str]] = {}
    for h in hits:
        blocks.setdefault(h.get("sub_question", ""), []).append(h["id"])
    out, lists = [], list(blocks.values())
    for i in range(max(map(len, lists), default=0)):
        out += [lst[i] for lst in lists if i < len(lst)]
    return out


def configs(R: Retriever) -> dict:
    def p2_auto(q):
        f = extract_filters(q)
        tickers = f.get("ticker", [])
        if len(tickers) > 1:  # as in phase2 05_ask: fan-out per company, keeping the year filter
            return [h["id"] for h in R.retrieve_fanout(q, "ticker", tickers, k_each=K // len(tickers), filters=f)]
        return [h["id"] for h in R.retrieve(q, k=K, filters="auto")]

    def p3(q):
        p, _, _ = plan(q)
        if p["route"] != "answer":
            return []  # the router refused: nothing retrieved (counts as a miss on answerable questions)
        return interleave(ask3.retrieve_for_plan(R, q, p["sub_questions"], k=K))

    return {
        "P1 dense": lambda q: [i for i, _ in R.dense(q, K, None)],
        "P2 bm25": lambda q: [i for i, _ in R.sparse(q, K, None)],
        "P2 hybrid (RRF)": lambda q: [h["id"] for h in R.retrieve(q, k=K, rerank=False)],
        "P2 hybrid+rerank": lambda q: [h["id"] for h in R.retrieve(q, k=K)],
        "P2 +auto filters": p2_auto,
        "P3 plan": p3,
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    split = ap.parse_args().split
    items = [json.loads(line) for line in (Path(__file__).parent / f"golden_{split}.jsonl").read_text().splitlines()]
    answerable = [x for x in items if x["expected"] == "answer"]
    R = Retriever()
    R.warmup()
    runs = {name: [fn(x["question"]) for x in answerable] for name, fn in configs(R).items()}

    per_q = {name: {m: [f(ids, {k: int(v) for k, v in x["relevant"].items()}, K) for ids, x in zip(ranked, answerable)]
                    for m, f in METRICS.items()} for name, ranked in runs.items()}

    print(f"{split}: {len(answerable)} answerable questions, metrics @{K}, mean [95% bootstrap CI]\n")
    print(f"{'config':18s} " + " ".join(f"{m:>18s}" for m in METRICS))
    for name, ms in per_q.items():
        cells = []
        for m in METRICS:
            mean, lo, hi = bootstrap_ci(ms[m])
            cells.append(f"{mean:.2f} [{lo:.2f},{hi:.2f}]")
        print(f"{name:18s} " + " ".join(f"{c:>18s}" for c in cells))

    base = "P1 dense"
    print(f"\nPaired difference in nDCG@{K} vs {base} (same questions): mean [95% CI], P(no improvement)")
    for name in per_q:
        if name != base:
            d, lo, hi, p = paired_bootstrap(per_q[base]["ndcg"], per_q[name]["ndcg"])
            print(f"  {name:18s} {d:+.3f} [{lo:+.3f},{hi:+.3f}]  p={p:.3f}" + ("   significant" if lo > 0 or hi < 0 else ""))

    print(f"\nBreakdown: hit@{K} by question type")
    kinds = sorted({x["type"] for x in answerable})
    print(f"  {'config':18s} " + " ".join(f"{k:>12s}" for k in kinds))
    for name, ms in per_q.items():
        cells = [f"{sum(h for h, x in zip(ms['hit'], answerable) if x['type'] == k)}/{sum(x['type'] == k for x in answerable)}" for k in kinds]
        print(f"  {name:18s} " + " ".join(f"{c:>12s}" for c in cells))

    cmp_items = [(i, x) for i, x in enumerate(answerable) if x["type"] == "comparison"]
    print(f"\nEntity coverage on {len(cmp_items)} comparison questions (companies whose evidence is in the top {K})")
    total = sum(len(x["relevant_by_ticker"]) for _, x in cmp_items)
    for name, ranked in runs.items():
        cov = sum(bool(set(ranked[i]) & set(ids)) for i, x in cmp_items for ids in x["relevant_by_ticker"].values())
        print(f"  {name:18s} {cov}/{total}")

    out = Path(__file__).parent / "results" / "retrieval_runs.json"  # overwritten per run (git-ignored)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"k": K, "runs": {n: dict(zip([x["id"] for x in answerable], r)) for n, r in runs.items()}}))
    print(f"\nper-question rankings saved to {out.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
