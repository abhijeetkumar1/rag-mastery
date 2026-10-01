"""Retrieval evaluation of the Phase 5 indexes on the golden dev split (Phase 4's metrics and statistics).

Run: uv run python -m phase5_advanced_indexing.05_retrieval_eval                 # dev (tune here)
     uv run python -m phase5_advanced_indexing.05_retrieval_eval --split test2   # the fresh held-out split, once, at the end

For each index (recursive350 = Phase 1's chunks, the baseline) and each retrieval config:
  rrf      hybrid dense + BM25, RRF, no reranker
  rerank   + cross-encoder (Phase 2)
  plan     Phase 3's LLM plan: sub-questions with filters, original question kept (the production config)
Metrics @6 over the answerable questions: hit, recall, MRR, nDCG, with labels re-derived for each chunk set
(labels.py), plus the TOKENS the passages put in the prompt: bigger chunks make hit@6 easier and the prompt
bigger, so a score without its context size isn't comparable. Parent-child is scored on the PARENTS (what the LLM reads);
"parent-child@3" retrieves 3 parents, about the token budget of 6 normal chunks (metrics still @6 positions).
"""
import argparse
import importlib
import json
from pathlib import Path

from common.chunking import count_tokens
from common.evaluation import bootstrap_ci, hit_at, mrr_at, ndcg_at, paired_bootstrap, recall_at
from common.parent_child import expand_to_parents
from common.query import plan

ask5 = importlib.import_module("phase5_advanced_indexing.06_ask")
retrieval4 = importlib.import_module("phase4_evaluation.02_retrieval_eval")
labels = importlib.import_module("phase5_advanced_indexing.labels")
HERE = Path(__file__).parent
K = 6
METRICS = {"hit": hit_at, "recall": recall_at, "mrr": mrr_at, "ndcg": ndcg_at}


def golden(split: str) -> list[dict]:
    path = HERE.parent / "phase4_evaluation" / f"golden_{split}.jsonl"  # all golden sets live with their builder
    return [json.loads(line) for line in path.read_text().splitlines()]


def run_index(ix, items: list[dict], k: int = K) -> dict[str, list[list[dict]]]:
    def flat(q: str, rerank: bool) -> list[dict]:
        if ix.parents is None:
            return ix.R.retrieve(q, k=k, rerank=rerank)
        return expand_to_parents(ix.R.retrieve(q, k=k * ask5.CHILD_FACTOR, rerank=rerank), ix.R.by_id, ix.parents, k)

    def planned(q: str) -> list[dict]:
        p, _, _ = plan(q)
        if p["route"] != "answer":
            return []
        hits = ask5.retrieve(ix, q, p["sub_questions"], k=k)
        order = retrieval4.interleave(hits)  # sub-question blocks -> interleaved ranks, as in Phase 4
        by = {h["id"]: h for h in hits}
        return [by[i] for i in order]

    return {"rrf": [flat(x["question"], False) for x in items],
            "rerank": [flat(x["question"], True) for x in items],
            "plan": [planned(x["question"]) for x in items]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--indexes", nargs="+", default=[*ask5.INDEXES, "parent-child@3"],
                    help="index names; name@k retrieves k passages instead of 6 (equal token budget)")
    args = ap.parse_args()
    items = [x for x in golden(args.split) if x["expected"] == "answer"]
    results, label_info = {}, {}
    for name in args.indexes:
        try:
            ix, k = ask5.open_index(name)
        except SystemExit as e:  # e.g. llmctx350 not built/indexed yet
            print(f"skip {name}: {e}")
            continue
        ix.warmup()
        rows = list(ix.parents.values()) if ix.parents else ix.R.rows
        rel = [labels.relabel(x, rows) for x in items]
        label_info[name] = (sum(not r for r in rel), sum(map(len, rel)) / len(rel))
        for cfg, hit_lists in run_index(ix, items, k or K).items():
            ids = [[h["id"] for h in hits] for hits in hit_lists]
            per_q = {m: [f(i, r, K) for i, r in zip(ids, rel)] for m, f in METRICS.items()}
            per_q["tokens"] = [sum(count_tokens(h["text"]) for h in hits[:K]) for hits in hit_lists]
            cov = 0
            for x, i in zip(items, ids):
                if x["type"] == "comparison":
                    cov += sum(bool(set(i) & set(v)) for v in labels.relabel_by_ticker(x, rows).values())
            per_q["coverage"] = cov
            results[(name, cfg)] = per_q
        print(f"  done {name}")

    n_cmp = sum(2 for x in items if x["type"] == "comparison")
    print(f"\n{args.split}: {len(items)} answerable questions, @{K}, mean [95% bootstrap CI]")
    print(f"labels re-derived per chunk set: " + ", ".join(f"{n} {m:.1f}/q ({e} without)" for n, (e, m) in label_info.items()))
    for cfg in ["rrf", "rerank", "plan"]:
        print(f"\n== {cfg} ==")
        print(f"{'index':15s} " + " ".join(f"{m:>17s}" for m in METRICS) + f" {'ctx tokens':>10s} {'coverage':>9s}")
        for name in label_info:
            r = results[(name, cfg)]
            cells = []
            for m in METRICS:
                mean, lo, hi = bootstrap_ci(r[m])
                cells.append(f"{mean:.2f} [{lo:.2f},{hi:.2f}]")
            print(f"{name:15s} " + " ".join(f"{c:>17s}" for c in cells)
                  + f" {sum(r['tokens']) / len(r['tokens']):>10.0f} {r['coverage']:>5}/{n_cmp}")
        base = results.get(("recursive350", cfg))
        if base:
            print(f"  paired nDCG@{K} vs recursive350:")
            for name in label_info:
                if name != "recursive350":
                    d, lo, hi, p = paired_bootstrap(base["ndcg"], results[(name, cfg)]["ndcg"])
                    print(f"    {name:14s} {d:+.3f} [{lo:+.3f},{hi:+.3f}]  p={p:.3f}" + ("  significant" if lo > 0 or hi < 0 else ""))

    kinds = sorted({x["type"] for x in items})
    print(f"\nhit@{K} by question type (plan):  " + "  ".join(kinds))
    for name in label_info:
        r = results[(name, "plan")]["hit"]
        print(f"  {name:14s} " + "  ".join(f"{sum(h for h, x in zip(r, items) if x['type'] == k):.0f}/{sum(x['type'] == k for x in items)}" for k in kinds))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"retrieval_{args.split}.json").write_text(json.dumps(
        {f"{n}|{c}": {m: (sum(v) / len(v) if isinstance(v, list) else v) for m, v in r.items()} for (n, c), r in results.items()}, indent=1))
    print(f"\nsaved results/retrieval_{args.split}.json")


if __name__ == "__main__":
    main()
