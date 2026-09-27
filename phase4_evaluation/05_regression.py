"""Regression gate: compare the latest evaluation summary against a committed baseline; exit 1 on regression.

Run: uv run python -m phase4_evaluation.05_regression                     # check results/test_summary.json
     uv run python -m phase4_evaluation.05_regression --update-baseline   # accept the current numbers as baseline
     uv run python -m phase4_evaluation.05_regression --pipeline "P2 hybrid+rerank"

How it's used: every change to prompts, chunking, retrieval or models reruns 04_end_to_end, then this gate.
In CI (Phase 7) the same idea runs on a small, cheap smoke subset on every PR, and on the full set nightly.

Tolerances exist because LLM pipelines are noisy (sampling, API drift, judge variance): a gate that fails on
every ±1 question flips gets ignored. Quality metrics may not drop by more than the tolerance; latency and cost
may not grow by more than the given factor.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
BASELINE = HERE / "baseline.json"
# metric: (direction, tolerance). "higher" = must not drop more than tol; "lower" = must not rise more than tol;
# "factor" = must not exceed baseline × tol
RULES = {
    "correct": ("higher", 0.05),
    "refused_ok": ("higher", 0.0),
    "false_refusals": ("lower", 1),
    "faithfulness": ("higher", 0.05),
    "latency_p50_s": ("factor", 1.5),
    "cost_per_q": ("factor", 1.5),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--pipeline", default="P3 query intel.")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()
    current = json.loads((HERE / "results" / f"{args.split}_summary.json").read_text())[args.pipeline]

    if args.update_baseline or not BASELINE.exists():
        BASELINE.write_text(json.dumps({"split": args.split, "pipeline": args.pipeline, "metrics": current}, indent=1))
        print(f"baseline written: {BASELINE.name} ({args.pipeline} on {args.split})")
        return

    base = json.loads(BASELINE.read_text())
    if (base["split"], base["pipeline"]) != (args.split, args.pipeline):
        sys.exit(f"baseline is for {base['pipeline']} on {base['split']}; pass matching --split/--pipeline")
    failures = []
    print(f"{'metric':16s} {'baseline':>10s} {'current':>10s}  rule")
    for m, (direction, tol) in RULES.items():
        b, c = base["metrics"].get(m), current.get(m)
        if b is None or c is None:
            continue
        ok = (c >= b - tol) if direction == "higher" else (c <= b + tol) if direction == "lower" else (c <= b * tol)
        rule = {"higher": f"≥ baseline − {tol}", "lower": f"≤ baseline + {tol}", "factor": f"≤ baseline × {tol}"}[direction]
        print(f"{m:16s} {b:>10.4g} {c:>10.4g}  {rule:22s} {'✅' if ok else '❌ REGRESSION'}")
        if not ok:
            failures.append(m)
    if failures:
        sys.exit(f"\nFAILED: {', '.join(failures)}")
    print("\npassed")


if __name__ == "__main__":
    main()
