"""End-to-end evaluation: Phase 1 vs Phase 2 vs Phase 3 pipelines, answer by answer, on a golden split.

Run: uv run python -m phase4_evaluation.04_end_to_end --split test          # the number to report
     uv run python -m phase4_evaluation.04_end_to_end --split dev --no-faithfulness

For every question and pipeline:
  behaviour      answered or refused; for unanswerable questions, refusing is correct
  correctness    gpt-4.1 judge vs the reference answer (correct / partially / incorrect / refused), plus a
                 deterministic check that the reference figures appear (number_matches)
  faithfulness   judge v2 (gpt-4.1): share of the answer's claims supported by the passages the pipeline used
  latency, cost  from the pipeline's own trace (tagged eval:<split>, hidden from normal trace views)
Then: 95% bootstrap CIs, paired comparisons on the same questions, breakdowns, and results/<split>_*.json.
"""
import argparse
import importlib
import json
import os
import time
from pathlib import Path

from common.evaluation import (bootstrap_ci, faithfulness_score, judge_correctness, judge_faithfulness_v2,
                               number_matches, paired_bootstrap)
from common.guardrails import is_refusal
from common.retriever import Retriever
from common.store import collection_name, get_collection
from common.trace import load_traces

HERE = Path(__file__).parent
p1 = importlib.import_module("phase1_naive_rag.04_ask")
p2 = importlib.import_module("phase2_better_retrieval.05_ask")
p3 = importlib.import_module("phase3_query_intelligence.06_ask")
TRACE_NAMES = {"P1 naive": "ask", "P2 hybrid+rerank": "ask_v2", "P3 query intel.": "ask_v3"}


def refused(answer: str, report: dict) -> bool:
    return bool(report.get("refused")) or is_refusal(answer) or answer.startswith(("I can only answer", "I can tell you", "That question isn't"))


def run_pipelines(items: list[dict]) -> dict[str, list[dict]]:
    col = get_collection(collection_name("recursive", 350))
    R = Retriever()
    R.warmup()
    pipelines = {
        "P1 naive": lambda q: p1.ask(col, q, 5),
        "P2 hybrid+rerank": lambda q: p2.ask(R, q, 5),
        "P3 query intel.": lambda q: p3.ask(R, q, 6),
    }
    out = {}
    for name, fn in pipelines.items():
        rows = []
        for x in items:
            t = time.perf_counter()
            answer, hits, report = fn(x["question"])
            rows.append({"id": x["id"], "answer": answer, "refused": refused(answer, report),
                         "passages": [h["text"] for h in hits], "hit_ids": [h["id"] for h in hits],
                         "latency_s": time.perf_counter() - t})
        # cost per request from the traces this run just wrote (same name + tag, newest per question)
        traces = {t["attrs"]["question"]: t for t in load_traces(TRACE_NAMES[name]) if t["attrs"].get("tag") == os.environ["TRACE_TAG"]}
        for r, x in zip(rows, items):
            t = traces.get(x["question"])
            r["cost_usd"] = sum(s["attrs"].get("cost_usd") or 0 for s in t["spans"]) if t else None
        out[name] = rows
        print(f"  ran {name}: {len(rows)} questions")
    return out


def grade(items: list[dict], runs: dict, faithfulness: bool) -> None:
    for name, rows in runs.items():
        for r, x in zip(rows, items):
            if x["expected"] == "refuse":
                r["correct"] = float(r["refused"])
                r["verdict"] = "refused" if r["refused"] else "answered_unanswerable"
                continue
            j, _ = judge_correctness(x["question"], x["answer"], r["answer"])
            r["verdict"] = "refused" if r["refused"] else j["verdict"]
            r["judge_reasoning"] = j["reasoning"]
            r["correct"] = float(r["verdict"] == "correct")
            r["numbers_ok"] = float(all(number_matches(n, r["answer"]) for n in x["numbers"])) if x["numbers"] else None
            if faithfulness and not r["refused"] and r["passages"]:
                f, _ = judge_faithfulness_v2(r["answer"], r["passages"])
                r["faithfulness"] = faithfulness_score(f)
                r["unsupported_claims"] = [c["claim"] + " | " + c["check"] for c in f["claims"] if c["verdict"] != "supported"]


def report(items: list[dict], runs: dict, split: str) -> dict:
    names = list(runs)
    summary = {}
    ans = [i for i, x in enumerate(items) if x["expected"] == "answer"]
    una = [i for i, x in enumerate(items) if x["expected"] == "refuse"]
    print(f"\n== {split}: {len(ans)} answerable + {len(una)} unanswerable questions ==\n")
    print(f"{'pipeline':18s} {'correct (answerable)':>22s} {'refused ok':>11s} {'false refusals':>15s} {'faithfulness':>14s} "
          f"{'latency p50':>12s} {'$/question':>11s}")
    for n in names:
        rows = runs[n]
        c = [rows[i]["correct"] for i in ans]
        mean, lo, hi = bootstrap_ci(c)
        ref_ok = sum(rows[i]["refused"] for i in una)
        false_ref = sum(rows[i]["refused"] for i in ans)
        faith = [rows[i]["faithfulness"] for i in ans if rows[i].get("faithfulness") is not None]
        lat = sorted(r["latency_s"] for r in rows)
        costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
        fs = f"{sum(faith) / len(faith):.2f} (n={len(faith)})" if faith else "-"
        print(f"{n:18s} {mean:>8.2f} [{lo:.2f},{hi:.2f}] {ref_ok:>7}/{len(una)} {false_ref:>11}/{len(ans)} {fs:>14s} "
              f"{lat[len(lat) // 2]:>10.1f} s {sum(costs) / max(len(costs), 1):>11.5f}")
        summary[n] = {"correct": mean, "correct_ci": [lo, hi], "refused_ok": ref_ok / len(una), "false_refusals": false_ref,
                      "faithfulness": sum(faith) / len(faith) if faith else None, "latency_p50_s": lat[len(lat) // 2],
                      "cost_per_q": sum(costs) / max(len(costs), 1)}

    print("\nPaired differences in correctness (same answerable questions): mean [95% CI], P(no improvement)")
    for a, b in [(names[0], names[1]), (names[1], names[2]), (names[0], names[2])]:
        d, lo, hi, p = paired_bootstrap([runs[a][i]["correct"] for i in ans], [runs[b][i]["correct"] for i in ans])
        print(f"  {b:18s} vs {a:18s} {d:+.3f} [{lo:+.3f},{hi:+.3f}]  p={p:.3f}" + ("  significant" if lo > 0 or hi < 0 else ""))

    kinds = sorted({items[i]["type"] for i in ans})
    print("\nCorrect by question type")
    print(f"  {'pipeline':18s} " + " ".join(f"{k:>11s}" for k in kinds))
    for n in names:
        cells = [f"{sum(runs[n][i]['correct'] for i in ans if items[i]['type'] == k):.0f}/{sum(items[i]['type'] == k for i in ans)}" for k in kinds]
        print(f"  {n:18s} " + " ".join(f"{c:>11s}" for c in cells))

    print("\nVerdict distribution (answerable)")
    for n in names:
        dist = {}
        for i in ans:
            dist[runs[n][i]["verdict"]] = dist.get(runs[n][i]["verdict"], 0) + 1
        print(f"  {n:18s} {dist}")
    nums = [i for i in ans if items[i]["numbers"]]
    print(f"\nDeterministic check (reference figures present in the answer), {len(nums)} questions with figures:")
    for n in names:
        agree = sum((runs[n][i]["numbers_ok"] == 1.0) == (runs[n][i]["correct"] == 1.0) for i in nums)
        print(f"  {n:18s} figures present {sum(runs[n][i]['numbers_ok'] for i in nums):.0f}/{len(nums)}   "
              f"agrees with the LLM judge on {agree}/{len(nums)}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--no-faithfulness", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="grade the saved answers instead of re-running the pipelines")
    args = ap.parse_args()
    os.environ["TRACE_TAG"] = f"eval:{args.split}"
    items = [json.loads(line) for line in (HERE / f"golden_{args.split}.jsonl").read_text().splitlines()]
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    answers = out / f"{args.split}_answers.json"
    if args.reuse:
        runs = json.loads(answers.read_text())
    else:
        print(f"running 3 pipelines on {len(items)} {args.split} questions...")
        runs = run_pipelines(items)
        answers.write_text(json.dumps(runs, indent=1))  # saved BEFORE grading: a judge crash mustn't cost the answers
    grade(items, runs, not args.no_faithfulness)
    answers.write_text(json.dumps(runs, indent=1))      # now with verdicts
    summary = report(items, runs, args.split)
    (out / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nsaved results/{args.split}_answers.json and results/{args.split}_summary.json")


if __name__ == "__main__":
    main()
