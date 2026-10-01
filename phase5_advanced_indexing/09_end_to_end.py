"""End-to-end evaluation of the Phase 5 indexes (Phase 4's judges and statistics), answer by answer.

Run: uv run python -m phase5_advanced_indexing.09_end_to_end --split dev                 # tune / choose here
     uv run python -m phase5_advanced_indexing.09_end_to_end --split test2               # fresh held-out split, ONCE
     uv run python -m phase5_advanced_indexing.09_end_to_end --split dev --reuse         # re-grade saved answers

Pipelines (all with Phase 3's planner; P5 = 06_ask: delimited prompt + injection scan):
  P3                  Phase 3's 06_ask on Phase 1 chunks: the Phase 4 baseline, unchanged
  P5 <index>          06_ask on that index; "P5 recursive350" isolates the new prompt from the new index

One change from Phase 4: the faithfulness judge sees each passage WITH the source line the generator saw
("TSLA 10-K FY2025, Item 7: ..."). Phase 4 gave the judge the bare chunk text, so a correct answer whose year was
only in the source line was judged "period unverifiable" although the generator had that information.
"""
import argparse
import importlib
import json
import os
import time
from pathlib import Path

from common.evaluation import bootstrap_ci, faithfulness_score, judge_correctness, judge_faithfulness_v2, number_matches, paired_bootstrap
from common.retriever import Retriever
from common.trace import load_traces

HERE = Path(__file__).parent
ask3 = importlib.import_module("phase3_query_intelligence.06_ask")
ask5 = importlib.import_module("phase5_advanced_indexing.06_ask")
e2e4 = importlib.import_module("phase4_evaluation.04_end_to_end")
retrieval5 = importlib.import_module("phase5_advanced_indexing.05_retrieval_eval")
DEFAULT = ["P3", "P5 recursive350", "P5 structctx350", "P5 llmctx350", "P5 parent-child", "P5 parent-child@3"]


def source_line(h: dict) -> str:
    m = h["meta"]
    return f"{m['ticker']} 10-K FY{m['fiscal_year']}, {m['item']}: {m['section']}\n{h['text']}"


def run(items: list[dict], names: list[str]) -> dict[str, list[dict]]:
    out = {}
    for name in names:
        if name == "P3":
            R = Retriever()
            R.warmup()
            fn, trace_name = (lambda q: ask3.ask(R, q, 6)), "ask_v3"
        else:
            ix, k = ask5.open_index(name.split(" ", 1)[1])
            ix.warmup()
            fn, trace_name = (lambda q, ix=ix, k=k: ask5.ask(ix, q, k or 6)), "ask_v5"
        rows = []
        started = time.time()
        for x in items:
            t = time.perf_counter()
            answer, hits, report = fn(x["question"])
            rows.append({"id": x["id"], "answer": answer, "refused": e2e4.refused(answer, report),
                         "refused_by": report.get("refused_by"), "best_score": report.get("best_score"),
                         "passages": [source_line(h) for h in hits], "hit_ids": [h["id"] for h in hits],
                         "quarantined": report.get("quarantined", []), "latency_s": time.perf_counter() - t})
        # cost from this run's traces (same tag; newest trace per question, written after `started`)
        traces = {}
        for t in load_traces(trace_name):
            if t["attrs"].get("tag") == os.environ["TRACE_TAG"] and t["started_at"] >= time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(started)):
                traces[t["attrs"]["question"]] = t
        for r, x in zip(rows, items):
            t = traces.get(x["question"])
            r["cost_usd"] = sum(s["attrs"].get("cost_usd") or 0 for s in t["spans"]) if t else None
            r["prompt_tokens"] = sum(s["attrs"].get("input_tokens") or 0 for s in t["spans"] if s["name"] == "generate") if t else None
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
            if "judge_reasoning" not in r:  # --reuse keeps earlier verdicts (judge calls are cached anyway)
                j, _ = judge_correctness(x["question"], x["answer"], r["answer"])
                r["verdict"] = "refused" if r["refused"] else j["verdict"]
                r["judge_reasoning"] = j["reasoning"]
                r["correct"] = float(r["verdict"] == "correct")
                r["numbers_ok"] = float(all(number_matches(n, r["answer"]) for n in x["numbers"])) if x["numbers"] else None
            if faithfulness and not r["refused"] and r["passages"] and "faithfulness" not in r:
                f, _ = judge_faithfulness_v2(r["answer"], r["passages"])
                r["faithfulness"] = faithfulness_score(f)
                r["unsupported_claims"] = [c["claim"] + " | " + c["check"] for c in f["claims"] if c["verdict"] != "supported"]


def report(items: list[dict], runs: dict, split: str) -> dict:
    ans = [i for i, x in enumerate(items) if x["expected"] == "answer"]
    una = [i for i, x in enumerate(items) if x["expected"] == "refuse"]
    print(f"\n== {split}: {len(ans)} answerable + {len(una)} unanswerable ==\n")
    print(f"{'pipeline':20s} {'correct [95% CI]':>19s} {'refused ok':>10s} {'false ref.':>10s} {'faithful':>9s} "
          f"{'prompt tok':>10s} {'p50 s':>6s} {'$/q':>8s}")
    summary = {}
    for n, rows in runs.items():
        mean, lo, hi = bootstrap_ci([rows[i]["correct"] for i in ans])
        faith = [rows[i]["faithfulness"] for i in ans if rows[i].get("faithfulness") is not None]
        lat = sorted(r["latency_s"] for r in rows)
        toks = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
        costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
        s = {"correct": mean, "correct_ci": [lo, hi], "refused_ok": sum(rows[i]["refused"] for i in una) / len(una),
             "false_refusals": sum(rows[i]["refused"] for i in ans), "faithfulness": sum(faith) / len(faith) if faith else None,
             "prompt_tokens": sum(toks) / len(toks) if toks else None, "latency_p50_s": lat[len(lat) // 2],
             "cost_per_q": sum(costs) / max(len(costs), 1)}
        summary[n] = s
        print(f"{n:20s} {mean:>7.2f} [{lo:.2f},{hi:.2f}] {sum(rows[i]['refused'] for i in una):>6}/{len(una)} "
              f"{s['false_refusals']:>6}/{len(ans)} {s['faithfulness'] or 0:>9.2f} {s['prompt_tokens'] or 0:>10.0f} "
              f"{s['latency_p50_s']:>6.1f} {s['cost_per_q']:>8.5f}")

    names = list(runs)
    print("\nPaired differences in correctness (same answerable questions)")
    for base in [b for b in ("P3", "P5 recursive350") if b in runs]:
        for n in names:
            if n != base and not (base == "P5 recursive350" and n == "P3"):
                d, lo, hi, p = paired_bootstrap([runs[base][i]["correct"] for i in ans], [runs[n][i]["correct"] for i in ans])
                print(f"  {n:20s} vs {base:16s} {d:+.3f} [{lo:+.3f},{hi:+.3f}]  p={p:.3f}" + ("  significant" if lo > 0 or hi < 0 else ""))

    kinds = sorted({items[i]["type"] for i in ans})
    print(f"\nCorrect by type: {'  '.join(kinds)}")
    for n in names:
        print(f"  {n:20s} " + "  ".join(f"{sum(runs[n][i]['correct'] for i in ans if items[i]['type'] == k):.0f}/"
                                         f"{sum(items[i]['type'] == k for i in ans)}" for k in kinds))
    print("\nVerdicts (answerable) and how unanswerables were refused")
    for n in names:
        dist = {}
        for i in ans:
            dist[runs[n][i]["verdict"]] = dist.get(runs[n][i]["verdict"], 0) + 1
        by = [runs[n][i].get("refused_by") or ("llm" if runs[n][i]["refused"] else "ANSWERED") for i in una]
        print(f"  {n:20s} {dist}  unanswerable: {by}")
    if len(names) > 1:
        a, b = names[0], names[-1]
        flips = [(items[i]["question"][:70], runs[a][i]["verdict"], runs[b][i]["verdict"]) for i in ans
                 if runs[a][i]["correct"] != runs[b][i]["correct"]]
        print(f"\nQuestions where {a} and {b} differ ({len(flips)}):")
        for q, va, vb in flips:
            print(f"  {va:>18s} → {vb:18s} {q}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--pipelines", nargs="+", default=DEFAULT)
    ap.add_argument("--no-faithfulness", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="grade the saved answers instead of re-running")
    args = ap.parse_args()
    os.environ["TRACE_TAG"] = f"eval:{args.split}"
    items = retrieval5.golden(args.split)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    answers = out / f"{args.split}_answers.json"
    if args.reuse:
        runs = {k: v for k, v in json.loads(answers.read_text()).items() if k in args.pipelines}
    else:
        print(f"running {len(args.pipelines)} pipelines on {len(items)} {args.split} questions...")
        runs = run(items, args.pipelines)
        answers.write_text(json.dumps(runs, indent=1))  # saved BEFORE grading
    grade(items, runs, not args.no_faithfulness)
    answers.write_text(json.dumps(runs, indent=1))
    summary = report(items, runs, args.split)
    (out / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nsaved results/{args.split}_answers.json and results/{args.split}_summary.json")


if __name__ == "__main__":
    main()
