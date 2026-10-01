"""End-to-end evaluation of the Phase 6 pipelines vs Phase 5 (Phase 4's judges), on golden + multi-hop questions.

Run: uv run python -m phase6_agentic_rag.05_end_to_end --split dev                # build and choose here
     uv run python -m phase6_agentic_rag.05_end_to_end --split test3              # fresh split, ONCE
     uv run python -m phase6_agentic_rag.05_end_to_end --split dev --reuse        # re-grade saved answers

Pipelines:
  P5            Phase 5's 06_ask on structctx350 (the baseline)
  CRAG          common/crag.py, grader/rewriter/verifier = gpt-4o-mini (the chat model)
  CRAG-4.1      the same graph with gpt-4.1 as grader/rewriter/verifier
  ReAct         common/react_agent.py with gpt-4o-mini, default AgentLimits
  Hybrid        06_hybrid.route(): ReAct for aggregate / multi-company questions, P5 otherwise. COMPOSED from this run's
                P5 and ReAct answers (no third run: the route depends only on the question and the cached plan)
Questions: golden_<split>.jsonl (single-company + two-company comparisons + unanswerables) plus the multi-hop set
(multihop.py), reported both together and by type.

Faithfulness here = judge v2 on the passages the answer CITED (with their source line). Agents collect 5-25 passages
and cite a few; judging all of them would mostly measure the judge's patience. Not comparable with Phase 5's numbers.
"""
import argparse
import importlib
import json
import os
import time
from pathlib import Path

from common.agent_limits import AgentLimits
from common.guardrails import CITE_RE
from common.retriever import Retriever
from common.trace import load_traces

HERE = Path(__file__).parent
ask5 = importlib.import_module("phase5_advanced_indexing.06_ask")
e2e5 = importlib.import_module("phase5_advanced_indexing.09_end_to_end")
e2e4 = importlib.import_module("phase4_evaluation.04_end_to_end")
retrieval5 = importlib.import_module("phase5_advanced_indexing.05_retrieval_eval")
react = importlib.import_module("phase6_agentic_rag.02_react")
crag = importlib.import_module("phase6_agentic_rag.03_crag")
multihop = importlib.import_module("phase6_agentic_rag.multihop")
hybrid = importlib.import_module("phase6_agentic_rag.06_hybrid")
TRACES = {"P5": "ask_v5", "CRAG": "ask_v6_crag", "CRAG-4.1": "ask_v6_crag41", "ReAct": "ask_v6_react"}


def questions(split: str) -> list[dict]:
    return retrieval5.golden(split) + multihop.items(split)


def run(items: list[dict], names: list[str]) -> dict[str, list[dict]]:
    ix = ask5.Index("structctx350")
    ix.warmup()
    R: Retriever = ix.R
    fns = {"P5": lambda q: ask5.ask(ix, q, 6),
           "CRAG": lambda q: crag.ask(ix, q, trace_name="ask_v6_crag", grader_model="gpt-4o-mini"),
           "CRAG-4.1": lambda q: crag.ask(ix, q, trace_name="ask_v6_crag41", grader_model="gpt-4.1"),
           "ReAct": lambda q: react.ask(R, q, AgentLimits())}
    out = {}
    for name in names:
        started = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        rows = []
        for x in items:
            t = time.perf_counter()
            answer, hits, rep = fns[name](x["question"])
            cited = sorted({int(n) for n in CITE_RE.findall(answer) if 1 <= int(n) <= len(hits)})
            used = [hits[n - 1] for n in cited] or hits
            rows.append({"id": x["id"], "answer": answer, "refused": e2e4.refused(answer, rep),
                         "refused_by": rep.get("refused_by"), "passages": [e2e5.source_line(h) for h in used],
                         "n_evidence": len(hits), "hit_ids": [h["id"] for h in hits], "latency_s": time.perf_counter() - t,
                         "stop_reason": rep.get("stop_reason"), "steps": rep.get("steps"), "tool_calls": rep.get("tool_calls"),
                         "rounds": rep.get("rounds"), "fixes": rep.get("fixes"),
                         "verified": (rep.get("verdict") or {}).get("ok")})
        traces = {t["attrs"]["question"]: t for t in load_traces(TRACES[name])
                  if t["attrs"].get("tag") == os.environ["TRACE_TAG"] and t["started_at"] >= started}
        for r, x in zip(rows, items):
            t = traces.get(x["question"])
            spans = t["spans"] if t else []
            r["cost_usd"] = sum(s["attrs"].get("cost_usd") or 0 for s in spans) if t else None
            r["prompt_tokens"] = sum((s["attrs"].get("input_tokens") or 0) for s in spans) if t else None
            r["llm_calls"] = sum(1 for s in spans if s["attrs"].get("input_tokens")) if t else None
        out[name] = rows
        print(f"  ran {name}: {len(rows)} questions", flush=True)
    return out


def compose_hybrid(items: list[dict], runs: dict) -> None:
    rows = []
    for i, x in enumerate(items):
        target, why = hybrid.route(x["question"])
        src = runs["ReAct" if target == "react" else "P5"][i]
        rows.append({**{k: v for k, v in src.items() if k not in ("correct", "verdict", "judge_reasoning", "numbers_ok",
                                                                 "faithfulness", "unsupported_claims")},
                     "routed_to": target, "route_reason": why})
    runs["Hybrid"] = rows


def agent_stats(items: list[dict], runs: dict) -> None:
    print("\nCost and behaviour per question type (mean): LLM calls, prompt tokens, $/question, latency p50")
    kinds = sorted({x["type"] for x in items})
    for n, rows in runs.items():
        for k in kinds:
            rs = [r for r, x in zip(rows, items) if x["type"] == k]
            calls = [r["llm_calls"] for r in rs if r.get("llm_calls") is not None]
            toks = [r["prompt_tokens"] for r in rs if r.get("prompt_tokens") is not None]
            cost = [r["cost_usd"] for r in rs if r.get("cost_usd") is not None]
            lat = sorted(r["latency_s"] for r in rs)
            print(f"  {n:9s} {k:13s} calls {sum(calls) / max(len(calls), 1):4.1f}  tokens {sum(toks) / max(len(toks), 1):7.0f}  "
                  f"${sum(cost) / max(len(cost), 1):.5f}  p50 {lat[len(lat) // 2]:5.1f} s")
    for n, rows in runs.items():
        if n.startswith("CRAG"):
            print(f"  {n}: corrective rounds used on {sum(1 for r in rows if (r.get('rounds') or 0) > 0)}/{len(rows)}, "
                  f"regenerations {sum(1 for r in rows if (r.get('fixes') or 0) > 0)}, verifier rejected "
                  f"{sum(1 for r in rows if r.get('verified') is False)}")
        if n == "ReAct":
            stops = {}
            for r in rows:
                stops[r.get("stop_reason") or "answered"] = stops.get(r.get("stop_reason") or "answered", 0) + 1
            print(f"  ReAct stop reasons: {stops}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--pipelines", nargs="+", default=list(TRACES))
    ap.add_argument("--no-faithfulness", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--limit", type=int, help="first N golden + 1 unanswerable + 2 multi-hop (smoke test; results go to *_smoke files)")
    args = ap.parse_args()
    os.environ["TRACE_TAG"] = f"eval:{args.split}" + (":smoke" if args.limit else "")
    items = questions(args.split)
    if args.limit:
        items = (items[: args.limit] + [x for x in items if x["expected"] == "refuse"][:1]
                 + [x for x in items if x["type"] == "multihop"][:2])
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    tag = args.split + ("_smoke" if args.limit else "")
    answers = out / f"{tag}_answers.json"
    if args.reuse:
        runs = {k: v for k, v in json.loads(answers.read_text()).items() if k in args.pipelines}
    else:
        print(f"running {len(args.pipelines)} pipelines on {len(items)} {args.split} questions...", flush=True)
        runs = run(items, args.pipelines)
        answers.write_text(json.dumps(runs, indent=1))  # saved BEFORE grading
    if "P5" in runs and "ReAct" in runs:
        compose_hybrid(items, runs)
        print(f"Hybrid routed {sum(r['routed_to'] == 'react' for r in runs['Hybrid'])}/{len(items)} questions to ReAct")
    e2e5.grade(items, runs, not args.no_faithfulness)  # judge calls are cached: Hybrid re-uses P5/ReAct verdicts
    answers.write_text(json.dumps(runs, indent=1))
    summary = e2e5.report(items, runs, args.split, bases=("P5",))
    mh = [i for i, x in enumerate(items) if x["type"] == "multihop"]
    print(f"\nMulti-hop only ({len(mh)}): " + "  ".join(f"{n} {sum(runs[n][i]['correct'] for i in mh):.0f}/{len(mh)}" for n in runs))
    agent_stats(items, runs)
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nsaved results/{tag}_answers.json and results/{tag}_summary.json")


if __name__ == "__main__":
    main()
