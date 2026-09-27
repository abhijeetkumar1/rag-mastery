"""Read the traces written by 04_ask: per-request timeline + aggregate metrics. No API calls.

Run: uv run python -m phase1_naive_rag.06_traces              # summary of all traces + last 5 requests
     uv run python -m phase1_naive_rag.06_traces --last 1     # timeline of the most recent request
     uv run python -m phase1_naive_rag.06_traces --json 1     # raw JSON of the most recent trace
     uv run python -m phase1_naive_rag.06_traces --name ask_v2  # Phase 2 pipeline traces

Observability = being able to answer "what happened on request X?" (traces) and
"how is the system doing overall?" (metrics aggregated from those traces).
"""
import argparse
import json

import numpy as np

from common.trace import load_traces


def timeline(t: dict) -> None:
    a = t["attrs"]
    flag = f"refused by {a.get('refused_by', 'LLM')}" if a.get("refused") else ("guardrails ✅" if a.get("guardrails_passed") else "guardrails ⚠")
    print(f"\n{t['started_at']}  trace={t['trace_id']}  total={t['duration_ms']:.0f} ms  [{flag}]")
    print(f"  Q: {a.get('question')}")
    scale = max(t["duration_ms"], 1) / 50  # 50-char wide bar
    for s in t["spans"]:
        pad = int(s["start_ms"] / scale)
        bar = "█" * max(1, int(s["duration_ms"] / scale))
        print(f"  {s['name']:14s} {s['duration_ms']:7.1f} ms  {' ' * pad}{bar}")
        at = s["attrs"]
        if at.get("top"):
            print(f"  {'':14s} top: " + ", ".join(f"{x[0]} ({x[1]})" for x in at["top"][:3]))
        if "route" in at:
            print(f"  {'':14s} route={at['route']} companies={at.get('companies')} cached={at.get('cached')}"
                  + (f" overrides={at['overrides']}" if at.get("overrides") else ""))
            for sq in at.get("sub_questions", []):
                print(f"  {'':14s}   sub: {sq}")
        if at.get("blocked") is not None:
            print(f"  {'':14s} best rerank score {at['best_score']} vs floor {at['floor']} -> blocked={at['blocked']}")
        if "hits" in at:
            print(f"  {'':14s} top hits: " + ", ".join(f"{h['id']} ({h['score']:.3f})" for h in at["hits"][:3]))
        if "input_tokens" in at:
            print(f"  {'':14s} tokens in/out: {at['input_tokens']}/{at['output_tokens']}  cost: ${at['cost_usd']:.5f}")
        if at.get("ungrounded_numbers") or at.get("invalid_citations"):
            print(f"  {'':14s} ungrounded: {at.get('ungrounded_numbers')}  invalid cites: {at.get('invalid_citations')}")


def summary(traces: list[dict]) -> None:
    total = np.array([t["duration_ms"] for t in traces])
    spans: dict[str, list[float]] = {}
    for t in traces:
        for s in t["spans"]:
            spans.setdefault(s["name"], []).append(s["duration_ms"])
    gen = [s["attrs"] for t in traces for s in t["spans"] if s["name"] == "generate" and "input_tokens" in s["attrs"]]
    # every LLM call records cost_usd (generate, and from Phase 3 also plan / multi_query / hyde)
    cost = sum(s["attrs"].get("cost_usd") or 0 for t in traces for s in t["spans"])
    refused = sum(bool(t["attrs"].get("refused")) for t in traces)
    failed = sum(t["attrs"].get("guardrails_passed") is False for t in traces)
    top1 = [s["attrs"]["hits"][0]["score"] for t in traces for s in t["spans"]  # phase 1: cosine
            if s["name"] == "vector_search" and s["attrs"].get("hits")]
    top1 += [s["attrs"]["top"][0][1] for t in traces for s in t["spans"]       # phase 2: rerank score
             if s["name"] == "rerank" and s["attrs"].get("top")]

    print(f"== {len(traces)} requests ==")
    print(f"latency   p50={np.percentile(total, 50):.0f} ms  p95={np.percentile(total, 95):.0f} ms  max={total.max():.0f} ms")
    for name, v in spans.items():
        print(f"  {name:14s} mean={np.mean(v):7.1f} ms   share of total={np.sum(v) / total.sum():5.1%}")
    if gen:
        print(f"tokens    mean in={np.mean([g['input_tokens'] for g in gen]):.0f}  "
              f"mean out={np.mean([g['output_tokens'] for g in gen]):.0f}")
    print(f"cost      total=${cost:.4f}  per request=${cost / len(traces):.5f}  "
          f"-> ${cost / len(traces) * 1e5:.0f} per 100k requests (all LLM calls, incl. refused requests)")
    print(f"quality   refusal rate={refused / len(traces):.0%}  guardrail warnings={failed / len(traces):.0%}  "
          + (f"top-1 retrieval score mean={np.mean(top1):.3f}" if top1 else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=5, help="show timelines of the N most recent requests")
    ap.add_argument("--json", type=int, default=0, help="dump raw JSON of the N most recent traces")
    ap.add_argument("--name", default="ask", help="trace name: ask (phase 1), ask_v2 (phase 2), ask_v3 (phase 3)")
    ap.add_argument("--tag", default=None, help="only traces with this tag (e.g. eval:test); default: untagged only")
    args = ap.parse_args()

    traces = [t for t in load_traces(args.name) if t["attrs"].get("tag") == args.tag]  # Phase 4 eval runs are tagged
    if not traces:
        raise SystemExit(f"No '{args.name}' traces yet: run the ask script of that phase first")
    if args.json:
        for t in traces[-args.json:]:
            print(json.dumps(t, indent=2))
        return
    summary(traces)
    for t in traces[-args.last:] if args.last > 0 else []:  # traces[-0:] would be ALL traces
        timeline(t)


if __name__ == "__main__":
    main()
