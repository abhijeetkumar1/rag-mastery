"""Read the traces written by 04_ask: per-request timeline + aggregate metrics. No API calls.

Run: uv run python -m phase1_naive_rag.06_traces              # summary of all traces + last 5 requests
     uv run python -m phase1_naive_rag.06_traces --last 1     # timeline of the most recent request
     uv run python -m phase1_naive_rag.06_traces --json 1     # raw JSON of the most recent trace

Observability = being able to answer "what happened on request X?" (traces) and
"how is the system doing overall?" (metrics aggregated from those traces).
"""
import argparse
import json

import numpy as np

from common.trace import load_traces


def timeline(t: dict) -> None:
    a = t["attrs"]
    flag = "refused" if a.get("refused") else ("guardrails ✅" if a.get("guardrails_passed") else "guardrails ⚠")
    print(f"\n{t['started_at']}  trace={t['trace_id']}  total={t['duration_ms']:.0f} ms  [{flag}]")
    print(f"  Q: {a.get('question')}")
    scale = max(t["duration_ms"], 1) / 50  # 50-char wide bar
    for s in t["spans"]:
        pad = int(s["start_ms"] / scale)
        bar = "█" * max(1, int(s["duration_ms"] / scale))
        print(f"  {s['name']:14s} {s['duration_ms']:7.1f} ms  {' ' * pad}{bar}")
        at = s["attrs"]
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
    cost = sum(g["cost_usd"] or 0 for g in gen)
    refused = sum(bool(t["attrs"].get("refused")) for t in traces)
    failed = sum(t["attrs"].get("guardrails_passed") is False for t in traces)
    top1 = [s["attrs"]["hits"][0]["score"] for t in traces for s in t["spans"]
            if s["name"] == "vector_search" and s["attrs"].get("hits")]

    print(f"== {len(traces)} requests ==")
    print(f"latency   p50={np.percentile(total, 50):.0f} ms  p95={np.percentile(total, 95):.0f} ms  max={total.max():.0f} ms")
    for name, v in spans.items():
        print(f"  {name:14s} mean={np.mean(v):7.1f} ms   share of total={np.sum(v) / total.sum():5.1%}")
    if gen:
        print(f"tokens    mean in={np.mean([g['input_tokens'] for g in gen]):.0f}  "
              f"mean out={np.mean([g['output_tokens'] for g in gen]):.0f}")
        print(f"cost      total=${cost:.4f}  per request=${cost / len(gen):.5f}  "
              f"-> ${cost / len(gen) * 1e5:.0f} per 100k requests")
    print(f"quality   refusal rate={refused / len(traces):.0%}  guardrail warnings={failed / len(traces):.0%}  "
          f"top-1 retrieval score mean={np.mean(top1):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=5, help="show timelines of the N most recent requests")
    ap.add_argument("--json", type=int, default=0, help="dump raw JSON of the N most recent traces")
    args = ap.parse_args()

    traces = load_traces("ask")
    if not traces:
        raise SystemExit("No traces yet: run phase1_naive_rag.04_ask first")
    if args.json:
        for t in traces[-args.json:]:
            print(json.dumps(t, indent=2))
        return
    summary(traces)
    for t in traces[-args.last:] if args.last > 0 else []:  # traces[-0:] would be ALL traces
        timeline(t)


if __name__ == "__main__":
    main()
