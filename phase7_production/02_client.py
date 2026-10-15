"""Streaming client: prints the answer as it arrives, plus the events around it (route, agent steps, sources,
guardrail verdict, latency and cost). Optionally sends feedback for the trace.

Run: uv run python -m phase7_production.02_client "What was Apple's total net sales in fiscal 2025?"
     uv run python -m phase7_production.02_client --key demo-key-aapl "What was Microsoft's revenue in fiscal 2025?"
     uv run python -m phase7_production.02_client --mode agent "Which of the five companies grew revenue fastest?"
     uv run python -m phase7_production.02_client --feedback down "..."
"""
import argparse
import json
import time

import httpx


def stream(base: str, key: str, question: str, mode: str = "auto", quiet: bool = False) -> dict:
    """Consume /ask/stream. Returns {"events": [...], "answer", "ttft_s", "total_s", "status"}."""
    out = {"events": [], "answer": "", "ttft_s": None, "status": None}
    t0 = time.perf_counter()
    with httpx.stream("POST", f"{base}/ask/stream", json={"question": question, "mode": mode},
                      headers={"X-API-Key": key}, timeout=120) as r:
        out["status"] = r.status_code
        if r.status_code != 200:
            out["error"] = r.read().decode()
            if not quiet:
                print(f"HTTP {r.status_code}: {out['error']}  {dict(r.headers).get('retry-after', '')}")
            return out
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event:
                data = json.loads(line[5:].strip())
                out["events"].append((event, data))
                if event == "token":
                    out["ttft_s"] = out["ttft_s"] or time.perf_counter() - t0
                    out["answer"] += data["text"]
                if not quiet:
                    show(event, data)
    out["total_s"] = time.perf_counter() - t0
    return out


def show(event: str, d: dict) -> None:
    if event == "token":
        print(d["text"], end="", flush=True)
    elif event == "meta":
        print(f"[meta] trace {d['trace_id']}  route={d['route']}  cache={d['cache']}  degraded={d.get('degraded')}"
              + (f"  pii_redacted={d['pii_redacted']}" if d.get("pii_redacted") else "") + "\n")
    elif event == "step":
        what = ", ".join(f"{c['name']}({json.dumps(c['args'])[:80]})" for c in d.get("tool_calls", [])) or \
            (f"{d.get('results')} results, evidence {d.get('evidence')}" if d["node"] == "tools" else d.get("stop_reason", ""))
        print(f"[step {d['step']}] {d['node']}: {what}")
    elif event == "sources":
        print("[sources] " + "  ".join(f"[{s['n']}] {s['ticker']} FY{s['fiscal_year']} {s['item']}" for s in d["sources"]) + "\n")
    elif event == "guardrails":
        ok = "passed" if d.get("passed") else f"FAILED ungrounded={d.get('ungrounded_numbers')} invalid={d.get('invalid_citations')}"
        print(f"\n\n[guardrails] {ok}" + (" (answer withheld)" if d.get("withheld") else ""))
    elif event == "done":
        print(f"[done] {d['latency_ms']:.0f} ms (first token {d['ttft_ms']} ms)  ${d['cost_usd']:.5f}  cache={d['cache']}  "
              f"degraded={d['degraded']}")
    elif event == "error":
        print(f"[error] {d['code']}: {d['message']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="demo-key-all")
    ap.add_argument("--mode", default="auto", choices=["auto", "pipeline", "agent"])
    ap.add_argument("--feedback", choices=["up", "down"])
    args = ap.parse_args()
    out = stream(args.base, args.key, args.question, args.mode)
    trace_id = next((d["trace_id"] for e, d in out["events"] if "trace_id" in d), None)
    if args.feedback and trace_id:
        r = httpx.post(f"{args.base}/feedback", json={"trace_id": trace_id, "rating": args.feedback},
                       headers={"X-API-Key": args.key})
        print(f"[feedback] {args.feedback} -> {r.status_code}")


if __name__ == "__main__":
    main()
