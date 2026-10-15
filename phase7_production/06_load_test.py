"""Load test: latency percentiles, time to first token and throughput as concurrency grows, plus what the cache and
the guardrail mode change. Runs the real server (uvicorn subprocess) with LLM_BACKEND=stub, so it costs nothing and
measures OUR overhead (planning, retrieval, rerank, guardrails, serialization) plus a simulated LLM stream.

Run: uv run python -m phase7_production.06_load_test                    # the standard scenarios
     uv run python -m phase7_production.06_load_test --quick             # fewer requests
     uv run python -m phase7_production.06_load_test --workers 2         # also the 2-process run

Scenarios (each on a fresh server, so caches start empty):
  concurrency 1 / 4 / 16, all-distinct golden dev questions        latency vs load (where does it queue?)
  concurrency 4, every question asked twice: later / at the same time   exact-cache hits, and a cache stampede
  concurrency 4, GUARDRAIL_MODE=buffer                              time to first token when the answer is held back
The stub streams the extractive answer at STUB_TOKEN_MS per word (default 15 ms ~ gpt-4o-mini's ~70 tokens/s).
Caveat: with the stub, the query-embedding API isn't called (cached embeddings are used, misses fall back to BM25),
so real latency adds ~100-300 ms per embedding call and the real LLM's time to first token (~300-700 ms).
"""
import argparse
import asyncio
import importlib
import json
import os
import subprocess
import sys
import time

import httpx
import numpy as np

retrieval5 = importlib.import_module("phase5_advanced_indexing.05_retrieval_eval")
KEY = "demo-key-load"
WARMUP = ["What does Apple say about its retail stores?", "How does Tesla describe its Supercharger network?"]


def start_server(port: int, env_extra: dict, workers: int = 1) -> subprocess.Popen:
    env = {**os.environ, "LLM_BACKEND": "stub", "PHOENIX_TRACING": "0", "ONLINE_EVAL_RATE": "0",
           "TRACE_TAG": "loadtest", **env_extra}
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "phase7_production.app:app", "--port", str(port),
                          "--workers", str(workers), "--log-level", "warning"], env=env)
    for _ in range(180):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/ready", timeout=1).status_code == 200:
                return p
        except httpx.HTTPError:
            pass
        time.sleep(1)
    p.kill()
    raise SystemExit("server did not become ready")


async def one(client: httpx.AsyncClient, base: str, q: str) -> dict:
    t0 = time.perf_counter()
    ttft, status, cache, event = None, None, None, None
    async with client.stream("POST", f"{base}/ask/stream", json={"question": q, "mode": "pipeline"},
                             headers={"X-API-Key": KEY}) as r:
        status = r.status_code
        async for line in r.aiter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                if event == "token" and ttft is None:
                    ttft = time.perf_counter() - t0
                if event == "meta":
                    cache = json.loads(line[5:])["cache"]
                if event == "error":
                    status = "error:" + json.loads(line[5:])["code"]
    return {"total": time.perf_counter() - t0, "ttft": ttft, "status": status, "cache": cache}


async def run(base: str, questions: list[str], concurrency: int) -> tuple[list[dict], float]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=300) as client:
        async def go(q):
            async with sem:
                return await one(client, base, q)
        t = time.perf_counter()
        res = await asyncio.gather(*(go(q) for q in questions))
        return res, time.perf_counter() - t


def report(name: str, res: list[dict], wall: float) -> dict:
    tot = np.array([r["total"] for r in res])
    tt = np.array([r["ttft"] for r in res if r["ttft"] is not None])
    ok = sum(r["status"] == 200 for r in res)
    hits = sum(r["cache"] in ("exact", "semantic") for r in res)
    row = {"scenario": name, "n": len(res), "ok": ok, "cache_hits": hits, "throughput_rps": round(len(res) / wall, 2),
           "p50_s": round(float(np.median(tot)), 2), "p95_s": round(float(np.percentile(tot, 95)), 2),
           "ttft_p50_s": round(float(np.median(tt)), 2) if len(tt) else None,
           "ttft_p95_s": round(float(np.percentile(tt, 95)), 2) if len(tt) else None}
    print(f"  {name:34s} n={row['n']:3d} ok={ok:3d} cache={hits:3d}  {row['throughput_rps']:5.2f} req/s   "
          f"total p50 {row['p50_s']:5.2f} s  p95 {row['p95_s']:5.2f} s   first token p50 {row['ttft_p50_s']} s  p95 {row['ttft_p95_s']} s",
          flush=True)
    errs = {r["status"] for r in res if r["status"] != 200}
    if errs:
        print(f"      non-200: {errs}")
    return row


def scenario(name: str, port: int, env: dict, questions: list[str], concurrency: int, workers: int = 1) -> dict:
    p = start_server(port, env, workers)
    try:
        # warm the first-call paths (Chroma, BM25) with questions NOT in the measured set (they'd be cache hits)
        asyncio.run(run(f"http://127.0.0.1:{port}", WARMUP, 1))
        res, wall = asyncio.run(run(f"http://127.0.0.1:{port}", questions, concurrency))
        return report(name, res, wall)
    finally:
        p.terminate()
        p.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()
    qs = [x["question"] for x in retrieval5.golden("dev") if x["expected"] != "refuse"]
    n = 16 if args.quick else 48
    distinct = (qs * 2)[:n]
    rows = []
    print(f"stub LLM at {os.getenv('STUB_TOKEN_MS', '15')} ms/word; {n} requests per scenario\n")
    for c in (1, 4, 16):
        rows.append(scenario(f"concurrency {c}", args.port, {}, distinct, c))
    half = distinct[: n // 2]
    # the repeat arrives after its first answer is cached, vs while the first is still in flight (a cache stampede)
    rows.append(scenario("concurrency 4, repeats later", args.port, {}, half + half, 4))
    rows.append(scenario("concurrency 4, repeats concurrent", args.port, {}, [q for q in half for _ in (0, 1)], 4))
    rows.append(scenario("concurrency 4, GUARDRAIL_MODE=buffer", args.port, {"GUARDRAIL_MODE": "buffer"}, distinct, 4))
    if args.workers > 1:
        rows.append(scenario(f"concurrency 16, {args.workers} workers", args.port, {}, distinct, 16, args.workers))
    out = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "load_test.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\nsaved results/load_test.json")


if __name__ == "__main__":
    main()
