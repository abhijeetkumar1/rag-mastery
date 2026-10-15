"""Online evaluation: judge a SAMPLE of real answers, after the fact, off the request path.

Run: uv run python -m phase7_production.08_online_eval --dry-run      # what's queued, the estimated cost (no API)
     uv run python -m phase7_production.08_online_eval                # judge the queue (gpt-4.1, needs the API)
     uv run python -m phase7_production.08_online_eval --max 20

Offline eval (Phase 4) measures the golden set; online eval measures the questions users actually ask, which drift
away from any golden set. Real traffic has no reference answer, so only REFERENCE-FREE judgments are possible:
faithfulness (are the answer's claims supported by the passages it cited?), here Phase 4's judge v2. Correctness
would need a human or a reference: that's what thumbs-down traces are for (07_dashboard lists them).
The service samples ONLINE_EVAL_RATE (default 10%) of answered, non-cached requests into data/online_eval/queue.jsonl,
already PII-redacted. Judging never blocks a request: it runs as a batch (cron, a queue worker), and its cost is a
dial (the rate). Scores go to data/online_eval/scores.jsonl, keyed by trace_id, and into the dashboard.
"""
import argparse
import json

from common.chunking import count_tokens
from common.config import DATA_DIR, JUDGE_MODEL
from common.evaluation import judge_faithfulness_v2
from common.trace import cost_usd

QUEUE = DATA_DIR / "online_eval" / "queue.jsonl"
SCORES = DATA_DIR / "online_eval" / "scores.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max", type=int, default=100)
    args = ap.parse_args()
    queue = [json.loads(x) for x in QUEUE.read_text().splitlines()] if QUEUE.exists() else []
    done = {json.loads(x)["trace_id"] for x in SCORES.read_text().splitlines()} if SCORES.exists() else set()
    todo = [q for q in queue if q["trace_id"] not in done][: args.max]
    est_in = sum(count_tokens(q["answer"]) + sum(count_tokens(p) for p in q["passages"]) + 450 for q in todo)
    est = cost_usd(JUDGE_MODEL, est_in, 300 * len(todo)) or 0
    print(f"queue {len(queue)}, already judged {len(done)}, to judge now {len(todo)}: ~{est_in:,} input tokens, "
          f"~${est:.3f} with {JUDGE_MODEL}")
    by_route = {}
    for q in todo:
        by_route[q["route"]] = by_route.get(q["route"], 0) + 1
    print(f"  by route {by_route}; degraded (fallback) answers {sum(bool(q['degraded']) for q in todo)}")
    if args.dry_run or not todo:
        return
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    for q in todo:
        try:
            res, _ = judge_faithfulness_v2(q["answer"], q["passages"])
        except Exception as e:  # noqa: BLE001: the API is down; keep what's done, retry the rest next run
            print(f"  stopped: {type(e).__name__}: {str(e)[:120]}")
            break
        claims = res["claims"]
        row = {"trace_id": q["trace_id"], "route": q["route"], "judge": f"faithfulness_v2/{JUDGE_MODEL}",
               "faithfulness": sum(c["verdict"] == "supported" for c in claims) / len(claims) if claims else 1.0,
               "unsupported": [c["claim"] for c in claims if c["verdict"] != "supported"]}
        scores.append(row)
        with open(SCORES, "a") as f:
            f.write(json.dumps(row) + "\n")
    if scores:
        mean = sum(s["faithfulness"] for s in scores) / len(scores)
        print(f"judged {len(scores)}: mean faithfulness {mean:.2f}; below 1.0: {sum(s['faithfulness'] < 1 for s in scores)}")


if __name__ == "__main__":
    main()
