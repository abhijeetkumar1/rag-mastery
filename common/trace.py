"""Minimal tracing: one JSON line per request, containing timed spans for each pipeline stage.

This is the same data model as OpenTelemetry / Langfuse / LangSmith, just hand-rolled:
  trace = one request (a question)          -> id, name, attrs, total duration
  span  = one stage inside it (embed, search, generate, guardrails) -> name, start, duration, attrs

Written to data/traces/YYYY-MM-DD.jsonl. Read them back with load_traces() or phase1_naive_rag.06_traces.
"""
import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from common.config import DATA_DIR

TRACE_DIR = DATA_DIR / "traces"

# USD per 1M tokens as (input, output). List prices at time of writing; verify at openai.com/api/pricing.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float | None:
    if model not in PRICES:
        return None  # unknown or local model (e.g. bge-small: free)
    p_in, p_out = PRICES[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1e6


class Trace:
    def __init__(self, name: str, **attrs):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.attrs = dict(attrs)
        self.spans: list[dict] = []
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str, **attrs):
        """Time a stage. Yields the span's attrs dict so the caller can record results into it."""
        s = {"name": name, "start_ms": self._ms(), "attrs": dict(attrs)}
        t = time.perf_counter()
        try:
            yield s["attrs"]
        except Exception as e:
            s["attrs"]["error"] = repr(e)
            raise
        finally:
            s["duration_ms"] = round((time.perf_counter() - t) * 1000, 1)
            self.spans.append(s)

    def set(self, **attrs) -> None:
        self.attrs.update(attrs)

    def end(self) -> dict:
        record = {
            "trace_id": self.id,
            "name": self.name,
            "started_at": self.started_at,
            "duration_ms": self._ms(),
            "attrs": self.attrs,
            "spans": self.spans,
        }
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRACE_DIR / f"{self.started_at[:10]}.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return record

    def _ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 1)


def load_traces(name: str | None = None) -> list[dict]:
    rows = []
    for path in sorted(TRACE_DIR.glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if name is None or r["name"] == name]
