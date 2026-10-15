"""Production telemetry (Phase 7): OpenTelemetry traces and Prometheus metrics.

TRACES: the service keeps writing our JSONL Trace (Phases 1-6; evaluation reads it) and EXPORTS each finished trace
as OpenTelemetry spans. OTel is the vendor-neutral standard: one exporter config sends the same spans to Phoenix,
Jaeger, Tempo, Datadog, Honeycomb or Langfuse (all accept OTLP). The bridge rebuilds the span tree from our
record (start offsets and durations), so no pipeline code has to change.
  root span   "rag.request": question (already PII-redacted), user, route, cache, refused, cost
  children    one per stage (plan, dense_search, bm25_search, fusion, rerank, generate, guardrails, ...)
Attribute names follow two conventions so backends can render them:
  OTel GenAI semantic conventions   gen_ai.request.model, gen_ai.usage.input_tokens / output_tokens
  OpenInference (Phoenix)           openinference.span.kind = CHAIN | RETRIEVER | RERANKER | LLM | GUARDRAIL
Exporters: OTLP/HTTP to Phoenix (:6006/v1/traces) when it is listening, plus a JSONL file exporter
(data/otel/spans.jsonl) so the export can be inspected without any backend.

METRICS: Prometheus counters and histograms, scraped from GET /metrics. Traces answer "what happened in THIS
request"; metrics answer "how is the system doing" (rates, percentiles, totals) cheaply, for dashboards and alerts.
Label values must have LOW cardinality (route, stage, outcome), never a user id or a question: each distinct label
set is a separate time series.
"""
import json
from datetime import datetime

from prometheus_client import CollectorRegistry, Counter, Histogram

from common.config import DATA_DIR, PHOENIX_ENDPOINT
from common.observability import _reachable
from common.pii import redact

OTEL_DIR = DATA_DIR / "otel"
_tracer = None
_provider = None

SPAN_KIND = {"plan": "CHAIN", "dense_search": "RETRIEVER", "bm25_search": "RETRIEVER", "fusion": "RETRIEVER",
             "multi_query_fusion": "RETRIEVER", "rerank": "RERANKER", "generate": "LLM", "guardrails": "GUARDRAIL",
             "injection_scan": "GUARDRAIL", "relevance_floor": "GUARDRAIL", "input_guardrails": "GUARDRAIL",
             "access_check": "GUARDRAIL", "access_violation": "GUARDRAIL", "output_check": "GUARDRAIL",
             "cache_lookup": "CHAIN", "agent": "AGENT", "agent_step": "LLM", "finalize": "LLM"}
RENAME = {"input_tokens": "gen_ai.usage.input_tokens", "output_tokens": "gen_ai.usage.output_tokens",
          "model": "gen_ai.request.model"}


class _FileExporter:
    """Minimal SpanExporter: one JSON object per span. Same data an OTLP backend receives."""

    def __init__(self, path):
        self.path = path

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            for s in spans:
                f.write(json.dumps(json.loads(s.to_json())) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000):
        return True


def setup_otel(service: str = "rag-mastery", otlp: bool = True, file: bool = True):
    """A private TracerProvider (not the global one, which Phoenix's register() may own). Returns the tracer."""
    global _tracer, _provider
    if _tracer is not None:
        return _tracer
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    _provider = TracerProvider(resource=Resource.create({"service.name": service, "openinference.project.name": service}))
    if otlp and _reachable(PHOENIX_ENDPOINT):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{PHOENIX_ENDPOINT}/v1/traces")))
    if file:
        _provider.add_span_processor(BatchSpanProcessor(_FileExporter(OTEL_DIR / "spans.jsonl")))
    _tracer = _provider.get_tracer("rag-mastery.phase7")
    return _tracer


def _attrs(d: dict) -> dict:
    """OTel attribute values must be str/bool/int/float (or lists of one type): everything else becomes JSON.
    Strings are redacted again here: the exporter is the last place PII can be stopped before it leaves."""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        if not isinstance(v, (str, bool, int, float)):
            v = json.dumps(v, default=str)
        if isinstance(v, str):
            v = redact(v)[0][:4000]
        out[RENAME.get(k, k)] = v
    return out


def export_trace(record: dict) -> None:
    """One finished common.trace record -> an OTel span tree with the original timings."""
    if _tracer is None:
        return
    from opentelemetry import trace
    t0 = int(datetime.fromisoformat(record["started_at"]).timestamp() * 1e9)
    root = _tracer.start_span("rag.request", start_time=t0, attributes={
        "openinference.span.kind": "CHAIN", "rag.trace_id": record["trace_id"], "rag.pipeline": record["name"],
        **_attrs(record["attrs"])})
    # our spans are flat (no parent ids); nesting is recovered: a span is appended when it ENDS, so a parent comes after
    # its children in the record, and its interval contains theirs (agent_step and tool:* inside "agent"). Time alone
    # isn't enough: timings are rounded to 0.1 ms, so a 0 ms span right after "plan" would look contained in it; a
    # child must also START strictly after its parent.
    rec = record["spans"]
    iv = [(s["start_ms"], s["start_ms"] + s.get("duration_ms", 0)) for s in rec]
    parent_of = {}
    for i in range(len(rec)):
        cands = [j for j in range(i + 1, len(rec)) if iv[j][0] < iv[i][0] and iv[i][1] <= iv[j][1]]
        parent_of[i] = min(cands, key=lambda j: iv[j][1] - iv[j][0]) if cands else None
    order = sorted(range(len(rec)), key=lambda i: (iv[i][0], -(iv[i][1] - iv[i][0])))  # start parents first
    otel = {}
    for i in order:
        s = rec[i]
        parent = otel[parent_of[i]] if parent_of[i] is not None and parent_of[i] in otel else root
        start = t0 + int(s["start_ms"] * 1e6)
        kind = "TOOL" if s["name"].startswith("tool:") else SPAN_KIND.get(s["name"], "CHAIN")
        child = _tracer.start_span(s["name"], context=trace.set_span_in_context(parent), start_time=start,
                                   attributes={"openinference.span.kind": kind, **_attrs(s["attrs"])})
        if "error" in s["attrs"]:
            child.set_status(trace.Status(trace.StatusCode.ERROR, str(s["attrs"]["error"])[:200]))
        otel[i] = child
    for i, child in otel.items():
        child.end(end_time=t0 + int(iv[i][1] * 1e6))
    root.end(end_time=t0 + int(record["duration_ms"] * 1e6))


def flush() -> None:
    if _provider is not None:
        _provider.force_flush()


# ---------------------------------------------------------------- Prometheus metrics
REGISTRY = CollectorRegistry()
REQUESTS = Counter("rag_requests_total", "Requests by outcome", ["route", "outcome"], registry=REGISTRY)
LATENCY = Histogram("rag_request_seconds", "End-to-end latency", ["route", "cached"], registry=REGISTRY,
                    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32))
TTFT = Histogram("rag_time_to_first_token_seconds", "Streaming: time to first answer token", registry=REGISTRY,
                 buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16))
STAGE = Histogram("rag_stage_seconds", "Latency per pipeline stage", ["stage"], registry=REGISTRY,
                  buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8))
COST = Counter("rag_cost_usd_total", "LLM spend in USD", ["stage"], registry=REGISTRY)
TOKENS = Counter("rag_tokens_total", "LLM tokens", ["direction"], registry=REGISTRY)
CACHE = Counter("rag_cache_total", "Cache lookups", ["layer", "result"], registry=REGISTRY)
GUARDRAIL = Counter("rag_guardrail_total", "Guardrail outcomes", ["check", "result"], registry=REGISTRY)
DEGRADED = Counter("rag_degraded_total", "Requests served with a fallback", ["dependency"], registry=REGISTRY)
FEEDBACK = Counter("rag_feedback_total", "User feedback", ["rating"], registry=REGISTRY)


def record_metrics(record: dict) -> None:
    a = record["attrs"]
    route, cached = a.get("route", "pipeline"), a.get("cache") or "miss"
    REQUESTS.labels(route, a.get("outcome", "answered")).inc()
    LATENCY.labels(route, str(cached != "miss")).observe(record["duration_ms"] / 1000)
    for s in record["spans"]:
        STAGE.labels(s["name"]).observe(s.get("duration_ms", 0) / 1000)
        if s["attrs"].get("cost_usd"):
            COST.labels(s["name"]).inc(s["attrs"]["cost_usd"])
        for k, d in (("input_tokens", "input"), ("output_tokens", "output")):
            if s["attrs"].get(k):
                TOKENS.labels(d).inc(s["attrs"][k])
    for dep in a.get("degraded") or []:
        DEGRADED.labels(dep).inc()
