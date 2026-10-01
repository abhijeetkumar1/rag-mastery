"""OpenTelemetry tracing to a local Arize Phoenix server (Phase 6): every LangGraph node, LLM call and tool call
becomes a span, viewable as a tree at http://localhost:6006.

    uv run phoenix serve          # in another terminal (data in data/phoenix/ if PHOENIX_WORKING_DIR=data/phoenix)

Two tracers now run side by side, on purpose:
  common.trace (Phases 1-5)  our JSONL spans: what evaluation reads (cost, tokens) and what 06_traces summarizes
  Phoenix (this module)      OpenInference spans, auto-instrumented: the agent's full step/tool tree, inputs and
                             outputs of every LLM call. Exactly what you want when an agent does something odd.
OpenInference = semantic conventions on top of OpenTelemetry for LLM apps (span kinds LLM, TOOL, CHAIN, AGENT,
RETRIEVER; attributes like llm.token_count.prompt). Any OTel backend can store these spans; Phoenix renders them.
"""
import socket
from urllib.parse import urlparse

from common.config import PHOENIX_ENDPOINT, PHOENIX_TRACING

_provider = None


def _reachable(url: str) -> bool:
    u = urlparse(url)
    try:
        with socket.create_connection((u.hostname, u.port or 80), timeout=0.3):
            return True
    except OSError:
        return False


def setup_phoenix(project: str = "phase6-agentic-rag"):
    """Instrument LangChain/LangGraph once per process. Returns the tracer provider, or None if tracing is off or no
    Phoenix server is listening (then: no-op).

    Only LangChain is instrumented, not the OpenAI SDK too: ChatOpenAI calls the OpenAI SDK internally, so with both
    instrumentors every LLM call appeared TWICE (a "ChatOpenAI" span wrapping a "ChatCompletion" span: 22 + 24 LLM
    spans for 22 calls in the first run) and Phoenix's token/cost totals double-counted. The price: our own chat_json
    calls (Phase 3's planner) don't show up in Phoenix; they are in the JSONL trace."""
    global _provider
    if _provider is not None or not PHOENIX_TRACING or not _reachable(PHOENIX_ENDPOINT):
        return _provider
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register
    _provider = register(project_name=project, endpoint=f"{PHOENIX_ENDPOINT}/v1/traces", batch=True, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=_provider)
    return _provider


def flush() -> None:
    """Batch span processor: push buffered spans before a short script exits."""
    if _provider is not None:
        _provider.force_flush()
