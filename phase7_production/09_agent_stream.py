"""The agent route of the service, checked with a SCRIPTED fake LLM (no API): streamed agent steps, and access
control inside the agent's tools.

Run: uv run python -m phase7_production.09_agent_stream

The fake replaces ChatOpenAI inside common.react_agent and plays a fixed script: search_tables for Apple's net sales,
then company_relations (all partners), then a final answer citing [1]. The graph, the tools, the budget, the
retriever and the service's checks are the real ones. Three runs:
  1. full-access user                                  -> steps stream as events; relations from all five filings
  2. AAPL-only user                                    -> the graph tool only returns edges from Apple's filings
  3. AAPL-only user, the tool's filter removed (a bug) -> evidence from other filings enters the agent's context;
                                                          the service's post-check WITHHOLDS the answer
With the API back, `02_client --mode agent "..."` runs the same route with the real model.
"""
import importlib

from langchain_core.messages import AIMessage, ToolMessage

import common.agent_tools as agent_tools
import common.react_agent as react_agent

service = importlib.import_module("phase7_production.service")


class FakeLLM:
    """bind_tools() + invoke() of a chat model, following a script keyed on how many tool results it has seen."""

    def __init__(self, *a, **kw):
        pass

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        seen = sum(isinstance(m, ToolMessage) for m in messages)
        usage = {"input_tokens": 1500 + 400 * seen, "output_tokens": 40, "total_tokens": 1540 + 400 * seen}
        if seen == 0:
            return AIMessage("", tool_calls=[{"name": "search_tables", "id": "c1", "args": {
                "query": "total net sales", "company": "AAPL", "fiscal_years": [2025]}}], usage_metadata=usage)
        if seen == 1:
            return AIMessage("", tool_calls=[{"name": "company_relations", "id": "c2", "args": {"relation": "PARTNERS_WITH"}}],
                             usage_metadata=usage)
        return AIMessage("Apple's total net sales are shown in its income statement [1].", usage_metadata=usage)


def run(svc, user, title: str) -> None:
    print(f"--- {title} (user {user.id}, tickers {','.join(user.tickers)})")
    for ev in svc.stream("Which partners does Apple describe, and what were its total net sales in fiscal 2025?", user, mode="agent"):
        d = ev["data"]
        if ev["event"] == "step":
            calls = ", ".join(f"{c['name']}({c['args']})" for c in d.get("tool_calls", []))
            print(f"  step {d['step']} {d['node']:6s} {calls or ''}{'' if calls else d}")
        elif ev["event"] == "sources":
            tick = sorted({s["ticker"] for s in d["sources"]})
            print(f"  sources: {len(d['sources'])} passages from {tick}")
        elif ev["event"] == "token":
            print(f"  answer: {d['text'][:100]}")
        elif ev["event"] in ("error", "meta"):
            print(f"  {ev['event']}: {d}")
        elif ev["event"] == "done":
            print(f"  done: route={d['route']} {d['latency_ms']:.0f} ms (cost is the fake's token count priced: ${d['cost_usd']:.4f})")


def main() -> None:
    react_agent.ChatOpenAI = FakeLLM
    svc = service.RAGService(service.Settings(llm_backend="openai", moderation=False))
    svc.warmup()
    # the planner's real LLM call isn't wanted here: use the rule plan directly
    service.plan = service.rule_plan
    service.BREAKERS["generate"].success()
    run(svc, svc.user("demo-key-all"), "1. full access")
    aapl = svc.user("demo-key-aapl")
    run(svc, aapl, "2. AAPL-only")
    real = agent_tools.make_tools
    agent_tools.make_tools = lambda R, ev, tr=None, allowed_tickers=None: real(R, ev, tr)  # the bug: filter dropped
    svc.exact = service.ExactCache(svc.s.version)  # run 2 cached this answer for this scope: start empty
    try:
        run(svc, aapl, "3. AAPL-only, graph-tool filter removed")
    finally:
        agent_tools.make_tools = real


if __name__ == "__main__":
    main()
