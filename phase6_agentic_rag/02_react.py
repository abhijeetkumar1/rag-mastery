"""ReAct agent (common/react_agent.py) end to end: router -> agent loop with tools and limits -> output guardrails.
Traced twice: our JSONL trace "ask_v6_react" (cost, steps, stop reason) and Phoenix (the full step/tool tree).

Run: uv run phoenix serve                                         # optional, other terminal: http://localhost:6006
     uv run python -m phase6_agentic_rag.02_react                 # demo questions
     uv run python -m phase6_agentic_rag.02_react "Which of the five companies grew revenue fastest in its latest fiscal year?"
     uv run python -m phase1_naive_rag.06_traces --name ask_v6_react
"""
import argparse
import importlib

from langchain_core.messages import HumanMessage

from common.agent_limits import AgentLimits, Budget
from common.agent_tools import Evidence, make_tools
from common.config import CHAT_MODEL
from common.guardrails import check_answer, is_refusal
from common.observability import flush, setup_phoenix
from common.query import plan
from common.react_agent import build_graph
from common.retriever import Retriever
from common.trace import Trace, cost_usd

ask3 = importlib.import_module("phase3_query_intelligence.06_ask")
_GRAPH = None

DEMO = [
    "How much did Amazon spend on purchases of property and equipment in 2025?",   # gross $131,819M vs net $128.3B
    "Compare the total revenue growth of Microsoft and Amazon in their latest fiscal year.",
    "Which of the five companies grew revenue fastest in its latest fiscal year?",  # 5 companies: multi-hop
    "Which of the five companies describe a partnership with OpenAI?",              # graph tool
]


def route(question: str, tr: Trace) -> str | None:
    """Phase 3's router as the input guardrail in front of the agent: refusal message, or None to proceed."""
    with tr.span("plan", route_only=True) as sp:
        p, usage, _ = plan(question)
        sp.update(route=p["route"], cached=usage["cached"], cost_usd=cost_usd(CHAT_MODEL, usage["input_tokens"], usage["output_tokens"]))
    if p["route"] == "answer":
        return None
    names = p["unsupported_companies"]
    return ask3.REFUSALS[p["route"]].format(names=", ".join(names) or "That company", verb="is" if len(names) < 2 else "are")


def ask(R: Retriever, question: str, limits: AgentLimits | None = None, trace_name: str = "ask_v6_react",
        tools_hook=None, bind: list[str] | None = None) -> tuple[str, list[dict], dict]:
    """tools_hook(tools, evidence) may replace or add tools (04_limits plants a malicious passage, adds a dangerous tool).
    bind: the tools the LLM is SHOWN (default: the allowlist). Execution is still checked against limits.allowed_tools,
    so binding a tool and allowing it are separate decisions (defense in depth)."""
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    limits = limits or AgentLimits()
    tr = Trace(trace_name, question=question, chat_model=CHAT_MODEL, limits={"max_steps": limits.max_steps,
               "max_tool_calls": limits.max_tool_calls, "max_cost_usd": limits.max_cost_usd, "allowed": sorted(limits.allowed_tools)})
    try:
        refusal = route(question, tr)
        if refusal:
            tr.set(answer=refusal, refused=True, refused_by="router", guardrails_passed=True)
            return refusal, [], {"refused": True, "refused_by": "router", "cited": [], "passed": True}
        ev, budget = Evidence(), Budget(limits, CHAT_MODEL)
        tools = make_tools(R, ev, tr)
        if tools_hook:
            tools_hook(tools, ev)
        run = {"tools": tools, "allowed": bind or sorted(limits.allowed_tools & set(tools)), "budget": budget, "trace": tr,
               "evidence": ev}
        state = _GRAPH.invoke({"messages": [HumanMessage(question)], "stop_reason": None, "checked": False},
                              config={"configurable": {"run": run}, "recursion_limit": 4 * limits.max_steps + 4,
                                      "run_name": trace_name, "metadata": {"question": question}})
        answer = state["messages"][-1].content
        with tr.span("guardrails", tolerant=True) as sp:
            report = check_answer(answer, [h["text"] for h in ev.hits], tolerant=True)
            report.update(stop_reason=state["stop_reason"], self_corrected=sum(isinstance(m, HumanMessage) for m in state["messages"]) > 1,
                          **budget.summary(),
                          tool_sequence=[[c["name"], c["args"]] for m in state["messages"] if getattr(m, "tool_calls", None) for c in m.tool_calls])
            sp.update({k: v for k, v in report.items() if k != "tool_sequence"})
        tr.set(answer=answer, refused=is_refusal(answer), guardrails_passed=report["passed"], stop_reason=state["stop_reason"],
               steps=budget.steps, tool_calls=budget.tool_calls, agent_cost_usd=budget.cost)
        return answer, ev.hits, report
    finally:
        tr.end()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--max-steps", type=int, default=AgentLimits.max_steps)
    args = ap.parse_args()
    setup_phoenix()
    R = Retriever("structctx", 350)
    R.warmup()
    for q in [args.question] if args.question else DEMO:
        answer, hits, rep = ask(R, q, AgentLimits(max_steps=args.max_steps))
        print(f"Q: {q}\n\n{answer}\n")
        for name, a in rep.get("tool_sequence", []):
            print(f"   → {name}({', '.join(f'{k}={v!r}' for k, v in a.items())})")
        if "steps" in rep:
            print(f"\n   steps={rep['steps']} tool_calls={rep['tool_calls']} tokens={rep['tokens']} cost=${rep['cost_usd']:.4f} "
                  f"stop={rep['stop_reason'] or 'answered'} blocked={rep['blocked'] or '-'} evidence={len(hits)} cited={rep['cited']}")
            print(f"   guardrails: {'✅' if rep['passed'] else '⚠ ungrounded ' + str(rep['ungrounded_numbers'])}")
        print("=" * 80 + "\n")
    flush()


if __name__ == "__main__":
    main()
