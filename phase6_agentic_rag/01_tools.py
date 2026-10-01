"""The agent's tools, called by hand: see exactly what the LLM will see (the JSON schema it gets, and each result).

Run: uv run python -m phase6_agentic_rag.01_tools

Prints each tool's name, description and argument schema (this is what bind_tools sends to the model), then calls
every tool once. Note how the results are numbered [1], [2]... across calls: one Evidence list per run.
"""
import json

from common.agent_tools import Evidence, make_tools, safe_eval
from common.retriever import Retriever


def main() -> None:
    R = Retriever("structctx", 350)
    R.warmup()
    ev = Evidence()
    tools = make_tools(R, ev)
    for t in tools.values():
        schema = t.args_schema.model_json_schema()
        print(f"== {t.name}: {t.description}\n   args: {json.dumps(schema['properties'])[:300]}...\n")

    calls = [
        ("search_tables", {"query": "purchases of property and equipment", "company": "AMZN", "fiscal_years": [2025]}),
        ("search_filings", {"query": "number of employees", "company": "TSLA"}),
        ("company_relations", {"other": "OpenAI"}),
        ("calculate", {"expression": "(6,411 / 4,540 - 1) * 100"}),
    ]
    for name, args in calls:
        out = tools[name].invoke(args)
        print(f"--- {name}({args})\n{out[:900]}{'...' if len(out) > 900 else ''}\n")
    print(f"evidence collected: {len(ev.hits)} passages, numbered 1..{len(ev.hits)}")

    for bad in ["__import__('os').system('ls')", "open('/etc/passwd').read()", "2**10"]:
        try:
            print(f"safe_eval({bad!r}) = {safe_eval(bad)}")
        except Exception as e:
            print(f"safe_eval({bad!r}) -> rejected: {e}")


if __name__ == "__main__":
    main()
