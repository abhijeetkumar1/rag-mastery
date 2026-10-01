"""Route each question to the cheapest pipeline that can answer it: the Phase 5 pipeline for single-fact questions,
the ReAct agent for questions that need several companies, arithmetic or relations ("when to use an agent").

Run: uv run python -m phase6_agentic_rag.06_hybrid "Rank the five companies by revenue growth."
     uv run python -m phase6_agentic_rag.06_hybrid "What was Apple's total net sales in fiscal 2025?"

The rule uses only the question (and Phase 3's cached plan): ≥ 3 sub-questions, or wording that asks to aggregate,
rank, compute or relate. It was written on dev (12 of 68 dev questions route to ReAct, including all 6 multi-hop
ones; 56 stay on P5), so its dev score is optimistic; test3 measures it.
Why not let an LLM route? It could; this rule is free, instant and inspectable, and the planner already counts companies.
"""
import argparse
import importlib
import re

from common.query import plan

ask5 = importlib.import_module("phase5_advanced_indexing.06_ask")
react = importlib.import_module("phase6_agentic_rag.02_react")

AGGREGATE = re.compile(r"\b(which of the (five|5)|rank|combined|total of|sum of|most|least|highest|lowest|fastest|"
                       r"by what percentage|how much more|grew more|partner|suppl|compet)\w*", re.I)


def route(question: str) -> tuple[str, str]:
    """('react' | 'pipeline', why)."""
    p, _, _ = plan(question)
    n = len(p["sub_questions"])
    if p["route"] != "answer":
        return "pipeline", f"router: {p['route']}"
    if n >= 3:
        return "react", f"{n} sub-questions"
    m = AGGREGATE.search(question)
    return ("react", f"aggregate wording: '{m.group(0)}'") if m else ("pipeline", f"{n} sub-question(s), no aggregation")


def ask(ix, question: str) -> tuple[str, list[dict], dict]:
    target, why = route(question)
    answer, hits, rep = react.ask(ix.R, question) if target == "react" else ask5.ask(ix, question, 6)
    rep["routed_to"], rep["route_reason"] = target, why
    return answer, hits, rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    args = ap.parse_args()
    ix = ask5.Index("structctx350")
    ix.warmup()
    answer, hits, rep = ask(ix, args.question)
    print(f"routed to {rep['routed_to']} ({rep['route_reason']})\n\n{answer}")


if __name__ == "__main__":
    main()
