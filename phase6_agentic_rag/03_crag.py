"""Corrective RAG (common/crag.py) end to end. Traced as "ask_v6_crag" (JSONL) and in Phoenix.

Run: uv run python -m phase6_agentic_rag.03_crag
     uv run python -m phase6_agentic_rag.03_crag "How much did Tesla spend on restructuring and other in 2024?" --show
     uv run python -m phase1_naive_rag.06_traces --name ask_v6_crag
"""
import argparse
import importlib

from common.config import GRADER_MODEL
from common.crag import build_graph
from common.guardrails import check_answer, is_refusal
from common.observability import flush, setup_phoenix
from common.trace import Trace

ask5 = importlib.import_module("phase5_advanced_indexing.06_ask")
react = importlib.import_module("phase6_agentic_rag.02_react")
_GRAPH = None

DEMO = [
    "How much did Amazon spend on purchases of property and equipment in 2025?",  # Phase 5: net figure, gross table missed
    "How much did Tesla spend on restructuring and other in 2024?",               # Phase 4/5: $583M component vs $684M total
    "Compare the total revenue growth of Microsoft and Amazon in their latest fiscal year.",  # Phase 5: MSFT segment 16%
    "What was Microsoft's net income in fiscal 2026?",
]


def ask(ix, question: str, trace_name: str = "ask_v6_crag", grader_model: str | None = None) -> tuple[str, list[dict], dict]:
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    tr = Trace(trace_name, question=question, index=ix.name, grader_model=grader_model or GRADER_MODEL)
    run = {"trace": tr, "R": ix.R, "cost": 0.0, "route": react.route, "grader_model": grader_model or GRADER_MODEL,
           "retrieve": lambda q, subs, t: ask5.retrieve(ix, q, subs, 6, t),
           "generate": lambda q, hits, t: ask5.generate(q, hits, delimit=True, tr=t)}
    try:
        s = _GRAPH.invoke({"question": question}, config={"configurable": {"run": run}, "recursion_limit": 25,
                                                          "run_name": trace_name, "metadata": {"question": question}})
        answer, passages = s["answer"], s.get("passages", [])
        report = check_answer(answer, [h["text"] for h in passages], tolerant=True) if passages else \
            {"refused": True, "cited": [], "passed": True}
        report.update(refused_by=s.get("refused_by"), rounds=s.get("rounds", 0), fixes=s.get("fixes", 0),
                      verdict=s.get("verdict"), missing_after=[q for q, g in (s.get("grades") or {}).items()
                                                               if not any(p["answers"] for p in g["passages"])],
                      corrective_queries={q: t[1:] for q, t in (s.get("tried") or {}).items() if len(t) > 1})
        tr.set(answer=answer, refused=is_refusal(answer) or bool(s.get("refused_by")), guardrails_passed=report["passed"],
               rounds=report["rounds"], fixes=report["fixes"], verified=(s.get("verdict") or {}).get("ok"))
        return answer, passages, report
    finally:
        tr.end()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--show", action="store_true", help="print grades and the verifier's verdict")
    ap.add_argument("--grader-model", default=GRADER_MODEL, help="model for grade / rewrite / verify")
    args = ap.parse_args()
    setup_phoenix()
    ix = ask5.Index("structctx350")
    ix.warmup()
    for q in [args.question] if args.question else DEMO:
        answer, passages, rep = ask(ix, q, grader_model=args.grader_model)
        print(f"Q: {q}\n\n{answer}\n")
        print(f"   corrective rounds={rep['rounds']} queries={rep['corrective_queries'] or '-'}  fixes={rep['fixes']}  "
              f"still missing={rep['missing_after'] or '-'}")
        v = rep.get("verdict") or {}
        if v and "ok" in v:
            print(f"   verify: ok={v['ok']} metric={v['metric_matches']} period={v['period_matches']} supported={v['supported']}"
                  + (f"  feedback: {v['feedback'][:160]}" if v.get("feedback") else ""))
        if args.show:
            for i, h in enumerate(passages, 1):
                print(f"   [{i}] {h['id']}")
        print("=" * 80 + "\n")
    flush()


if __name__ == "__main__":
    main()
