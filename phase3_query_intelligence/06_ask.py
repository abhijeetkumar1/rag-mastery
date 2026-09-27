"""Phase 3 RAG: plan (route + self-query + decomposition) -> per-sub-question hybrid retrieval -> floor
-> Phase 1 generator (unchanged) -> tolerant output guardrails. Traced as "ask_v3".

Run: uv run python -m phase3_query_intelligence.06_ask                     # demo questions
     uv run python -m phase3_query_intelligence.06_ask "your question" --show-context
     uv run python -m phase3_query_intelligence.06_ask --multi-query "..."   # + RAG-Fusion per sub-question
     uv run python -m phase3_query_intelligence.06_ask --hyde "..."          # + HyDE on the dense side
     uv run python -m phase1_naive_rag.06_traces --name ask_v3              # traces

Guardrails, in order: router (input) -> relevance floor (before generation) -> output checks (after).
"""
import argparse
import importlib

from common.config import CHAT_MODEL, EMBED_MODEL, RERANK_MODEL
from common.guardrails import check_answer
from common.llm import chat_completion
from common.query import COMPANIES, PROMPT_VERSION, hyde, multi_query, plan
from common.retriever import Retriever
from common.trace import Trace, cost_usd

phase1 = importlib.import_module("phase1_naive_rag.04_ask")
phase2 = importlib.import_module("phase2_better_retrieval.05_ask")

REFUSALS = {
    "unsupported_company": "I can only answer from the 10-K filings of " + ", ".join(COMPANIES.values())
                           + ". {names} {verb} not covered.",
    "investment_advice": "I can tell you what the 10-K filings say, but I can't give investment advice or recommendations.",
    "out_of_scope": "That question isn't about the 10-K filings I cover (" + ", ".join(COMPANIES.values()) + ").",
}

DEMO = [
    "Compare the total revenue growth of Microsoft and Amazon in their latest fiscal year.",  # Phase 2: both halves wrong
    "What was Apple's total net sales in fiscal 2023?",       # Phase 2 filter pitfall: no FY2023 filing
    "How did the EV maker's deliveries change last year?",    # indirect company + relative time
    "What was Google's advertising revenue in 2025?",         # router: unsupported company
    "Should I buy NVIDIA stock?",                             # router: investment advice
    "How do I reset my iPhone?",                              # router lets it through (policy): refused downstream?
]


def ask(R: Retriever, question: str, k: int = 6, use_plan: bool = True, use_multi: bool = False,
        use_hyde: bool = False, tolerant: bool = True, keep_original: bool = True) -> tuple[str, list[dict], dict]:
    """keep_original: retrieve each sub-question with its rewrite AND the user's own words (RRF-fused). The planner
    once rewrote "purchases of property and equipment" (the 10-K's exact phrase) into "capital expenditures",
    and the answer was lost: a rewrite may add words, it must never be the only query."""
    tr = Trace("ask_v3", question=question, k=k, prompt_version=PROMPT_VERSION if use_plan else None,
               multi_query=use_multi, hyde=use_hyde, embed_model=EMBED_MODEL, rerank_model=RERANK_MODEL,
               chat_model=CHAT_MODEL)
    try:
        if use_plan:
            with tr.span("plan", version=PROMPT_VERSION) as sp:
                p, usage, overrides = plan(question)
                sp.update(route=p["route"], companies=p["companies"], overrides=overrides, cached=usage["cached"],
                          input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
                          cost_usd=cost_usd(CHAT_MODEL, usage["input_tokens"], usage["output_tokens"]),
                          sub_questions=[[s["ticker"], s["filing_years"], s["question"]] for s in p["sub_questions"]])
            if p["route"] != "answer":
                names = p["unsupported_companies"]
                msg = REFUSALS[p["route"]].format(names=", ".join(names) or "That company",
                                                  verb="is" if len(names) < 2 else "are")
                tr.set(answer=msg, refused=True, refused_by=f"router:{p['route']}", guardrails_passed=True)
                return msg, [], {"refused": True, "refused_by": f"router:{p['route']}", "cited": [], "passed": True}
            subs = p["sub_questions"]
        else:
            subs = [{"question": question, "ticker": None, "filing_years": []}]

        k_each = k if len(subs) == 1 else max(2, k // len(subs))
        hits = []
        for s in subs:
            f = {**({"ticker": [s["ticker"]]} if s["ticker"] else {}),
                 **({"fiscal_year": s["filing_years"]} if s["filing_years"] else {})} or None
            if use_multi:
                with tr.span("multi_query") as sp:
                    qs, u = multi_query(s["question"])
                    sp.update(queries=qs, cached=u["cached"], cost_usd=cost_usd(CHAT_MODEL, u["input_tokens"], u["output_tokens"]))
                sub_hits = R.retrieve_multi([s["question"], *qs], s["question"], k=k_each, filters=f, trace=tr)
            elif keep_original and not use_hyde and s["question"] != question:
                sub_hits = R.retrieve_multi([s["question"], question], s["question"], k=k_each, filters=f, trace=tr)
            else:
                passage = None
                if use_hyde:
                    with tr.span("hyde") as sp:
                        passage, u = hyde(s["question"])
                        sp.update(passage=passage[:300], cached=u["cached"],
                                  cost_usd=cost_usd(CHAT_MODEL, u["input_tokens"], u["output_tokens"]))
                sub_hits = R.retrieve(s["question"], k=k_each, filters=f, trace=tr, hyde_passage=passage)
            for h in sub_hits:
                h["sub_question"] = s["question"]
            hits += sub_hits

        with tr.span("relevance_floor", floor=phase2.RELEVANCE_FLOOR) as sp:
            best = max((h.get("rerank_score", float("-inf")) for h in hits), default=float("-inf"))
            sp.update(best_score=round(best, 3), blocked=best < phase2.RELEVANCE_FLOOR)
        if sp["blocked"] and RERANK_MODEL == phase2.FLOOR_MODEL:
            tr.set(answer=phase2.REFUSAL, refused=True, refused_by="relevance_floor", guardrails_passed=True)
            return phase2.REFUSAL, hits, {"refused": True, "refused_by": "relevance_floor", "best_score": best,
                                          "cited": [], "passed": True}

        with tr.span("generate", model=CHAT_MODEL) as sp:
            messages = [{"role": "system", "content": phase1.SYSTEM},
                        {"role": "user", "content": f"Context:\n\n{phase1.format_context(hits)}\n\nQuestion: {question}"}]
            resp = chat_completion(messages)
            answer = resp.choices[0].message.content
            u = resp.usage
            sp.update(input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens,
                      cost_usd=cost_usd(CHAT_MODEL, u.prompt_tokens, u.completion_tokens))

        with tr.span("guardrails", tolerant=tolerant) as sp:
            report = check_answer(answer, [h["text"] for h in hits], tolerant=tolerant)
            sp.update(report)
        tr.set(answer=answer, refused=report["refused"], guardrails_passed=report["passed"])
        return answer, hits, report
    finally:
        tr.end()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--no-plan", action="store_true", help="skip the analyzer (Phase 2 behaviour, no filters)")
    ap.add_argument("--multi-query", action="store_true")
    ap.add_argument("--hyde", action="store_true")
    ap.add_argument("--strict", action="store_true", help="strict numeric guardrail (no derived numbers)")
    ap.add_argument("--no-original", action="store_true", help="retrieve with the planner's rewrite only (ablation)")
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()

    R = Retriever()
    R.warmup()
    for q in [args.question] if args.question else DEMO:
        answer, hits, report = ask(R, q, args.k, not args.no_plan, args.multi_query, args.hyde, not args.strict,
                                   not args.no_original)
        print(f"Q: {q}\n\n{answer}\n")
        for n, h in enumerate(hits, 1):
            m = h["meta"]
            mark = "*" if n in report.get("cited", []) else " "
            print(f" {mark}[{n}] {h['score']:6.2f}  {m['ticker']} FY{m['fiscal_year']} {m['item']:8s} ({h['id']})"
                  f"  <- {h.get('sub_question', '')[:60]}")
            if args.show_context:
                print("      " + h["text"][:300].replace("\n", " ") + "...")
        by = report.get("refused_by")
        if by:
            print(f"\nguardrails: refused by {by}" + (f" (best rerank {report['best_score']:.2f})" if "best_score" in report else ""))
        elif report["refused"]:
            print("\nguardrails: refusal by the LLM (no answer in context)")
        elif report["passed"]:
            derived = report.get("derived_numbers", {})
            print("\nguardrails: ✅ citations valid, numbers grounded" + (f" ({len(derived)} derived: {derived})" if derived else ""))
        else:
            print(f"\nguardrails: ⚠ invalid cites {report['invalid_citations']}, uncited={report['uncited_answer']}, "
                  f"ungrounded numbers {report['ungrounded_numbers']}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
