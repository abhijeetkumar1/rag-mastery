"""Phase 2 RAG: filters -> hybrid (BM25 + dense) -> RRF -> cross-encoder rerank -> relevance floor -> generate.

Run: uv run python -m phase2_better_retrieval.05_ask                      # demo questions
     uv run python -m phase2_better_retrieval.05_ask "your question" --show-context
     uv run python -m phase2_better_retrieval.05_ask --mode dense --no-rerank --filters none "..."   # ablations
     uv run python -m phase1_naive_rag.06_traces --name ask_v2           # traces of this pipeline

Controlled experiment: the prompt and generator are Phase 1's, UNCHANGED (imported from 04_ask).
Only retrieval differs, so any change in answers is caused by retrieval.

Guardrails:
  before the LLM: relevance floor. If the best rerank score is below RELEVANCE_FLOOR, refuse without
                  calling the LLM (cheaper, and no chance to hallucinate from irrelevant context)
  after the LLM:  Phase 1's output checks (citations, numeric grounding)
"""
import argparse
import importlib

from common.config import CHAT_MODEL, EMBED_MODEL, RERANK_MODEL
from common.filters import extract_filters
from common.guardrails import check_answer
from common.llm import chat_completion
from common.retriever import Retriever
from common.trace import Trace, cost_usd

phase1 = importlib.import_module("phase1_naive_rag.04_ask")  # module names starting with a digit need importlib

# Calibrated in 04_rerank for cross-encoder/ms-marco-MiniLM-L-6-v2 (raw logits): keeps 12/12 answerable
# probes, blocks off-topic questions. Meaningless for another reranker: recalibrate if RERANK_MODEL changes.
RELEVANCE_FLOOR = -3.0
FLOOR_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REFUSAL = "I don't know based on the provided filings."

DEMO = phase1.DEMO + [
    "Dojo supercomputer",                                   # not in the corpus: refused by the floor, no LLM call
    "Is Tesla too reliant on its CEO?",                     # paraphrase: rerank lifts the Musk chunk
]


def ask(R: Retriever, question: str, k: int = 5, mode: str = "hybrid", rerank: bool = True,
        filters: str | None = "auto", n_candidates: int = 50) -> tuple[str, list[dict], dict]:
    tr = Trace("ask_v2", question=question, k=k, mode=mode, rerank=rerank, filters=filters,
               embed_model=EMBED_MODEL, rerank_model=RERANK_MODEL if rerank else None, chat_model=CHAT_MODEL)
    try:
        f = extract_filters(question) if filters == "auto" else None
        tickers = (f or {}).get("ticker", [])
        if len(tickers) > 1:  # multi-company question: one retrieval per company so each gets slots
            per = max(2, k // len(tickers) + 1)
            hits = R.retrieve_fanout(question, "ticker", tickers, k_each=per, mode=mode, rerank=rerank,
                                     n_candidates=n_candidates, filters=f, trace=tr)
            tr.set(fanout=tickers)
        else:
            hits = R.retrieve(question, k=k, mode=mode, rerank=rerank, n_candidates=n_candidates,
                              filters=filters, trace=tr)

        report = {"refused": False, "cited": [], "passed": True}
        if rerank and RERANK_MODEL == FLOOR_MODEL:
            with tr.span("relevance_floor", floor=RELEVANCE_FLOOR) as sp:
                best = max((h["rerank_score"] for h in hits), default=float("-inf"))
                sp["best_score"] = round(best, 3)
                sp["blocked"] = best < RELEVANCE_FLOOR
            if sp["blocked"]:
                report = {"refused": True, "refused_by": "relevance_floor", "best_score": best, "cited": [], "passed": True}
                tr.set(answer=REFUSAL, refused=True, refused_by="relevance_floor", guardrails_passed=True)
                return REFUSAL, hits, report

        with tr.span("generate", model=CHAT_MODEL) as sp:
            messages = [
                {"role": "system", "content": phase1.SYSTEM},
                {"role": "user", "content": f"Context:\n\n{phase1.format_context(hits)}\n\nQuestion: {question}"},
            ]
            resp = chat_completion(messages)
            answer = resp.choices[0].message.content
            u = resp.usage
            sp.update(input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens,
                      cost_usd=cost_usd(CHAT_MODEL, u.prompt_tokens, u.completion_tokens))

        with tr.span("guardrails") as sp:
            report = check_answer(answer, [h["text"] for h in hits])
            sp.update(report)
        tr.set(answer=answer, refused=report["refused"], guardrails_passed=report["passed"])
        return answer, hits, report
    finally:
        tr.end()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--filters", choices=["auto", "none"], default="auto")
    ap.add_argument("-n", "--candidates", type=int, default=50, help="candidates passed to the reranker")
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()

    R = Retriever()
    if not args.no_rerank:
        R.warmup()  # model load at startup, not in the first request (see WALKTHROUGH: cold start)
    for q in [args.question] if args.question else DEMO:
        answer, hits, report = ask(R, q, args.k, args.mode, not args.no_rerank,
                                   None if args.filters == "none" else "auto", args.candidates)
        print(f"Q: {q}\n\n{answer}\n")
        for n, h in enumerate(hits, 1):
            m = h["meta"]
            mark = "*" if n in report.get("cited", []) else " "
            stages = f"dense#{h.get('dense_rank', '-')} bm25#{h.get('bm25_rank', '-')} fused#{h['fused_rank']}"
            print(f" {mark}[{n}] {h['score']:7.3f}  {m['ticker']} FY{m['fiscal_year']} {m['item']:8s} ({h['id']})  {stages}")
            if args.show_context:
                print("      " + h["text"][:300].replace("\n", " ") + "...")
        if report.get("refused_by") == "relevance_floor":
            print(f"\nguardrails: refused BEFORE the LLM call (best rerank score {report['best_score']:.2f} < floor {RELEVANCE_FLOOR})")
        elif report["refused"]:
            print("\nguardrails: refusal by the LLM (no answer in context)")
        elif report["passed"]:
            print("\nguardrails: ✅ citations valid, all numbers found in cited passages")
        else:
            print(f"\nguardrails: ⚠ invalid cites {report['invalid_citations']}, uncited={report['uncited_answer']}, "
                  f"ungrounded numbers {report['ungrounded_numbers']}")
        print(f"\n(* = cited, score = rerank score if reranked else RRF score)\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
