"""Naive RAG end to end: retrieve top-k chunks -> stuff into prompt -> answer with [n] citations.

Run: uv run python -m phase1_naive_rag.04_ask                                   # demo questions
     uv run python -m phase1_naive_rag.04_ask "What was Amazon's AWS operating income in 2025?"
     uv run python -m phase1_naive_rag.04_ask -k 10 --show-context "..."

"Naive" = no query rewriting, no hybrid search, no reranking, no filters. The demo questions are
picked so you can SEE where it breaks; later phases fix each failure.

Every question is traced (data/traces/, view with 06_traces) and its answer passes through
output guardrails (common/guardrails.py, demo in 05_guardrails_demo).
"""
import argparse

from common.config import CHAT_MODEL, EMBED_MODEL
from common.guardrails import check_answer
from common.llm import chat_completion, embed_query
from common.store import collection_name, get_collection, search_by_vector
from common.trace import Trace, cost_usd

SYSTEM = """You answer questions about SEC 10-K filings using ONLY the numbered context passages.
Rules:
- Cite every factual claim with the passage number in brackets, e.g. [2] or [1][3].
- If the passages do not contain the answer, say "I don't know based on the provided filings." Do not guess.
- Numbers in tables are usually in millions of USD unless the passage says otherwise; state units.
- Be concise."""

DEMO = [
    # should work: single company, single fact, distinctive wording
    "What was Apple's total net sales in fiscal 2025?",
    # should work: risk-factor prose
    "How does NVIDIA describe the risk from U.S. export controls on China?",
    # cross-document comparison: top-k may be dominated by one company/year
    "Compare the total revenue growth of Microsoft and Amazon in their latest fiscal year.",
    # not in the corpus: the model must refuse, not hallucinate
    "What was Google's advertising revenue in 2025?",
]


def format_context(hits: list[dict]) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        m = h["meta"]
        blocks.append(f"[{n}] {m['ticker']} 10-K FY{m['fiscal_year']}, {m['item']}: {m['section']}\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def ask(col, question: str, k: int = 5) -> tuple[str, list[dict], dict]:
    tr = Trace("ask", question=question, k=k, collection=col.name, embed_model=EMBED_MODEL, chat_model=CHAT_MODEL)
    try:
        with tr.span("embed_query"):
            q = embed_query(question)

        with tr.span("vector_search", k=k) as sp:
            hits = search_by_vector(col, q, k=k)
            sp["hits"] = [{"id": h["id"], "score": round(h["score"], 4)} for h in hits]

        with tr.span("generate", model=CHAT_MODEL) as sp:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context:\n\n{format_context(hits)}\n\nQuestion: {question}"},
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
        tr.end()  # written even if a stage raised: failed requests are the ones you most need to see


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--chunker", default="recursive")
    ap.add_argument("--size", type=int, default=350)
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()

    col = get_collection(collection_name(args.chunker, args.size))
    if col.count() == 0:
        raise SystemExit("Empty index: run phase1_naive_rag.03_index first")

    for q in [args.question] if args.question else DEMO:
        answer, hits, report = ask(col, q, args.k)
        print(f"Q: {q}\n\n{answer}\n")
        for n, h in enumerate(hits, 1):
            m = h["meta"]
            mark = "*" if n in report["cited"] else " "
            print(f" {mark}[{n}] {h['score']:.3f}  {m['ticker']} FY{m['fiscal_year']} {m['item']}  ({h['id']})")
            if args.show_context:
                print("      " + h["text"][:300].replace("\n", " ") + "...")

        # Policy = WARN: surface problems to the user instead of blocking (see guardrails.py docstring)
        if report["refused"]:
            print("\nguardrails: refusal (no answer in context)")
        elif report["passed"]:
            print("\nguardrails: ✅ citations valid, all numbers found in cited passages")
        else:
            if report["invalid_citations"]:
                print(f"\nguardrails: ⚠ citations to passages that don't exist: {report['invalid_citations']}")
            if report["uncited_answer"]:
                print("\nguardrails: ⚠ answer makes claims without any citation")
            if report["ungrounded_numbers"]:
                print(f"\nguardrails: ⚠ numbers not found in cited passages (hallucinated or derived?): "
                      f"{report['ungrounded_numbers']}")
        print(f"\n(* = cited, model={CHAT_MODEL})\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
