"""Naive RAG end to end: retrieve top-k chunks -> stuff into prompt -> answer with [n] citations.

Run: uv run python -m phase1_naive_rag.04_ask                                   # demo questions
     uv run python -m phase1_naive_rag.04_ask "What was Amazon's AWS operating income in 2025?"
     uv run python -m phase1_naive_rag.04_ask -k 10 --show-context "..."

"Naive" = no query rewriting, no hybrid search, no reranking, no filters. The demo questions are
picked so you can SEE where it breaks; later phases fix each failure.
"""
import argparse
import re

from common.config import CHAT_MODEL
from common.llm import chat
from common.store import collection_name, get_collection, search

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


def ask(col, question: str, k: int = 5) -> tuple[str, list[dict]]:
    hits = search(col, question, k=k)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n\n{format_context(hits)}\n\nQuestion: {question}"},
    ]
    return chat(messages), hits


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
        answer, hits = ask(col, q, args.k)
        print(f"Q: {q}\n\n{answer}\n")
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        for n, h in enumerate(hits, 1):
            m = h["meta"]
            mark = "*" if n in cited else " "
            print(f" {mark}[{n}] {h['score']:.3f}  {m['ticker']} FY{m['fiscal_year']} {m['item']}  ({h['id']})")
            if args.show_context:
                print("      " + h["text"][:300].replace("\n", " ") + "...")
        print(f"\n(* = cited, model={CHAT_MODEL})\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
