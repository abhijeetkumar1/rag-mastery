"""Metadata filters: restrict retrieval by ticker / fiscal year, and where filters are NOT enough.

Run: uv run python -m phase2_better_retrieval.03_filters

1. Rule-based filter extraction from the question
2. Apple FY2025 net sales: without vs with the filter
3. The comparison question: no filter vs ticker ∈ {MSFT, AMZN} vs per-ticker fan-out
4. A filter pitfall: "fiscal 2023" filters by FILING year, but FY2023 numbers live in the FY2024 10-K
"""
from common.filters import extract_filters, to_chroma_where
from common.retriever import Retriever


def show(label: str, hits: list[dict]) -> None:
    print(f"  {label:34s} " + (", ".join(f"{h['id']}" for h in hits) or "(nothing: every chunk filtered out)"))


def main() -> None:
    R = Retriever()

    print("1. Filter extraction (rule-based: company aliases + 'fiscal/FY <year>' regex)")
    for q in ["What was Apple's total net sales in fiscal 2025?",
              "Compare the total revenue growth of Microsoft and Amazon in their latest fiscal year.",
              "What was AWS operating income in 2025?",
              "Which companies mention export controls?"]:
        f = extract_filters(q)
        print(f"  {q[:70]:70s} -> {f}   chroma where={to_chroma_where(f)}")
    print("  note: 'in 2025' (no 'fiscal') is NOT turned into a year filter: calendar vs fiscal vs filing year is ambiguous\n")

    q = "What was Apple's total net sales in fiscal 2025?"
    print(f"2. {q}")
    show("no filter", R.retrieve(q, k=5))
    show("filters=auto {AAPL, 2025}", R.retrieve(q, k=5, filters="auto"))
    print()

    q = "Compare the total revenue growth of Microsoft and Amazon in their latest fiscal year."
    print(f"3. {q}")
    show("no filter", R.retrieve(q, k=6))
    show("ticker ∈ {MSFT, AMZN}", R.retrieve(q, k=6, filters="auto"))
    show("fan-out: 3 per ticker, interleaved", R.retrieve_fanout(q, "ticker", ["MSFT", "AMZN"], k_each=3))
    print("  -> the $in filter only restricts WHAT may come back; MSFT chunks still outscore AMZN ones.")
    print("     Fan-out guarantees each company gets slots (Phase 3 generalizes this as query decomposition).\n")

    q = "What was Apple's total net sales in fiscal 2023?"
    print(f"4. {q}")
    show("filters=auto {AAPL, 2023}", R.retrieve(q, k=3, filters="auto"))
    show("filters={AAPL} only", R.retrieve(q, k=3, filters={"ticker": ["AAPL"]}))
    print("  -> fiscal_year metadata = the FILING's year. A 10-K reports 3 years, so FY2023 figures live in")
    print("     the FY2024 and FY2025 filings, and the 'correct' filter removes all of them.")
    print("     Fix at indexing time: tag chunks with the years they COVER (Phase 5, table handling).")


if __name__ == "__main__":
    main()
