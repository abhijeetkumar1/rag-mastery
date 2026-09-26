"""Output guardrails on hand-written answers: see what they catch and what they miss. No API calls.

Run: uv run python -m phase1_naive_rag.05_guardrails_demo

A guardrail is a classifier ("is this answer safe to show?"), so it has precision and recall like
any other: it can block good answers (false positives) and let bad ones through (false negatives).
Test it with known cases, like any other code.
"""
import json

from common.config import DATA_DIR
from common.guardrails import check_answer

# Real passages from the index: Apple's FY2025 net-sales table + one unrelated passage
rows = {json.loads(line)["id"]: json.loads(line) for line in (DATA_DIR / "processed" / "chunks_recursive350.jsonl").open()}
passages = [rows["AAPL_FY2025_Item8_049"]["text"], rows["AAPL_FY2025_Item1A_000"]["text"]]
# passage [1] contains: "Total net sales | 416,161 | 391,035 | 383,285" and "China (1) | 64,377 | 66,952 | 72,559"

CASES = [
    # (label, expected verdict, answer)
    ("grounded", "pass",
     "Apple's total net sales in fiscal 2025 were $416,161 million [1]."),
    ("hallucinated number", "flag",
     "Apple's total net sales in fiscal 2025 were $420,500 million [1]."),
    ("citation to a passage that wasn't in the prompt", "flag",
     "Apple's total net sales in fiscal 2025 were $416,161 million [7]."),
    ("no citations at all", "flag",
     "Apple's total net sales in fiscal 2025 were $416,161 million."),
    ("right number, WRONG passage cited", "flag",
     "Apple's total net sales in fiscal 2025 were $416,161 million [2]."),
    ("refusal", "pass",
     "I don't know based on the provided filings."),
    # --- known limitations ---
    ("FALSE POSITIVE: correctly derived growth %", "pass",
     "Net sales grew 6.4% from $391,035 million to $416,161 million [1]."),     # 416161/391035-1 = 6.43%
    ("FALSE POSITIVE: unit conversion", "pass",
     "Apple's total net sales in fiscal 2025 were about $416.2 billion [1]."),
    ("FALSE NEGATIVE: real number, wrong meaning", "flag",
     "Apple's net sales in China in fiscal 2025 were $416,161 million [1]."),    # China was 64,377
    ("FALSE NEGATIVE: wrong year for a real number", "flag",
     "Apple's total net sales in fiscal 2025 were $391,035 million [1]."),       # that's FY2024
]


def main() -> None:
    correct = 0
    for label, expected, answer in CASES:
        r = check_answer(answer, passages)
        verdict = "pass" if r["passed"] else "flag"
        ok = verdict == expected
        correct += ok
        why = []
        if r["invalid_citations"]:
            why.append(f"invalid citations {r['invalid_citations']}")
        if r["uncited_answer"]:
            why.append("no citations")
        if r["ungrounded_numbers"]:
            why.append(f"ungrounded numbers {r['ungrounded_numbers']}")
        if r["refused"]:
            why.append("refusal")
        print(f"{'✅' if ok else '❌'} {label:48s} expected={expected:4s} got={verdict:4s} {'; '.join(why)}")
        print(f"   \"{answer}\"")

    print(f"\n{correct}/{len(CASES)} verdicts correct.")
    print("The misses are the lesson: string-matching numbers can't verify arithmetic, unit conversion,")
    print("or whether a number is attached to the right label/year. That needs a semantic check, e.g.")
    print("an LLM-as-judge or NLI faithfulness check (Phase 4), at the cost of latency and money.")


if __name__ == "__main__":
    main()
