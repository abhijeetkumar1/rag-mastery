"""Multi-hop questions: the kind an agent is FOR. Each needs several companies (often all five), arithmetic, or the
relation graph, so a single top-k retrieval can't hold all the evidence. Hand-written, ground truth verified by text
search in the chunks on 2026-10-01 (run this file: it re-checks every figure against the corpus).

Run: uv run python -m phase6_agentic_rag.multihop

Two disjoint sets: "dev" (to build and tune the agents) and "test3" (reported once, with the fresh test3 split).
The golden sets of Phases 4-5 have at most two companies per question, which flatters single-pass RAG; these are
scored separately so the two question types don't average each other away.
"""
import json
import re

from common.config import DATA_DIR

# (question, reference answer, numbers the answer must state, {chunk id prefix: regex that must match a chunk})
DEV = [
    ("Which of the five companies grew revenue fastest in its latest fiscal year?",
     "NVIDIA: revenue grew about 65% in fiscal 2026 ($130,497M to $215,938M). Others: Microsoft ~18%, Amazon ~12%, "
     "Apple ~6%, Tesla declined ~3%.",
     ["65"], {"NVDA_FY2026": r"215,938 \| 130,497", "MSFT_FY2026": r"Total revenue \| 331,839 \| 281,724",
              "AMZN_FY2025": r"Total net sales \| 574,785 \| 637,959 \| 716,924", "AAPL_FY2025": r"Total net sales \| 416,161 \| 391,035",
              "TSLA_FY2025": r"Total revenues \| 94,827 \| 97,690"}),
    ("Which of the five companies had the most employees at the end of its latest fiscal year?",
     "Amazon, with approximately 1,576,000 employees (December 31, 2025); next Microsoft (~223,000), Apple (~166,000), "
     "Tesla (134,785), NVIDIA (~42,000).",
     ["1,576,000"], {"AMZN_FY2025": r"employed approximately 1,576,000", "MSFT_FY2026": r"approximately 223,000",
                     "NVDA_FY2026": r"approximately 42,000 employees"}),
    ("Which of the five companies describe a partnership with OpenAI in their 10-Ks?",
     "Microsoft (a long-term strategic partnership with OpenAI) and NVIDIA (finalizing an investment and partnership "
     "agreement with OpenAI). Apple, Tesla and Amazon do not mention OpenAI.",
     [], {"MSFT_FY2026": r"partnership with OpenAI", "NVDA_FY2026": r"partnership agreement with OpenAI"}),
    ("By what percentage did Tesla's research and development expense grow from 2024 to 2025?",
     "About 41%: from $4,540 million in 2024 to $6,411 million in 2025.",
     ["41"], {"TSLA_FY2025": r"Research and development \| 6,411 \| 4,540"}),
    ("What were the combined total net sales of Apple and Amazon in their latest fiscal years?",
     "$1,133,085 million (about $1.13 trillion): Apple $416,161 million (fiscal 2025) + Amazon $716,924 million (2025).",
     ["1,133,085"], {"AAPL_FY2025": r"Total net sales \| 416,161", "AMZN_FY2025": r"716,924"}),
    ("Rank Apple, Microsoft and Amazon by operating income in their latest fiscal years.",
     "Microsoft $155,237 million (fiscal 2026) > Apple $133,050 million (fiscal 2025) > Amazon $79,975 million (2025).",
     ["155,237", "133,050", "79,975"], {"MSFT_FY2026": r"Operating income \| 155,237", "AAPL_FY2025": r"Operating income \| 133,050",
                                        "AMZN_FY2025": r"Operating income \| 36,852 \| 68,593 \| 79,975"}),
]

TEST3 = [
    ("Rank the five companies by total assets at their latest fiscal year-end.",
     "Amazon $818,042M > Microsoft $758,376M > Apple $359,241M > NVIDIA $206,803M > Tesla $137,806M.",
     ["818,042", "758,376", "359,241", "206,803", "137,806"],
     {"AMZN_FY2025": r"Total assets \| 624,894 \| 818,042", "MSFT_FY2026": r"Total assets \| 758,376",
      "AAPL_FY2025": r"Total assets \| 359,241", "NVDA_FY2026": r"Total assets \| 206,803", "TSLA_FY2025": r"Total assets \| 137,806"}),
    ("Which of the five companies had the lowest net income in its latest fiscal year?",
     "Tesla: net income attributable to common stockholders of $3,794 million in 2025 (vs Amazon $77,670M, Apple "
     "$112,010M, NVIDIA $120,067M, Microsoft $133,749M).",
     ["3,794"], {"TSLA_FY2025": r"Net income attributable to common stockholders \| 3,794", "AAPL_FY2025": r"Net income \| 112,010"}),
    ("By what percentage did Microsoft's net income grow in fiscal 2026?",
     "About 31%: from $101,832 million in fiscal 2025 to $133,749 million in fiscal 2026.",
     ["31"], {"MSFT_FY2026": r"Net income \| 133,749 \| 101,832"}),
    ("Which of the five companies name TSMC as a supplier in their 10-Ks?",
     "Only NVIDIA (TSMC and Samsung supply its semiconductor wafers). The other four filings do not mention TSMC.",
     [], {"NVDA_FY2026": r"Taiwan Semiconductor"}),
    ("How much more did Microsoft spend on research and development than NVIDIA in their latest fiscal years?",
     "$17,065 million more: Microsoft $35,562 million (fiscal 2026) vs NVIDIA $18,497 million (fiscal 2026).",
     ["17,065"], {"MSFT_FY2026": r"Research and development \| 35,562", "NVDA_FY2026": r"Research and development \| 18,497"}),
    ("Whose total assets grew more in percentage terms in the latest fiscal year, Microsoft's or Apple's?",
     "Microsoft's: up about 22.5% (from $619,003M to $758,376M at June 30, 2026); Apple's fell about 1.6% (from "
     "$364,980M to $359,241M at September 27, 2025).",
     ["22.5"], {"MSFT_FY2026": r"Total assets \| 758,376 \| 619,003", "AAPL_FY2025": r"Total assets \| 359,241 \| 364,980"}),
]


def items(split: str) -> list[dict]:
    data = {"dev": DEV, "test3": TEST3}[split]
    return [{"id": f"{split}_mh_{i:03d}", "split": split, "source": "manual", "type": "multihop", "stratum": "multihop",
             "question": q, "answer": a, "evidence": "", "numbers": nums, "expected": "answer",
             "tickers": sorted({p.split("_")[0] for p in ev}), "relevant": {}} for i, (q, a, nums, ev) in enumerate(data)]


def main() -> None:
    rows = [json.loads(line) for line in (DATA_DIR / "processed" / "chunks_structctx350.jsonl").read_text().splitlines()]
    for split, data in [("dev", DEV), ("test3", TEST3)]:
        for q, _, _, ev in data:
            missing = [p for p, pat in ev.items() if not any(r["id"].startswith(p) and re.search(pat, r["body"]) for r in rows)]
            print(f"{'✅' if not missing else '❌ ' + str(missing)} {split:5s} {q}")


if __name__ == "__main__":
    main()
