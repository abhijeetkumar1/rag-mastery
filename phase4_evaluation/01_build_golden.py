"""Build the golden sets: golden_dev.jsonl and golden_test.jsonl (committed: eval sets belong in git).

Run: uv run python -m phase4_evaluation.01_build_golden --split dev     # ~160 gpt-4.1 calls first time, cached after
     uv run python -m phase4_evaluation.01_build_golden --split test

dev   used to DIAGNOSE and FIX the system (Phase 4 did: filing years, a router gap). Scores on it are optimistic.
test  built afterwards from chunks dev never used, reviewed for question quality ONLY, never used to tune
      anything. Reported once, at the end. Once you tune on it, it becomes dev and you need a new test set.

Three sources, none seen by any prompt, vocabulary rule or probe written in Phases 1-3:
  synthetic   stratified random chunks (5 companies × 4 section groups × 4) → gpt-4.1 writes ONE question whose
              answer is stated in the chunk + a VERBATIM evidence quote → validated in code → gpt-4.1 critic filter
  comparison  6 hand-written two-company questions, ground truth verified by text search in the chunks
  unanswerable 8 questions the corpus can't answer (other companies, years not covered, not in a 10-K, advice)

Relevance labels (for retrieval metrics), same company only:
  grade 2  chunks containing the evidence quote in the source chunk's filing
  grade 1  the same quote in the other year's filing (10-Ks repeat text), OR, for numeric answers, any chunk that
           contains ALL of the answer's distinctive figures (≥ 4 significant digits, e.g. 167,045). The FY2025 10-K
           reports FY2024 figures as a comparative column: quote-only labels called those chunks "irrelevant" and
           made a correct retrieval look like a miss (found while diagnosing Phase 3 in 02_retrieval_eval).
"""
import json
import random
import re
from pathlib import Path

from common.bm25 import tokenize
from common.config import JUDGE_MODEL
from common.guardrails import extract_numbers
from common.llm import chat_json
from phase2_better_retrieval.probes import load_rows

SPLITS = {"dev": {"seed": 42, "per_stratum": 4}, "test": {"seed": 7, "per_stratum": 4}}  # 5 × 4 × 4 = 80 candidates
STRATA = {"business": ["Item 1"], "risk": ["Item 1A"], "mdna": ["Item 7"], "financials": ["Item 8", "Item 15"]}

GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "evidence": {"type": "string"},
        "answer_type": {"type": "string", "enum": ["numeric", "text"]},
    },
    "required": ["usable", "question", "answer", "evidence", "answer_type"], "additionalProperties": False,
}
GEN_SYSTEM = """You write evaluation questions for a question-answering system over SEC 10-K filings.
Given ONE passage, write ONE question an investor or analyst might ask whose answer is stated explicitly in it.
Rules:
- Standalone: name the company, and the fiscal year when the answer depends on it (the year the FACT refers to,
  which may differ from the filing's year: 10-K tables show three years).
- One unambiguous answer, stated in the passage. Prefer specific facts (a figure, a named item, a stated reason).
- Phrase it the way a person would ask, NOT by copying the passage's wording (e.g. "how many people work at" instead
  of "full-time equivalent employees"). Keep company and product names.
- answer: short and complete, with units. evidence: an EXACT verbatim quote (copy-paste, 30-300 characters) from
  the passage that contains the answer.
- usable=false if the passage has no clear, askable fact (boilerplate, table fragments without labels, legal
  cross-references)."""

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "keep": {"type": "boolean"}},
    "required": ["reasoning", "keep"], "additionalProperties": False,
}
CRITIC_SYSTEM = """You review evaluation questions for a QA system over SEC 10-K filings of Apple, Microsoft, NVIDIA,
Tesla and Amazon (two fiscal years each). keep=true only if ALL hold:
1. Standalone: someone without the passage knows which company and (if relevant) which fiscal year is meant.
2. Exactly one correct answer, and the given answer is it, fully supported by the evidence quote.
3. It is a genuine question (not a yes/no about trivia, not asking to quote legal text).
Reason first, then decide."""

COMPARISONS = [
    # (question, reference answer, {ticker: (chunk id prefix, regex that marks evidence)}, numbers the answer must state)
    ("Which had higher total revenue in its most recent fiscal year, Apple or Microsoft?",
     "Apple: total net sales of $416,161 million in fiscal 2025, vs Microsoft's total revenue of $331,839 million in fiscal 2026.",
     {"AAPL": ("AAPL_FY2025", r"Total net sales \| 416,161"), "MSFT": ("MSFT_FY2026", r"Total revenue \| 331,839")},
     ["416,161", "331,839"]),
    ("How did NVIDIA's revenue growth in its latest fiscal year compare with Microsoft's?",
     "NVIDIA's revenue grew 65% in fiscal 2026 (to $215.9 billion); Microsoft's grew 18% in fiscal 2026 ($50.1 billion increase).",
     {"NVDA": ("NVDA_FY2026", r"up 65%"), "MSFT": ("MSFT_FY2026", r"\$50\.1 billion or 18%")},
     ["65", "18"]),
    ("Which company earned more operating income in its latest fiscal year, Amazon or Apple?",
     "Apple: $133,050 million in fiscal 2025, vs Amazon's $79,975 million in fiscal 2025.",
     {"AAPL": ("AAPL_FY2025", r"Operating income \| 133,050"), "AMZN": ("AMZN_FY2025", r"Operating income \| 36,852 \| 68,593 \| 79,975")},
     ["133,050", "79,975"]),
    ("Compare the number of employees at NVIDIA and Apple at the end of their latest fiscal years.",
     "NVIDIA had approximately 42,000 employees (end of fiscal 2026); Apple had approximately 166,000 full-time equivalent employees (September 2025).",
     {"NVDA": ("NVDA_FY2026", r"approximately 42,000 employees"), "AAPL": ("AAPL_FY2025", r"approximately 166,000 full-time")},
     ["42,000", "166,000"]),
    ("Compare Tesla's and NVIDIA's overall gross margin in their latest fiscal years.",
     "Tesla's total gross margin was 18.0% in fiscal 2025; NVIDIA's gross margin was 71.1% in fiscal 2026.",
     {"TSLA": ("TSLA_FY2025", r"Total gross margin \| 18\.0"), "NVDA": ("NVDA_FY2026", r"Gross margin \| 71\.1")},
     ["18.0", "71.1"]),
    ("What were Apple's and Amazon's total net sales in their most recent fiscal years?",
     "Apple: $416,161 million (fiscal 2025); Amazon: $716,924 million (fiscal 2025).",
     {"AAPL": ("AAPL_FY2025", r"Total net sales \| 416,161"), "AMZN": ("AMZN_FY2025", r"716,924")},
     ["416,161", "716,924"]),
]

# Human review of the synthetic questions (the gpt-4.1 critic rejected NONE of them; these 9 slipped through).
# Keyed by question text so the review survives re-generation. Every drop has a reason.
REVIEW_DROPS = {
    "What does Apple identify as a principal competitive factor for its business in fiscal year 2024?":
        "open-ended: several factors are valid, the judge only knows one reference",
    "What is one competitive advantage of Azure mentioned by Microsoft in its fiscal year 2026 10-K?":
        "open-ended ('one ...'): many valid answers",
    "What is one risk that NVIDIA identified in its 2026 10-K related to its dependency on third-party suppliers?":
        "open-ended ('one risk')",
    "What is one way Tesla's Megapack battery line is designed to benefit utility-scale customers in 2024?":
        "open-ended ('one way')",
    "What are some of the electronic devices that Amazon manufactured and sold in fiscal year 2025?":
        "open-ended ('some of'): any subset is correct",
    "What are some of the payment methods Amazon accepted from customers in fiscal year 2024?":
        "open-ended ('some of')",
    "Does Microsoft have any employees represented by unions or works councils as of fiscal year 2026?":
        "yes/no: 50% guessable, tests nothing about retrieval",
    "Has NVIDIA experienced a security incident involving a third-party supplier in the past, according to the fiscal year 2025 10-K?":
        "yes/no",
    "For Amazon in 2025, did any vendor account for 10% or more of the company's purchases?":
        "yes/no",
}

UNANSWERABLE = [
    "What were Apple's total net sales in fiscal 2019?",           # year not covered by the indexed filings
    "How many subscribers did Netflix have at the end of 2025?",   # company not covered
    "What will NVIDIA's revenue be in fiscal 2028?",               # future: a 10-K doesn't state it
    "What is Tesla's current stock price?",                        # not in a 10-K
    "How much was Amazon's CEO paid in 2025?",                     # Items 10-14: incorporated by reference, not in the text
    "What was Meta's advertising revenue in 2025?",                # company not covered
    "How many employees did Microsoft have in 2015?",              # year not covered
    "Should I buy Apple stock before the next earnings report?",   # investment advice
]


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def relevant_for(rows: list[dict], ticker: str, pattern: str, source_fy: int | None, literal: bool) -> dict[str, int]:
    rel = {}
    for r in rows:
        if r["ticker"] != ticker:
            continue
        hit = (pattern in norm_ws(r["text"])) if literal else re.search(pattern, r["text"], re.I)
        if hit:
            rel[r["id"]] = 2 if source_fy is None or r["fiscal_year"] == source_fy else 1
    return rel


def distinctive(numbers: set[str]) -> set[str]:
    """Figures specific enough to identify a fact: ≥ 4 significant digits (416161, 5.1 is too generic)."""
    return {n for n in numbers if len(n.replace(".", "").lstrip("0")) >= 4}


def answer_bearing(rows: list[dict], ticker: str, numbers: set[str]) -> set[str]:
    if not numbers:
        return set()
    return {r["id"] for r in rows if r["ticker"] == ticker and numbers <= extract_numbers(r["text"], skip_trivial=False)}


def overlap(question: str, text: str) -> float:
    """Share of the question's content words that appear in the source chunk: high = too easy for BM25/dense."""
    q = set(tokenize(question))
    return len(q & set(tokenize(text))) / len(q) if q else 0.0


TEST_COMPARISONS = [
    ("Which company had more total assets at its latest fiscal year-end, Microsoft or Apple?",
     "Microsoft: $758,376 million (June 30, 2026), vs Apple's $359,241 million (September 27, 2025).",
     {"MSFT": ("MSFT_FY2026", r"Total assets \| 758,376"), "AAPL": ("AAPL_FY2025", r"Total assets \| 359,241")},
     ["758,376", "359,241"]),
    ("Compare Amazon's and Tesla's net income in their latest fiscal years.",
     "Amazon's net income was $77,670 million in 2025; Tesla's net income attributable to common stockholders was $3,794 million in 2025.",
     {"AMZN": ("AMZN_FY2025", r"Net income \| 30,425 \| 59,248 \| 77,670"),
      "TSLA": ("TSLA_FY2025", r"Net income attributable to common stockholders \| 3,794")},
     ["77,670", "3,794"]),
    ("Did NVIDIA or Microsoft spend more on research and development in their latest fiscal year?",
     "Microsoft: $35,562 million in fiscal 2026, vs NVIDIA's $18,497 million in fiscal 2026.",
     {"MSFT": ("MSFT_FY2026", r"Research and development \| 35,562"), "NVDA": ("NVDA_FY2026", r"Research and development \| 18,497")},
     ["35,562", "18,497"]),
    ("How did Apple's and Tesla's revenue growth compare in their latest fiscal years?",
     "Apple's total net sales grew 6% in fiscal 2025 (to $416,161 million); Tesla's total revenues declined 3% in 2025 (from $97,690 million to $94,827 million).",
     {"AAPL": ("AAPL_FY2025", r"Total net sales \| 416,161 \| 6"), "TSLA": ("TSLA_FY2025", r"Total revenues \| 94,827 \| 97,690")},
     ["6", "3"]),
]
TEST_UNANSWERABLE = [
    "What were NVIDIA's revenues in fiscal 2020?",                 # year not covered
    "How many cars will Tesla deliver in 2027?",                   # future
    "What is Microsoft's stock price today?",                      # not in a 10-K
    "What was Oracle's cloud revenue in fiscal 2025?",             # company not covered
    "Is Amazon stock undervalued right now?",                      # investment advice
]
# Review of the generated TEST questions for QUALITY ONLY (no system was run on them). 7 of 51 dropped.
TEST_REVIEW_DROPS = {
    "What was Tesla's net income for the fiscal year 2023?":
        "WRONG REFERENCE: generator misread the column order; $7,091M is FY2024, FY2023 is $14,997M",
    "According to Apple's 2024 10-K, what is Apple's market share position in the global smartphone, personal computer, and tablet markets?":
        "near-duplicate of the FY2025 'minority market share' question",
    "Who was the President and Chief Executive Officer of Amazon in fiscal year 2025?":
        "near-duplicate of the FY2024 CEO question",
    "What risks does Apple identify in its 2024 10-K related to defects in its products and services?":
        "open-ended list: many valid phrasings, hard to grade against one reference",
    "Who were some of NVIDIA's main competitors in the GPU and AI hardware market in fiscal year 2025?":
        "open-ended ('some of')",
    "What types of legal and regulatory issues did Amazon face in fiscal year 2025 according to its 10-K?":
        "open-ended list",
    "Has Microsoft experienced cybersecurity incidents involving unauthorized access to its systems and data as of fiscal year 2025?":
        "yes/no",
    "Does Tesla expect its current sources of funds to provide adequate liquidity during the 12 months following December 31, 2024?":
        "yes/no",
}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), default="dev")
    split = ap.parse_args().split
    cfg = SPLITS[split]
    out = Path(__file__).parent / f"golden_{split}.jsonl"
    comparisons, unanswerable = (COMPARISONS, UNANSWERABLE) if split == "dev" else (TEST_COMPARISONS, TEST_UNANSWERABLE)
    drops = REVIEW_DROPS if split == "dev" else TEST_REVIEW_DROPS
    rows = load_rows()
    rng = random.Random(cfg["seed"])
    used = set()
    if split == "test":  # never reuse a chunk the dev set was generated from
        dev = Path(__file__).parent / "golden_dev.jsonl"
        used = {json.loads(line).get("source_chunk") for line in dev.read_text().splitlines()}
    items, stats = [], {"sampled": 0, "unusable": 0, "bad_evidence": 0, "bad_numbers": 0, "critic_rejected": 0}

    for ticker in ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]:
        for stratum, sections in STRATA.items():
            pool = [r for r in rows if r["ticker"] == ticker and r["item"] in sections and len(r["text"]) > 800
                    and r["id"] not in used]
            for r in rng.sample(pool, min(cfg["per_stratum"], len(pool))):
                stats["sampled"] += 1
                header = f"{ticker} 10-K for fiscal year {r['fiscal_year']}, {r['item']}: {r['section']}"
                g, _ = chat_json([{"role": "system", "content": GEN_SYSTEM},
                                  {"role": "user", "content": f"{header}\n\n{r['text']}"}],
                                 GEN_SCHEMA, name="golden_question", model=JUDGE_MODEL)
                if not g["usable"]:
                    stats["unusable"] += 1
                    continue
                ev = norm_ws(g["evidence"])
                if len(ev) < 20 or ev not in norm_ws(r["text"]):          # the quote must really be in the chunk
                    stats["bad_evidence"] += 1
                    continue
                if g["answer_type"] == "numeric":
                    ans_nums = extract_numbers(g["answer"])
                    if not ans_nums or not ans_nums <= extract_numbers(g["evidence"], skip_trivial=False):
                        stats["bad_numbers"] += 1                         # answer figures must come from the quote
                        continue
                c, _ = chat_json([{"role": "system", "content": CRITIC_SYSTEM},
                                  {"role": "user", "content": f"Question: {g['question']}\nAnswer: {g['answer']}\n"
                                                              f"Evidence: {g['evidence']}\nSource: {header}"}],
                                 CRITIC_SCHEMA, name="golden_critic", model=JUDGE_MODEL)
                if not c["keep"]:
                    stats["critic_rejected"] += 1
                    continue
                if g["question"] in drops:
                    stats["human_dropped"] = stats.get("human_dropped", 0) + 1
                    continue
                items.append({
                    "id": f"{split}_syn_{len(items):03d}", "split": split, "source": "synthetic", "type": g["answer_type"], "stratum": stratum,
                    "question": g["question"], "answer": g["answer"], "evidence": g["evidence"],
                    "numbers": sorted(extract_numbers(g["answer"])) if g["answer_type"] == "numeric" else [],
                    "expected": "answer", "source_chunk": r["id"], "tickers": [ticker],
                    "relevant": {**{i: 1 for i in answer_bearing(rows, ticker, distinctive(extract_numbers(g["answer"])))
                                        if g["answer_type"] == "numeric"},
                                 **relevant_for(rows, ticker, ev, r["fiscal_year"], literal=True)},
                    "lexical_overlap": round(overlap(g["question"], r["text"]), 2),
                })

    for i, (q, ref, per, nums) in enumerate(comparisons):
        rel_by = {t: relevant_for(rows, t, pat, None, literal=False) for t, (pre, pat) in per.items()}
        rel_by = {t: {k: v for k, v in rel.items() if k.startswith(per[t][0])} for t, rel in rel_by.items()}
        if not all(rel_by.values()):
            raise ValueError(f"comparison without evidence: {q}")
        items.append({"id": f"{split}_cmp_{i:03d}", "split": split, "source": "manual", "type": "comparison", "stratum": "comparison",
                      "question": q, "answer": ref, "evidence": "", "numbers": nums, "expected": "answer",
                      "tickers": list(per), "relevant": {k: v for rel in rel_by.values() for k, v in rel.items()},
                      "relevant_by_ticker": {t: list(rel) for t, rel in rel_by.items()}})

    for i, q in enumerate(unanswerable):
        items.append({"id": f"{split}_una_{i:03d}", "split": split, "source": "manual", "type": "unanswerable", "stratum": "unanswerable",
                      "question": q, "answer": "(should refuse)", "evidence": "", "numbers": [], "expected": "refuse",
                      "tickers": [], "relevant": {}})

    out.write_text("\n".join(json.dumps(x) for x in items) + "\n")
    syn = [x for x in items if x["source"] == "synthetic"]
    print(f"synthetic: {stats}  -> kept {len(syn)}")
    print(f"  types: numeric={sum(x['type'] == 'numeric' for x in syn)} text={sum(x['type'] == 'text' for x in syn)}"
          f"  strata: " + ", ".join(f"{s}={sum(x['stratum'] == s for x in syn)}" for s in STRATA))
    print(f"  mean lexical overlap (question words found in source chunk): {sum(x['lexical_overlap'] for x in syn) / len(syn):.2f}")
    print(f"  relevant chunks per question: mean {sum(len(x['relevant']) for x in syn) / len(syn):.1f}")
    print(f"total: {len(items)} items ({len(syn)} synthetic, {len(comparisons)} comparison, {len(unanswerable)} unanswerable) -> {out.name}")


if __name__ == "__main__":
    main()
