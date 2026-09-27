"""Labeled cases for Phase 3. Two kinds:

ANALYZER_CASES: expected output of common.query.analyze() (route, companies, which filings to search,
                words each sub-question must keep). A regression test for the prompt: prompt edits
                changed Microsoft's "latest fiscal year" and turned Tesla "deliveries" into "revenues".
COMPARISON_PROBES: multi-company questions with a text-verified relevant chunk PER company. Scored by
                entity coverage: for how many companies is the evidence in the top k? (recall across
                entities, the thing a single top-k list can't guarantee)

Retrieval probes for single-entity questions are reused from phase2_better_retrieval/probes.py.
Written after seeing Phase 2's failures and the first analyzer outputs, so they are NOT held out:
treat scores as optimistic. Phase 4 builds a separate evaluation set.
"""
import re

from phase2_better_retrieval.probes import load_rows

# (question, route, companies (set, or None = don't check), {ticker: allowed filing years (set) or None = any},
#  {ticker: regex the sub-question for that ticker must match})
ANALYZER_CASES = [
    ("Compare the total revenue growth of Microsoft and Amazon in their latest fiscal year.", "answer",
     {"MSFT", "AMZN"}, {"MSFT": {2026}, "AMZN": {2025}}, {"MSFT": r"revenue", "AMZN": r"net sales"}),
    ("What was Apple's total net sales in fiscal 2023?", "answer",
     {"AAPL"}, {"AAPL": {2024, 2025}}, {"AAPL": r"2023"}),
    ("How did the EV maker's deliveries change last year?", "answer",
     {"TSLA"}, {"TSLA": {2025}}, {"TSLA": r"deliver"}),
    ("How big is Amazon's workforce?", "answer", {"AMZN"}, {"AMZN": None}, {"AMZN": r"employ"}),
    ("Which has more employees, NVIDIA or Microsoft?", "answer",
     {"NVDA", "MSFT"}, {"NVDA": None, "MSFT": None}, {"NVDA": r"employ", "MSFT": r"employ|people"}),
    ("Compare Apple's and NVIDIA's R&D spending in their latest fiscal years.", "answer",
     {"AAPL", "NVDA"}, {"AAPL": {2025}, "NVDA": {2026}}, {"AAPL": r"research|r&d", "NVDA": r"research|r&d"}),
    ("What does the iPhone maker say about tariffs?", "answer", {"AAPL"}, {"AAPL": None}, {"AAPL": r"tariff"}),
    ("Is Tesla too reliant on its CEO?", "answer", {"TSLA"}, {"TSLA": None}, {"TSLA": r"musk|ceo|chief executive"}),
    ("What was AWS operating income in 2025?", "answer", {"AMZN"}, {"AMZN": {2025}}, {"AMZN": r"operating income"}),
    ("How much did Apple spend on share buybacks in fiscal 2025?", "answer",
     {"AAPL"}, {"AAPL": {2025}}, {"AAPL": r"repurchas|buyback"}),
    ("What risks does NVIDIA see from its reliance on TSMC?", "answer",
     {"NVDA"}, {"NVDA": None}, {"NVDA": r"tsmc|taiwan semiconductor|foundr|supplier"}),
    ("What did Microsoft say about the Activision Blizzard acquisition?", "answer",
     {"MSFT"}, {"MSFT": None}, {"MSFT": r"activision"}),
    ("Which companies mention export controls?", "answer", None, {}, {}),
    ("What was Google's advertising revenue in 2025?", "unsupported_company", None, {}, {}),
    ("How many employees does Netflix have?", "unsupported_company", None, {}, {}),
    ("What was Meta's capital expenditure in 2025?", "unsupported_company", None, {}, {}),
    ("Should I buy NVIDIA stock?", "investment_advice", None, {}, {}),
    ("Is Tesla a good investment right now given its margins?", "investment_advice", None, {}, {}),
    ("What is a fair price target for Apple shares?", "investment_advice", None, {}, {}),
    ("What's the capital of France?", "out_of_scope", None, {}, {}),
    ("Write me a poem about GPUs.", "out_of_scope", None, {}, {}),
    ("How do I reset my iPhone?", "out_of_scope", None, {}, {}),
]

# (question, {ticker: (id prefix, regex over chunk text)}): one verified relevant chunk set per company
COMPARISON_PROBES = [
    ("Compare the total revenue growth of Microsoft and Amazon in their latest fiscal year.",
     # MSFT: the MD&A sentence OR the income statement (total revenue 331,xxx): both let you compute the growth.
     # The first version accepted only the sentence; the income statement was valid evidence too (incomplete labels)
     {"MSFT": ("MSFT_FY2026", r"revenue increased \$50\.1 billion or 18%|331,\d{3}"),
      "AMZN": ("AMZN_FY2025", r"Consolidated \| 637,959 \| 716,924")}),
    ("Which has more employees, NVIDIA or Microsoft?",
     {"NVDA": ("NVDA_FY2026", r"approximately 42,000 employees"),
      "MSFT": ("MSFT_FY2026", r"approximately 223,000 people")}),
    ("Compare Apple's and NVIDIA's R&D spending in their latest fiscal years.",
     {"AAPL": ("AAPL_FY2025", r"Research and development \| 34,550"),
      "NVDA": ("NVDA_FY2026", r"Research and development (expenses? )?\| \$?[\d,]{5,}")}),
    ("Does Amazon or Apple have more employees?",
     {"AMZN": ("AMZN_FY2025", r"approximately 1,576,000"),
      "AAPL": ("AAPL_FY2025", r"approximately 166,000 full-time")}),
]


def comparison_relevant(rows: list[dict] | None = None) -> list[dict[str, set[str]]]:
    rows = rows or load_rows()
    out = []
    for q, per in COMPARISON_PROBES:
        rel = {t: {r["id"] for r in rows if r["id"].startswith(p) and re.search(pat, r["text"], re.I)}
               for t, (p, pat) in per.items()}
        if not all(rel.values()):
            raise ValueError(f"comparison probe has a company with no relevant chunk: {q!r}")
        out.append(rel)
    return out


def score_analysis(case: tuple, a: dict) -> dict[str, bool]:
    """Per-check pass/fail for one analyzer output."""
    q, route, companies, years, keep = case
    subs = {s["ticker"]: s for s in a["sub_questions"]}
    return {
        "route": a["route"] == route,
        "companies": companies is None or set(a["companies"]) == companies,
        "filings": all(t in subs and (ok is None or (subs[t]["filing_years"] and set(subs[t]["filing_years"]) <= ok))
                       for t, ok in years.items()),
        "keeps_meaning": all(t in subs and re.search(pat, subs[t]["question"], re.I) for t, pat in keep.items()),
    }


def run_probes(strategy, rows: list[dict] | None = None) -> dict:
    """Score a retrieval strategy on the 12 Phase 2 probes. strategy(question) -> (ranked ids, llm_usage dict).
    Returns hit@5, MRR@5, hit@50 (candidate pool), per-kind hit@5, LLM tokens spent and the misses."""
    from phase2_better_retrieval.probes import PROBES, hit_and_rr, relevant_ids
    rel = relevant_ids(rows or load_rows())
    h5 = rr = h50 = tokens = 0
    kinds = {"keyword": 0, "semantic": 0, "numeric": 0}
    misses = []
    for (q, kind, *_), r in zip(PROBES, rel):
        ids, usage = strategy(q)
        h, x = hit_and_rr(ids, r, 5)
        h5, rr, kinds[kind] = h5 + h, rr + x, kinds[kind] + h
        h50 += hit_and_rr(ids, r, 50)[0]
        tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if not h:
            misses.append(q)
    return {"hit@5": h5, "mrr@5": rr / len(PROBES), "hit@50": h50, "kinds": kinds, "tokens": tokens, "misses": misses, "n": len(PROBES)}
