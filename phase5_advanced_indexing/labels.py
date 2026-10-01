"""Relevance labels for a NEW chunking. The golden sets label Phase 1 chunk ids (AAPL_FY2025_Item8_049), which mean
nothing in another chunk set. The labels are re-derived with Phase 4's own rules (01_build_golden), applied to the
new chunks' BODY (the filing text, without the contextual header):
  synthetic   grade 2: the evidence quote is in the chunk, in the source filing; grade 1: in the other year's filing,
              or (numeric answers) the chunk contains all of the answer's distinctive figures
  comparison  the hand-verified regex per company (COMPARISONS / TEST_COMPARISONS / TEST2_COMPARISONS)
A question whose evidence quote is split across two new chunks gets only number-based labels, or none: counted
and reported by 05_retrieval_eval, because a chunking that "loses" labels would otherwise look worse (or better).
"""
import importlib
import re

golden = importlib.import_module("phase4_evaluation.01_build_golden")
_PATTERNS = {q: per for q, _, per, _ in golden.COMPARISONS + golden.TEST_COMPARISONS + golden.TEST2_COMPARISONS}


def relabel(item: dict, rows: list[dict]) -> dict[str, int]:
    body_rows = [{**r, "text": r.get("body", r["text"])} for r in rows]
    if item["type"] == "comparison":
        rel = {}
        for t, (prefix, pat) in _PATTERNS[item["question"]].items():
            rel.update({k: v for k, v in golden.relevant_for(body_rows, t, pat, None, literal=False).items() if k.startswith(prefix)})
        return rel
    if item["expected"] != "answer":
        return {}
    ticker = item["tickers"][0]
    source_fy = int(re.search(r"_FY(\d{4})_", item["source_chunk"]).group(1))
    rel = {}
    if item["type"] == "numeric":
        nums = golden.distinctive(golden.extract_numbers(item["answer"]))
        rel.update({i: 1 for i in golden.answer_bearing(body_rows, ticker, nums)})
    rel.update(golden.relevant_for(body_rows, ticker, golden.norm_ws(item["evidence"]), source_fy, literal=True))
    return rel


def relabel_by_ticker(item: dict, rows: list[dict]) -> dict[str, list[str]]:
    """Comparison questions: relevant chunk ids per company (for entity coverage)."""
    body_rows = [{**r, "text": r.get("body", r["text"])} for r in rows]
    return {t: [k for k in golden.relevant_for(body_rows, t, pat, None, literal=False) if k.startswith(prefix)]
            for t, (prefix, pat) in _PATTERNS[item["question"]].items()}
