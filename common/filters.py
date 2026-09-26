"""Metadata filters: restrict retrieval to chunks whose metadata matches (ticker, fiscal year).

A filter is a plain dict:  {"ticker": ["MSFT", "AMZN"], "fiscal_year": [2025]}   (values OR-ed, keys AND-ed)
The same filter is applied to both retrievers:
  to_chroma_where()  -> Chroma `where=` clause (dense side, filtered inside the vector DB)
  to_mask()          -> boolean array over chunk rows (BM25 side, our own index)

extract_filters() is deliberately rule-based (aliases + regex): transparent and free, but brittle.
Phase 3 replaces it with LLM-based query understanding ("self-query").
"""
import re

import numpy as np

COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl", "iphone"],
    "MSFT": ["microsoft", "msft", "azure"],
    "NVDA": ["nvidia", "nvda"],
    "TSLA": ["tesla", "tsla"],
    "AMZN": ["amazon", "amzn", "aws"],
}
# Only fiscal-qualified years: "FY2025", "fiscal 2025", "fiscal year 2025". A bare "2025" is ambiguous
# (calendar year? the filing year? a year mentioned inside a 10-K that covers three years?)
FY_RE = re.compile(r"\b(?:fy\s?|fiscal\s+(?:year\s+)?)'?(20\d\d)\b", re.I)


def extract_filters(query: str) -> dict:
    q = query.lower()
    tickers = [t for t, names in COMPANY_ALIASES.items() if any(re.search(rf"\b{n}\b", q) for n in names)]
    years = sorted({int(y) for y in FY_RE.findall(query)})
    f = {}
    if tickers:
        f["ticker"] = tickers
    if years:
        f["fiscal_year"] = years
    return f


def to_chroma_where(f: dict | None) -> dict | None:
    if not f:
        return None
    clauses = [{k: {"$in": v}} for k, v in f.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def to_mask(f: dict | None, rows: list[dict]) -> np.ndarray | None:
    if not f:
        return None
    return np.array([all(r[k] in v for k, v in f.items()) for r in rows])
