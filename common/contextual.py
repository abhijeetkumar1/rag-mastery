"""Contextual chunk headers: make every chunk say WHERE it comes from, inside the text that gets embedded,
BM25-indexed and shown to the LLM.

A chunk like "Total gross margin | 18.0 | 17.9 | 18.2" is ambiguous on its own: whose margin? which filing?
Metadata (ticker, fiscal_year) helps FILTERING, but metadata is not in the vector and not in the prompt. Two
ways to put the context into the text itself:

  context_header()  deterministic, from metadata: company, form, fiscal year and period end, Item, table caption.
                    Free, instant, can't hallucinate. Same for every chunk of a section: it adds no chunk-specific
                    meaning.
  llm_context()     Anthropic's "Contextual Retrieval" (2024): an LLM writes 1-2 sentences situating the chunk in
                    its document ("This table from Tesla's 2025 MD&A breaks down gross profit and margin by
                    segment..."). Chunk-specific, costs one LLM call per chunk at index time, can be wrong.
"""
from datetime import date

from common.llm import chat_json

COMPANY_NAMES = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation",
                 "TSLA": "Tesla, Inc.", "AMZN": "Amazon.com, Inc."}
# Standard 10-K item names (Form 10-K General Instructions). The parsed titles are the filers' own and vary in
# case and wording ("MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION..."); one canonical name is cleaner.
ITEM_NAMES = {
    "Cover": "Cover page", "Item 1": "Business", "Item 1A": "Risk Factors", "Item 1B": "Unresolved Staff Comments",
    "Item 1C": "Cybersecurity", "Item 2": "Properties", "Item 3": "Legal Proceedings", "Item 4": "Mine Safety Disclosures",
    "Item 5": "Market for Common Equity and Stockholder Matters", "Item 6": "Reserved",
    "Item 7": "Management's Discussion and Analysis (MD&A)", "Item 7A": "Market Risk Disclosures",
    "Item 8": "Financial Statements and Supplementary Data", "Item 9": "Changes in and Disagreements with Accountants",
    "Item 9A": "Controls and Procedures", "Item 9B": "Other Information", "Item 9C": "Foreign Jurisdiction Disclosure",
    "Item 10": "Directors, Executive Officers and Corporate Governance", "Item 11": "Executive Compensation",
    "Item 12": "Security Ownership", "Item 13": "Certain Relationships and Related Transactions",
    "Item 14": "Principal Accountant Fees and Services", "Item 15": "Exhibits and Financial Statement Schedules",
    "Item 16": "Form 10-K Summary",
}


def context_header(ticker: str, fiscal_year: int, period_end: str, item: str, caption: str = "") -> str:
    """e.g. "[Tesla, Inc. (TSLA) | Form 10-K, fiscal year 2025, ended December 31, 2025 | Item 7: Management's
    Discussion and Analysis (MD&A) | Table: Cost of Revenues and Gross Margin]"."""
    end = date.fromisoformat(period_end)
    parts = [f"{COMPANY_NAMES.get(ticker, ticker)} ({ticker})",
             f"Form 10-K, fiscal year {fiscal_year}, ended {end:%B} {end.day}, {end.year}",
             f"{item}: {ITEM_NAMES.get(item, item)}" if item != "Cover" else "Cover page"]
    if caption:
        parts.append(f"Table: {caption[:150]}")
    return "[" + " | ".join(parts) + "]"


CONTEXT_SCHEMA = {"type": "object", "properties": {"context": {"type": "string"}},
                  "required": ["context"], "additionalProperties": False}
CONTEXT_SYSTEM = """You help a search engine index SEC 10-K filings. Given a CHUNK and the text around it, write
1-2 short sentences (max 40 words) that situate the chunk within the filing, to improve search retrieval of the chunk.
The filing line (company, fiscal year, Item) is ALREADY attached to the chunk: do not repeat it, and do not start
with "This chunk". Say what the chunk is about: the topic or subsection it belongs to (often named in the text
before it), what it explains, and for a table what its rows and columns measure (e.g. "gross profit and gross
margin by segment, 2023-2025"). Use only the text given. Do not restate the chunk's numbers."""


def llm_context(header: str, before: str, chunk: str, after: str, model: str | None = None) -> tuple[str, dict]:
    """One cached chat_json call. Anthropic's recipe puts the WHOLE document in the prompt (with prompt caching);
    a 10-K is ~100k tokens, so here the window is the chunk's neighbours instead (~1k tokens): far cheaper, but
    the LLM can't see facts stated many pages away."""
    user = (f"Filing: {header}\n\n<before>\n{before}\n</before>\n\n<chunk>\n{chunk}\n</chunk>\n\n"
            f"<after>\n{after}\n</after>")
    kw = {"model": model} if model else {}
    out, usage = chat_json([{"role": "system", "content": CONTEXT_SYSTEM}, {"role": "user", "content": user}],
                           CONTEXT_SCHEMA, name="chunk_context", **kw)
    return out["context"], usage
