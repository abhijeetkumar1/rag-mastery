"""Query intelligence: reshape the user's question BEFORE retrieval. All LLM calls are structured
(JSON schema) and cached (common.llm.chat_json), and return usage so callers can trace cost.

  rewrite()      question -> one search query in the documents' vocabulary ("revenue" -> "net sales")
  multi_query()  question -> N diverse search queries, retrieved separately and fused (RAG-Fusion)
  hyde()         question -> a HYPOTHETICAL 10-K passage that would answer it; embed that instead
  analyze()      question -> route + filters + sub-questions in ONE call (self-query + decomposition + routing)
"""
import json
import re
from datetime import date

from common.config import DATA_DIR
from common.filters import extract_filters
from common.llm import chat_json


def catalog() -> dict[str, list[int]]:
    """What is actually indexed, e.g. {"AAPL": [2024, 2025], ...}. Given to the analyzer so it can resolve
    "latest fiscal year" per company and recognise companies we DON'T cover."""
    cat: dict[str, list[int]] = {}
    for d in json.loads((DATA_DIR / "raw" / "manifest.json").read_text()):
        cat.setdefault(d["ticker"], []).append(d["fiscal_year"])
    return {t: sorted(ys) for t, ys in cat.items()}


COMPANIES = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "TSLA": "Tesla", "AMZN": "Amazon"}

# Domain vocabulary: how 10-K filings phrase things users ask casually. Real systems keep a glossary
# like this (or learn it from query logs). NOTE: written while looking at our probe failures, so gains
# on those probes are optimistic; Phase 4 measures on held-out questions.
VOCAB = """10-K vocabulary hints:
- revenue: Amazon and Apple say "net sales" / "total net sales"; Microsoft and NVIDIA say "revenue"; Tesla says "total revenues"
- buybacks: "repurchased shares", "share repurchase program"
- workforce / headcount / staff: "employees", "full-time and part-time employees"
- profit: "operating income", "net income"; margin: "gross margin"
- capex: "purchases of property and equipment", "capital expenditures"
- growth: "increased X% compared to prior year", "year-over-year percentage growth" """


def rewrite(question: str, domain: bool = True) -> tuple[str, dict]:
    """One retrieval-friendly query. domain=False = generic rewrite, for the ablation in 01_rewrite."""
    system = ("Rewrite the user's question into ONE search query for retrieving passages from SEC 10-K "
              "annual reports. Keep company names, products, years and numbers. Remove filler words. "
              "Use the words a 10-K would use, and add close synonyms.")
    if domain:
        system += "\n\n" + VOCAB
    out, usage = chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False},
        name="rewrite")
    return out["query"], usage


def multi_query(question: str, n: int = 3) -> tuple[list[str], dict]:
    """N different phrasings: each retrieves different chunks, RRF fuses them (RAG-Fusion)."""
    system = (f"Write {n} DIFFERENT search queries for retrieving SEC 10-K passages that answer the user's "
              "question. Vary the wording: one close to the question, one in formal 10-K language, one "
              "focused on the specific table or section where the answer would appear.\n\n" + VOCAB)
    out, usage = chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
         "required": ["queries"], "additionalProperties": False},
        name="multi_query")
    return out["queries"][:n], usage


def hyde(question: str) -> tuple[str, dict]:
    """HyDE (Gao et al., 2022): a fake answer passage lives closer to real answer passages in embedding
    space than a short question does. Its facts may be wrong; only its SHAPE and VOCABULARY matter."""
    system = ("Write a short passage (3-5 sentences) as it would appear in a company's SEC 10-K annual report "
              "and that answers the question. Use the formal style and terminology of 10-K filings, including "
              "the table or section wording. If you don't know exact figures, write plausible ones.")
    out, usage = chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        {"type": "object", "properties": {"passage": {"type": "string"}}, "required": ["passage"], "additionalProperties": False},
        name="hyde")
    return out["passage"], usage


ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {  # ORDER MATTERS: the model writes fields in this order (see v3 below)
        "route": {"type": "string", "enum": ["answer", "unsupported_company", "investment_advice", "out_of_scope"]},
        "reason": {"type": "string"},
        "companies": {"type": "array", "items": {"type": "string", "enum": list(COMPANIES)}},
        "unsupported_companies": {"type": "array", "items": {"type": "string"}},
        "sub_questions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "ticker": {"type": ["string", "null"], "enum": [*COMPANIES, None]},
                "filing_years": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["question", "ticker", "filing_years"], "additionalProperties": False}},
    },
    "required": ["route", "reason", "companies", "unsupported_companies", "sub_questions"],
    "additionalProperties": False,
}


# v2 additions, each fixing a measured v1 failure on phase3_query_intelligence/cases.py (v1: 17/22 all-correct)
V2_RULES = """
Scope: 10-K filings cover business and products, strategy, competition, risks (including dependence on key people
such as the CEO), acquisitions, legal proceedings, employees, and financial results. Questions on ANY of these for a
listed company are "answer". Use "out_of_scope" only for questions unrelated to company filings (general knowledge,
product support or how-to, creative writing).
Sub-question wording: translate the user's casual terms into the hints' 10-K terms for THAT company (workforce ->
employees; Amazon revenue -> net sales), but never change WHICH metric is asked: deliveries stay deliveries."""
# v3: same prompt as v2, but the schema asks for reason -> companies -> route: the model identifies the companies
# and explains BEFORE it commits to a route (generation is left to right; a decision can't use reasoning written after it)
ANALYZE_SCHEMA_V3 = {**ANALYZE_SCHEMA, "properties": {k: ANALYZE_SCHEMA["properties"][k] for k in
                     ["reason", "companies", "unsupported_companies", "route", "sub_questions"]}}
PROMPT_VERSION = "v3"


def analyze(question: str, version: str = PROMPT_VERSION) -> tuple[dict, dict]:
    """Route + self-query + decomposition in one structured call. `version` selects the prompt ("v1" kept for
    comparison in 05_route); record it in traces so an answer can be tied to the prompt that produced it."""
    cat = catalog()
    # "latest" is computed here, not left for the model to infer: a prompt change once flipped MSFT's latest to FY2025
    coverage = "\n".join(f"- {t} ({COMPANIES[t]}): 10-K filings for fiscal years {ys} (latest: FY{ys[-1]})"
                         for t, ys in cat.items())
    # Today's date grounds relative time ("last year", "latest"): the model can't know it otherwise
    system = f"""You plan retrieval for a question-answering system over SEC 10-K filings. Today is {date.today():%Y-%m-%d}.
Indexed filings (nothing else is available):
{coverage}

Return:
- route:
  "answer" if the question can be answered from these filings;
  "unsupported_company" ONLY if it asks about a company NOT in the list (put its name in unsupported_companies).
    A listed company with a year that has no filing of its own is still "answer": use the next filing, which reports it;
  "investment_advice" if it asks for a recommendation (buy/sell/hold, price targets, "should I invest");
  "out_of_scope" if it is not about these companies' filings at all.
- companies: tickers of the listed companies the question is about. Resolve descriptions ("the iPhone maker" = AAPL).
- sub_questions (only when route is "answer"): ONE per company and per distinct fact needed. Each is a standalone
  search query naming the company, in THAT company's 10-K vocabulary (see hints: Amazon "net sales", Microsoft
  "revenue"). Keep the period the user asked about in the question text (e.g. "fiscal 2023"), even when you search
  a later filing. ticker = that company.
  filing_years = which filings to search: a 10-K for fiscal year X reports figures for X, X-1 and X-2, so a question
  about fiscal year X is best answered by the FY X filing, or by FY X+1 if FY X is not indexed. "Latest fiscal year"
  = that company's newest filing. "Last year" = the most recent fiscal year that has ended, per company.
  Use [] when the question doesn't depend on a year.
{V2_RULES if version in ("v2", "v3") else ""}
{VOCAB}"""
    schema = ANALYZE_SCHEMA_V3 if version == "v3" else ANALYZE_SCHEMA
    return chat_json([{"role": "system", "content": system}, {"role": "user", "content": question}],
                     schema, name="analyze")


# ---- plan(): LLM analysis + deterministic policy. The LLM proposes, code enforces what shouldn't be a judgment call.
# Measured motivation (05_route): the router refused "AWS operating income" and "Activision" as out_of_scope,
# explaining that those "are not in the 10-K filings". It was guessing about content it never saw.

# Per-company vocabulary, appended (not substituted) to sub-questions so the original words stay searchable too
TICKER_TERMS = {
    "AMZN": [(r"\brevenues?\b", "net sales")],
    "AAPL": [(r"\brevenues?\b", "net sales")],
    "TSLA": [(r"\brevenues?\b", "total revenues")],
}
GENERIC_TERMS = [
    (r"\b(workforce|headcount|staff)\b", "employees"),
    (r"\bbuybacks?\b", "share repurchases"),
    (r"\b(capex|capital expenditures?)\b", "purchases of property and equipment"),
]


def expand_terms(text: str, ticker: str | None) -> str:
    extra = [term for pat, term in GENERIC_TERMS + TICKER_TERMS.get(ticker or "", [])
             if re.search(pat, text, re.I) and term.lower() not in text.lower()]
    return f"{text} ({', '.join(extra)})" if extra else text


def plan(question: str, version: str = PROMPT_VERSION) -> tuple[dict, dict, list[str]]:
    """analyze() + policy. Returns (plan, usage, overrides) where overrides lists every rule that fired."""
    a, usage = analyze(question, version)
    a = json.loads(json.dumps(a))  # don't mutate the cached object
    overrides = []
    regex_tickers = extract_filters(question).get("ticker", [])
    companies = list(dict.fromkeys(a["companies"] + regex_tickers))

    # 1. never refuse out_of_scope / unsupported_company when a covered company is named. (Phase 4 found the
    #    second case: "...Activision Blizzard's senior notes, as referenced in Microsoft's 10-K" was refused because
    #    Activision counted as an uncovered company, although the question is about Microsoft's filing.)
    if companies and a["route"] in ("out_of_scope", "unsupported_company"):
        overrides.append(f"route {a['route']} -> answer: covered company named ({', '.join(companies)})")
        a["route"] = "answer"
    a["companies"] = companies
    if a["route"] == "answer":
        # 2. every named company gets a sub-question (the model sometimes drops one)
        have = {s["ticker"] for s in a["sub_questions"]}
        for t in companies:
            if t not in have:
                a["sub_questions"].append({"question": question, "ticker": t, "filing_years": []})
                overrides.append(f"added sub-question for {t}")
        if not a["sub_questions"]:
            a["sub_questions"] = [{"question": question, "ticker": None, "filing_years": []}]
        # 3. explicit years -> filings computed in code: year X is reported by the FY X filing AND by FY X+1
        #    (comparative column). Phase 4 found the LLM sending "fiscal year 2024" to the FY2025 filing only,
        #    missing facts stated only in the FY2024 10-K. The LLM's choice stays when no such filing is indexed.
        cat = catalog()
        for s in a["sub_questions"]:
            years = {int(y) for y in re.findall(r"\b(20\d\d)\b", s["question"])}
            want = sorted({y for x in years for y in (x, x + 1)} & set(cat.get(s["ticker"] or "", [])))
            if want and want != sorted(s["filing_years"]):
                overrides.append(f"filings {s['ticker']}: {s['filing_years']} -> {want}")
                s["filing_years"] = want
        # 4. deterministic vocabulary expansion per company
        for s in a["sub_questions"]:
            new = expand_terms(s["question"], s["ticker"])
            if new != s["question"]:
                overrides.append(f"vocab: {s['ticker']}: + {new[len(s['question']):].strip()}")
                s["question"] = new
    return a, usage, overrides
