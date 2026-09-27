"""Self-query: let an LLM extract the metadata filters (companies, which filings) from the question.

Run: uv run python -m phase3_query_intelligence.03_self_query

Compares, on questions that name companies INDIRECTLY or use relative time:
  rules  Phase 2's extract_filters(): alias table + "fiscal YYYY" regex. Free, instant, brittle.
  llm    common.query.analyze(): structured output over a catalog of what's indexed.
  plan   analyze() + deterministic policy (union with the rules, etc.): what 06_ask uses.
Filing years are scored per company: the question "latest fiscal year" means FY2026 for MSFT but FY2025 for AMZN.
"""
from common.filters import extract_filters
from common.query import analyze, plan

# (question, expected tickers, {ticker: expected filing years or None = don't care})
CASES = [
    ("What does the iPhone maker say about tariffs?", {"AAPL"}, {}),
    ("How did the Redmond software giant's cloud business grow?", {"MSFT"}, {}),
    ("What risks does Jensen Huang's company see in China?", {"NVDA"}, {}),
    ("How many vehicles did the EV maker deliver last year?", {"TSLA"}, {"TSLA": {2025}}),
    ("What does Bezos's company say about AWS competition?", {"AMZN"}, {}),
    ("What did the Cupertino company spend on R&D in fiscal 2023?", {"AAPL"}, {"AAPL": {2024, 2025}}),
    ("Compare the total revenue growth of Microsoft and Amazon in their latest fiscal year.",
     {"MSFT", "AMZN"}, {"MSFT": {2026}, "AMZN": {2025}}),
    ("How did GPU sales at the Blackwell maker change in fiscal 2026?", {"NVDA"}, {"NVDA": {2026}}),
    ("What was Apple's total net sales in fiscal 2025?", {"AAPL"}, {"AAPL": {2025}}),
    ("What was AWS operating income in 2025?", {"AMZN"}, {}),
]


def score(tickers: set, years: dict, case: tuple) -> tuple[bool, bool]:
    _, exp_t, exp_y = case
    return tickers == exp_t, all(years.get(t) and set(years[t]) <= ok for t, ok in exp_y.items())


def main() -> None:
    totals = {"rules": [0, 0], "llm": [0, 0], "plan": [0, 0]}
    print(f"{'question':62s} {'rules':>18s} {'llm':>22s} {'plan':>22s}")
    for case in CASES:
        q = case[0]
        f = extract_filters(q)
        rule_t, rule_y = set(f.get("ticker", [])), {t: f.get("fiscal_year", []) for t in f.get("ticker", [])}
        a, _ = analyze(q)
        llm_t, llm_y = set(a["companies"]), {s["ticker"]: s["filing_years"] for s in a["sub_questions"]}
        p, _, _ = plan(q)
        plan_t, plan_y = set(p["companies"]), {s["ticker"]: s["filing_years"] for s in p["sub_questions"]}
        row = []
        for name, t, y in [("rules", rule_t, rule_y), ("llm", llm_t, llm_y), ("plan", plan_t, plan_y)]:
            ok_t, ok_y = score(t, y, case)
            totals[name][0] += ok_t
            totals[name][1] += ok_y
            yrs = ",".join(f"{k}{v}" for k, v in y.items() if v) if y else ""
            row.append(f"{'✅' if ok_t and ok_y else '❌'} {'/'.join(sorted(t)) or '-'} {yrs}"[:22])
        print(f"{q[:62]:62s} {row[0]:>18s} {row[1]:>22s} {row[2]:>22s}")
    n = len(CASES)
    print("\n" + "  ".join(f"{k}: companies {v[0]}/{n}, filings {v[1]}/{n}" for k, v in totals.items()))
    print("rules can't resolve descriptions ('the iPhone maker') or relative time ('latest', 'last year');")
    print("the LLM can, because it gets a catalog of what's indexed and today's date in its prompt.")


if __name__ == "__main__":
    main()
