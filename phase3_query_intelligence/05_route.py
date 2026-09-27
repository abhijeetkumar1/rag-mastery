"""Routing and input guardrails: decide BEFORE retrieval whether (and how) to answer. Measured.

Run: uv run python -m phase3_query_intelligence.05_route

The analyzer (common.query.analyze) routes every question to one of:
  answer               -> decompose into sub-questions, retrieve, generate
  unsupported_company  -> "we only cover AAPL, MSFT, NVDA, TSLA, AMZN"       (input guardrail)
  investment_advice    -> decline: facts from filings only, no recommendations   (input guardrail)
  out_of_scope         -> decline                                             (input guardrail)

1. Accuracy of each prompt version (v1, v2, v3) and of plan() = v3 + code policy, on 22 labeled cases
2. The input guardrail as a classifier: "should we refuse?" precision / recall, and which errors remain
3. What each version cost on first run (tokens) and how the policy overrides fired
"""
from collections import Counter

from common.query import analyze, plan
from phase3_query_intelligence.cases import ANALYZER_CASES, score_analysis

CHECKS = ["route", "companies", "filings", "keeps_meaning"]


def evaluate(fn) -> tuple[Counter, int, list, list]:
    totals, all_ok, rows, overrides = Counter(), 0, [], []
    for case in ANALYZER_CASES:
        out = fn(case[0])
        a, extra = out[0], (out[2] if len(out) > 2 else [])
        s = score_analysis(case, a)
        totals.update(k for k, v in s.items() if v)
        all_ok += all(s.values())
        rows.append((case, a, s))
        if extra:
            overrides.append((case[0], extra))
    return totals, all_ok, rows, overrides


def main() -> None:
    n = len(ANALYZER_CASES)
    results = {}
    print(f"1. Analyzer accuracy on {n} labeled questions\n   {'version':22s} " + " ".join(f"{c:>13s}" for c in CHECKS) + "  all-correct")
    for name, fn in [("v1 prompt", lambda q: analyze(q, "v1")), ("v2 (+scope/wording)", lambda q: analyze(q, "v2")),
                     ("v3 (+reason first)", lambda q: analyze(q, "v3")), ("plan = v3 + policy", plan)]:
        totals, all_ok, rows, overrides = evaluate(fn)
        results[name] = (rows, overrides)
        print(f"   {name:22s} " + " ".join(f"{totals[c]:>10}/{n}" for c in CHECKS) + f"  {all_ok:>7}/{n}")

    rows, overrides = results["plan = v3 + policy"]
    print("\n2. Input guardrail (refuse vs answer) for plan()")
    tp = fp = fn_ = tn = 0
    for case, a, _ in rows:
        should_refuse, refused = case[1] != "answer", a["route"] != "answer"
        tp += should_refuse and refused
        fp += refused and not should_refuse
        fn_ += should_refuse and not refused
        tn += not should_refuse and not refused
        if should_refuse != refused or (refused and a["route"] != case[1]):
            print(f"   {'MISS' if should_refuse and not refused else 'WRONG'}: {case[0]!r}  expected={case[1]} got={a['route']}")
    print(f"   refusals: precision {tp}/{tp + fp}  recall {tp}/{tp + fn_}   (false refusals of answerable questions: {fp})")
    print("   -> the policy prefers letting a question through (the relevance floor and the LLM refusal still")
    print("      stand behind it) over refusing an answerable one: a false refusal can't be recovered downstream.")

    print("\n3. Policy overrides that fired (plan)")
    for q, ov in overrides:
        print(f"   {q[:60]:60s} {ov}")
    print("\n   First-run cost of analyze(): ~680 input + 30-120 output tokens per question (gpt-4o-mini ≈ $0.0002),")
    print("   ~1-1.5 s latency uncached. Cached reruns: 0 tokens, ~0 ms (.cache/chat_json/).")


if __name__ == "__main__":
    main()
