"""Evaluate the EVALUATORS: do the faithfulness checks catch "right number, wrong label"? (meta-evaluation)

Run: uv run python -m phase4_evaluation.03_judges

16 labeled cases built from real failures in Phases 2-3 (answer + the passages it was generated from + the
correct verdict). Four checkers:
  strict numeric     Phase 1 guardrail: every number in the answer appears in a passage
  tolerant numeric   Phase 3 guardrail: + unit conversions / growth from one table row
  LLM judge (mini)   claim-level faithfulness judge with gpt-4o-mini (the SAME model that wrote the answers)
  LLM judge (4.1)    the same judge prompt with gpt-4.1
An answer counts as "flagged" if the checker finds any unsupported/contradicted claim (or ungrounded number).
We care most about RECALL on unfaithful answers (a missed wrong number reaches the user) and about FALSE
ALARMS on faithful ones (a noisy check gets ignored or blocks good answers).
"""
import time

from common.config import JUDGE_MODEL
from common.evaluation import judge_faithfulness, judge_faithfulness_v2
from common.guardrails import check_answer
from common.trace import cost_usd
from phase2_better_retrieval.probes import load_rows

# (label, faithful?, chunk ids given as passages [1], [2]..., answer)
# "faithful" means SUPPORTED BY THE PASSAGES AS GIVEN, not "true". Three cases were first labeled faithful because the
# answers are true; the gpt-4.1 judge flagged them, and it was right: those chunks don't contain the table's column
# header (2025 | 2024 | 2023 sits in the previous chunk), so the passage can't show which year a number belongs to.
# Relabeled False ("true, but not verifiable from the passage"). The chunking bug behind it is fixed in Phase 5.
CASES = [
    # --- the three "right number, wrong label" failures found in Phases 2-3
    ("MSFT segment reported as total", False, ["MSFT_FY2026_Item7_010"],
     "Microsoft's total revenue increased by $19.2 billion or 16% in fiscal 2026 [1]."),
    ("AMZN net capex as purchases", False, ["AMZN_FY2025_Item7_028"],
     "Amazon spent $128.3 billion on purchases of property and equipment in 2025 [1]."),
    ("TSLA segment margin as total", False, ["TSLA_FY2025_Item7_025"],
     "Tesla's gross margin in fiscal 2025 was 16.2% [1]."),
    ("AMZN guidance as actual growth", False, ["AMZN_FY2025_Item7_030"],
     "Amazon's net sales grew 15% in 2025 [1]."),
    # --- their correct counterparts
    ("MSFT total revenue growth", True, ["MSFT_FY2026_Item7_007"],
     "Microsoft's revenue increased $50.1 billion or 18% in fiscal 2026 [1]."),
    ("AMZN gross purchases (no year header in chunk)", False, ["AMZN_FY2025_Item8_006"],
     "Amazon's purchases of property and equipment were $131,819 million in 2025 [1]."),
    ("AMZN net capex, labeled correctly", True, ["AMZN_FY2025_Item7_028"],
     "Amazon's purchases of property and equipment, net of proceeds from sales and incentives, were $128.3 billion in 2025 [1]."),
    ("TSLA total margin (no year header in chunk)", False, ["TSLA_FY2025_Item7_025"],
     "Tesla's total gross margin in fiscal 2025 was 18.0% [1]."),
    ("TSLA segment margin, labeled (no year header)", False, ["TSLA_FY2025_Item7_025"],
     "In fiscal 2025 the gross margin of Tesla's automotive & services and other segment was 16.2% [1]."),
    ("AMZN growth derived from table", True, ["AMZN_FY2025_Item7_015"],
     "Amazon's net sales grew about 12.4% in 2025, from $638.0 billion to $716.9 billion [1]."),
    # --- other error types
    ("AAPL wrong year", False, ["AAPL_FY2025_Item8_049"],
     "Apple's total net sales in fiscal 2025 were $391,035 million [1]."),
    ("AAPL correct", True, ["AAPL_FY2025_Item8_049"],
     "Apple's total net sales in fiscal 2025 were $416,161 million [1]."),
    ("AAPL hallucinated number", False, ["AAPL_FY2025_Item8_049"],
     "Apple's total net sales in fiscal 2025 were $420,500 million [1]."),
    ("AAPL unit error (billion vs million)", False, ["AAPL_FY2025_Item8_049"],
     "Apple's total net sales in fiscal 2025 were $416,161 billion [1]."),
    ("TSLA Musk: faithful paraphrase", True, ["TSLA_FY2025_Item1A_024"],
     "Tesla says it is highly dependent on Elon Musk, who does not devote his full time and attention to Tesla [1]."),
    ("TSLA Musk: contradicted", False, ["TSLA_FY2025_Item1A_024"],
     "Tesla says Elon Musk devotes his full time and attention to Tesla [1]."),
]


def main() -> None:
    by_id = {r["id"]: r for r in load_rows()}
    checkers = {
        "strict numeric": lambda a, p: (not check_answer(a, p)["passed"], ""),
        "tolerant numeric": lambda a, p: (not check_answer(a, p, tolerant=True)["passed"], ""),
    }
    usage_by = {}

    def llm(model, judge=judge_faithfulness):
        def run(a, p):
            res, u = judge(a, p, model=model)
            usage_by.setdefault(model, []).append(u)
            bad = [c for c in res["claims"] if c["verdict"] != "supported"]
            return bool(bad), (bad[0]["check"][:150] if bad else "")
        return run

    checkers["judge v1 (4o-mini)"] = llm("gpt-4o-mini")
    checkers[f"judge v1 ({JUDGE_MODEL})"] = llm(JUDGE_MODEL)
    checkers["judge v2 (4o-mini)"] = llm("gpt-4o-mini", judge_faithfulness_v2)
    checkers[f"judge v2 ({JUDGE_MODEL})"] = llm(JUDGE_MODEL, judge_faithfulness_v2)

    results = {name: [] for name in checkers}
    print(f"{'case':46s} {'truth':>9s} " + " ".join(f"{n[:16]:>16s}" for n in checkers))
    for label, faithful, ids, answer in CASES:
        passages = [by_id[i]["text"] for i in ids]
        cells = []
        for name, fn in checkers.items():
            t = time.perf_counter()
            flagged, why = fn(answer, passages)
            results[name].append((faithful, flagged, time.perf_counter() - t, why))
            ok = flagged != faithful
            cells.append(f"{'✅' if ok else '❌'} {'flag' if flagged else 'pass'}")
        print(f"{label:46s} {'faithful' if faithful else 'NOT':>9s} " + " ".join(f"{c:>16s}" for c in cells))

    print(f"\n{'checker':22s} {'accuracy':>9s} {'catches wrong':>14s} {'false alarms':>13s} {'ms/check':>9s}")
    for name, rs in results.items():
        acc = sum(f != fl for f, fl, _, _ in rs)
        caught = sum(fl for f, fl, _, _ in rs if not f)
        alarms = sum(fl for f, fl, _, _ in rs if f)
        n_bad, n_good = sum(not f for f, *_ in rs), sum(f for f, *_ in rs)
        ms = 1000 * sum(t for *_, t, _ in rs) / len(rs)
        print(f"{name:22s} {acc:>5}/{len(rs)} {caught:>9}/{n_bad} {alarms:>9}/{n_good} {ms:>9.0f}")
    print("(LLM timings are ~0 ms on reruns: judge outputs are cached. First run: ~1-3 s per check.)")

    for model, us in usage_by.items():
        tin, tout = sum(u["input_tokens"] for u in us), sum(u["output_tokens"] for u in us)
        if tin:
            print(f"{model}: {tin} in / {tout} out tokens for {len(us)} checks = ${cost_usd(model, tin, tout):.4f}")

    print(f"\nWhy judge v2 ({JUDGE_MODEL}) flagged the unfaithful cases:")
    for (label, faithful, *_), (_, flagged, _, why) in zip(CASES, results[f"judge v2 ({JUDGE_MODEL})"]):
        if not faithful and flagged and why:
            print(f"  {label}: {why}...")


if __name__ == "__main__":
    main()
