"""Query decomposition: split a multi-company question into per-company sub-questions, each retrieved
with its own filter (ticker + filing) and in its own company's vocabulary.

Run: uv run python -m phase3_query_intelligence.04_decompose

1. The plan for the comparison question (sub-questions, filters, policy overrides)
2. Entity coverage on 4 comparison probes: for how many companies is the evidence in the top k?
     single query     Phase 2 hybrid + rerank, no filters
     fan-out (rules)  Phase 2: regex filters, same question text per company
     decomposition    Phase 3: LLM sub-questions + per-company filing filters + vocabulary expansion
3. End to end: the MSFT vs AMZN answer, which Phase 2 got wrong twice (segment figure; guidance)
"""
from common.filters import extract_filters
from common.guardrails import check_answer
from common.llm import chat_completion
from common.query import plan
from common.retriever import Retriever
from phase3_query_intelligence.cases import COMPARISON_PROBES, comparison_relevant
import importlib

phase1 = importlib.import_module("phase1_naive_rag.04_ask")


def decompose_retrieve(R: Retriever, question: str, k_each: int = 3, trace=None) -> tuple[list[dict], dict]:
    p, _, _ = plan(question)
    hits = []
    for s in p["sub_questions"]:
        f = {}
        if s["ticker"]:
            f["ticker"] = [s["ticker"]]
        if s["filing_years"]:
            f["fiscal_year"] = s["filing_years"]
        # retrieve with the sub-question; rerank against it too: it IS the user's intent for that company
        for h in R.retrieve(s["question"], k=k_each, filters=f or None, trace=trace):
            h["sub_question"] = s["question"]
            hits.append(h)
    return hits, p


def coverage(hit_ids: list[str], rel: dict[str, set[str]]) -> int:
    return sum(bool(set(hit_ids) & ids) for ids in rel.values())


def main() -> None:
    R = Retriever()
    R.warmup()
    q = COMPARISON_PROBES[0][0]
    p, usage, overrides = plan(q)
    print(f"1. {q}")
    for s in p["sub_questions"]:
        print(f"   [{s['ticker']}] filings={s['filing_years']}  {s['question']}")
    print(f"   policy overrides: {overrides}\n")

    rels = comparison_relevant(R.rows)
    print(f"2. Entity coverage (companies whose evidence is in the top 6)")
    print(f"   {'question':62s} {'single':>7} {'fan-out':>8} {'decomp':>7}")
    tot = [0, 0, 0]
    for (cq, per), rel in zip(COMPARISON_PROBES, rels):
        single = [h["id"] for h in R.retrieve(cq, k=6)]
        tickers = extract_filters(cq).get("ticker", [])
        fan = [h["id"] for h in R.retrieve_fanout(cq, "ticker", tickers, k_each=3)] if len(tickers) > 1 else single
        dec = [h["id"] for h in decompose_retrieve(R, cq, k_each=3)[0]]
        c = [coverage(x, rel) for x in (single, fan, dec)]
        tot = [a + b for a, b in zip(tot, c)]
        n = len(per)
        print(f"   {cq[:62]:62s} {c[0]:>5}/{n} {c[1]:>6}/{n} {c[2]:>5}/{n}")
    total = sum(len(per) for _, per in COMPARISON_PROBES)
    print(f"   {'TOTAL':62s} {tot[0]:>5}/{total} {tot[1]:>6}/{total} {tot[2]:>5}/{total}\n")

    print(f"3. End to end: {q}")
    hits, _ = decompose_retrieve(R, q, k_each=3)
    messages = [{"role": "system", "content": phase1.SYSTEM},
                {"role": "user", "content": f"Context:\n\n{phase1.format_context(hits)}\n\nQuestion: {q}"}]
    answer = chat_completion(messages).choices[0].message.content
    print(answer + "\n")
    for n, h in enumerate(hits, 1):
        print(f"   [{n}] {h['score']:6.2f} {h['id']:24s} <- {h['sub_question'][:70]}")
    print("   ground truth: Microsoft FY2026 revenue +$50.1B / +18%; Amazon FY2025 net sales 637,959 -> 716,924 (+12%)\n")

    texts = [h["text"] for h in hits]
    strict, tolerant = check_answer(answer, texts), check_answer(answer, texts, tolerant=True)
    print(f"4. Numeric guardrail on this answer\n   strict  : passed={strict['passed']}  ungrounded={strict['ungrounded_numbers']}")
    print(f"   tolerant: passed={tolerant['passed']}  ungrounded={tolerant['ungrounded_numbers']}")
    for num, how in tolerant.get("derived_numbers", {}).items():
        print(f"             {num:>6} = {how}")
    # tamper test: a tolerant check is only useful if it still catches WRONG numbers
    for good, bad in [("12.4", "15.3"), ("716.9", "736.9")]:
        if good in answer:
            t = check_answer(answer.replace(good, bad), texts, tolerant=True)
            print(f"   tamper {good} -> {bad}: passed={t['passed']}  ungrounded={t['ungrounded_numbers']}")


if __name__ == "__main__":
    main()
