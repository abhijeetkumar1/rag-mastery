"""Query rewriting and multi-query (RAG-Fusion): change the QUERY, keep the retriever (Phase 2) fixed.

Run: uv run python -m phase3_query_intelligence.01_rewrite

Strategies, all on top of Phase 2's hybrid + rerank retriever, scored on the 12 Phase 2 probes:
  original                    the user's question as-is (Phase 2 baseline)
  generic rewrite (rr=rewrite) LLM rewrite, no domain hints, reranked against the REWRITE (a common mistake)
  generic rewrite (rr=orig)    same rewrite, reranked against the user's ORIGINAL question
  domain rewrite              LLM rewrite + 10-K vocabulary hints (common.query.VOCAB), rerank vs original
  multi-query     3 LLM phrasings + the original, each retrieved, RRF-fused, reranked against the ORIGINAL
Every LLM call is cached (.cache/chat_json/), so reruns cost nothing.
"""
from common.query import multi_query, rewrite
from common.retriever import Retriever
from phase3_query_intelligence.cases import run_probes


def main() -> None:
    R = Retriever()
    R.warmup()
    q = "How much did Apple spend on share buybacks in fiscal 2025?"
    print(f"Example: {q}")
    print(f"  generic rewrite: {rewrite(q, domain=False)[0]!r}")
    print(f"  domain rewrite : {rewrite(q)[0]!r}")
    print(f"  multi-query    : {multi_query(q)[0]}\n")

    def ids(hits):
        return [h["id"] for h in hits]

    strategies = {
        # k=50 returns the whole reranked pool, so hit@50 = "was it in the candidates at all?"
        "original": lambda q: (ids(R.retrieve(q, k=50)), {}),
        "generic (rr=rewrite)": lambda q: (lambda rq, u: (ids(R.retrieve(rq, k=50)), u))(*rewrite(q, domain=False)),
        # below, the reranker judges against the USER's question: the rewrite only changes what gets retrieved
        "generic (rr=orig)": lambda q: (lambda rq, u: (ids(R.retrieve_multi([rq], q, k=50)), u))(*rewrite(q, domain=False)),
        "domain rewrite": lambda q: (lambda rq, u: (ids(R.retrieve_multi([rq], q, k=50)), u))(*rewrite(q)),
        "multi-query (3+orig)": lambda q: (lambda qs, u: (ids(R.retrieve_multi([q, *qs], q, k=50)), u))(*multi_query(q)),
    }
    print(f"{'strategy':22s} {'hit@5':>6} {'MRR@5':>6} {'hit@50':>7}  keyword/semantic/numeric  LLM tokens*  misses@5")
    for name, fn in strategies.items():
        r = run_probes(fn, R.rows)
        k = r["kinds"]
        print(f"{name:22s} {r['hit@5']:>4}/12 {r['mrr@5']:6.2f} {r['hit@50']:>5}/12   {k['keyword']}/4   {k['semantic']}/4   {k['numeric']}/4 "
              f"{r['tokens']:>10}  {[m[:28] for m in r['misses']]}")
    print("\n* LLM tokens spent on THIS run; 0 = every call served from .cache/chat_json (first run: ~1.0k generic,")
    print("  ~2.3k domain, ~3.1k multi-query for the 12 probes, i.e. ~85-255 tokens per question).")
    print("The probes informed the vocabulary hints, so domain gains are optimistic (not a held-out test).")


if __name__ == "__main__":
    main()
