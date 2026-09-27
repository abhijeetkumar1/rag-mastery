"""HyDE (Hypothetical Document Embeddings): search with a FAKE answer passage instead of the question.

Run: uv run python -m phase3_query_intelligence.02_hyde

Why: a short question and a long 10-K passage look different in embedding space (query/document
asymmetry, Phase 0). An LLM-written passage that "would answer" the question looks like the real passage:
same style, same vocabulary, so it lands near it. Its facts may be wrong; only its shape matters.

1. One example: the question, its HyDE passage, dense top-3 for each
2. Dense-only comparison on the probes (the classic HyDE setting): question vs HyDE passage
3. Full pipeline (hybrid + rerank): HyDE on the dense side vs the plain question
"""
from common.query import hyde
from common.retriever import Retriever
from phase3_query_intelligence.cases import run_probes


def main() -> None:
    R = Retriever()
    R.warmup()
    q = "Is Tesla too reliant on its CEO?"
    passage, _ = hyde(q)
    print(f"1. {q}\n   HyDE passage: {passage[:300]}...\n")
    print("   dense(question):", [i for i, _ in R.dense(q, 3, None)])
    print("   dense(HyDE)    :", [i for i, _ in R.dense(passage, 3, None, as_document=True)])
    print("   (relevant: TSLA_*_Item1A_02x chunks with 'highly dependent on the services of Elon Musk')\n")

    def ids(hits):
        return [h["id"] for h in hits]

    strategies = {
        "dense(question)": lambda q: ([i for i, _ in R.dense(q, 50, None)], {}),
        "dense(HyDE)": lambda q: (lambda p, u: ([i for i, _ in R.dense(p, 50, None, as_document=True)], u))(*hyde(q)),
        "hybrid+rerank (question)": lambda q: (ids(R.retrieve(q, k=50)), {}),
        "hybrid+rerank (HyDE dense)": lambda q: (lambda p, u: (ids(R.retrieve(q, k=50, hyde_passage=p)), u))(*hyde(q)),
    }
    print(f"2-3. {'strategy':28s} {'hit@5':>6} {'MRR@5':>6} {'hit@50':>7}  keyword/semantic/numeric  misses@5")
    for name, fn in strategies.items():
        r = run_probes(fn, R.rows)
        k = r["kinds"]
        print(f"     {name:28s} {r['hit@5']:>4}/12 {r['mrr@5']:6.2f} {r['hit@50']:>5}/12   {k['keyword']}/4   {k['semantic']}/4"
              f"   {k['numeric']}/4   {[m[:28] for m in r['misses']]}")

    print("\n4. The risk: HyDE passages invent facts. For numeric questions they invent NUMBERS and DATES:")
    for q in ["What was Apple's total net sales in fiscal 2025?", "How big is Amazon's workforce?"]:
        print(f"   {q}\n     -> {hyde(q)[0][:220]}...")
    print("   The fake passage is only a search key and is never shown to the user or the generator,")
    print("   but a made-up year or figure can pull the search toward the wrong filing or table.")


if __name__ == "__main__":
    main()
