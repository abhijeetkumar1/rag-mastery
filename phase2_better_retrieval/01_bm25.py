"""BM25 from scratch vs dense retrieval: where each one wins.

Run: uv run python -m phase2_better_retrieval.01_bm25
     uv run python -m phase2_better_retrieval.01_bm25 "your query"

Shows (1) BM25's internals on one query (tokens, idf, per-term contribution), then
(2) BM25 vs dense top-3 side by side on keyword and paraphrase queries.
"""
import sys
import time

from common.bm25 import BM25Index, tokenize
from common.store import collection_name, get_collection, search
from phase2_better_retrieval.probes import load_rows

QUERIES = [
    ("Vision Pro", "product name"),
    ("Dojo supercomputer", "rare name NOT in the corpus"),
    ("Item 1C cybersecurity", "keywords that also match the table of contents"),
    ("Can the company lose key people?", "paraphrase, few shared words"),
    ("What are the risks of depending on a single supplier for chips?", "paraphrase"),
]


def explain(bm: BM25Index, rows: list[dict], query: str) -> None:
    terms = list(dict.fromkeys(tokenize(query)))
    print(f"query tokens: {terms}")
    for t in terms:
        df = len(bm.postings.get(t, []))
        print(f"  {t:12s} df={df:5d}/{bm.n}  idf={bm.idf.get(t, 0):5.2f}" + ("   (not in vocabulary)" if not df else ""))
    (i, s), = bm.search(query, 1)
    print(f"top-1: {rows[i]['id']}  score={s:.2f}  (doc length {int(bm.doc_len[i])} tokens, avg {bm.avgdl:.0f})")
    for t in terms:
        part = BM25Index.__new__(BM25Index)  # score one term at a time to show each term's contribution
        part.__dict__ = bm.__dict__
        print(f"  contribution of {t!r}: {part.scores(t)[i]:.2f}")


def main() -> None:
    rows = load_rows()
    t = time.perf_counter()
    bm = BM25Index([r["text"] for r in rows])
    print(f"BM25 index: {bm.n} chunks, vocabulary {len(bm.postings):,} terms, built in {time.perf_counter() - t:.2f}s\n")

    queries = [(sys.argv[1], "your query")] if len(sys.argv) > 1 else QUERIES
    explain(bm, rows, queries[0][0])

    col = get_collection(collection_name("recursive", 350))
    for q, why in queries:
        print(f"\nQ: {q}   [{why}]")
        b = [f"{rows[i]['id']} ({s:.1f})" for i, s in bm.search(q, 3)]
        d = [f"{h['id']} ({h['score']:.3f})" for h in search(col, q, 3)]
        print(f"  bm25 : {', '.join(b) or '(no chunk shares a term with the query)'}")
        print(f"  dense: {', '.join(d)}")


if __name__ == "__main__":
    main()
