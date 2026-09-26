"""Embed chunks and load them into a persistent Chroma collection, then try a few raw searches.

Run: uv run python -m phase1_naive_rag.03_index                    # indexes chunks_recursive350
     uv run python -m phase1_naive_rag.03_index --chunker fixed --size 350

Re-running is cheap: embeddings come from the disk cache, only Chroma is rebuilt.
"""
import argparse
import json
import time

from common.config import DATA_DIR, EMBED_MODEL
from common.store import add_chunks, collection_name, get_collection, search

QUERIES = [
    "What was Apple's total net sales in fiscal 2025?",
    "How does NVIDIA describe export control risks for China?",
    "What are the risks of Tesla's reliance on Elon Musk?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", default="recursive")
    ap.add_argument("--size", type=int, default=350)
    args = ap.parse_args()

    path = DATA_DIR / "processed" / f"chunks_{args.chunker}{args.size}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    name = collection_name(args.chunker, args.size)
    col = get_collection(name, reset=True)

    t0 = time.time()
    add_chunks(
        col,
        ids=[r["id"] for r in rows],
        texts=[r["text"] for r in rows],
        metadatas=[{k: r[k] for k in ("ticker", "fiscal_year", "item", "section", "url")} for r in rows],
    )
    print(f"indexed {col.count()} chunks into '{name}' with {EMBED_MODEL} in {time.time() - t0:.1f}s\n")

    for q in QUERIES:
        print(f"Q: {q}")
        for h in search(col, q, k=3):
            m = h["meta"]
            snippet = h["text"][:110].replace("\n", " ")
            print(f"  {h['score']:.3f}  {m['ticker']} FY{m['fiscal_year']} {m['item']:9s} {snippet}")
        print()


if __name__ == "__main__":
    main()
