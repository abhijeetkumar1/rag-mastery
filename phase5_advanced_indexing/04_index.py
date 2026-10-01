"""Embed and index every Phase 5 chunk set into its own Chroma collection, then compare one search across them.

Run: uv run python -m phase5_advanced_indexing.04_index                 # all chunk sets that exist
     uv run python -m phase5_advanced_indexing.04_index --only structctx350

One collection per chunk set (common.store.collection_name): vectors of different chunkings must never mix, for the
same reason vectors of different models must not. parent800 is NOT indexed: parents are looked up by id after the
children are retrieved (common.parent_child). `kind` (text|table) is stored as metadata too, so a retriever can be
restricted to tables (Phase 6's search_tables tool: filters={"kind": ["table"]}). Embeddings are disk-cached per
text, so re-indexing is cheap, but any change to the header format changes every text and re-embeds everything
(~$0.02 per chunk set with OpenAI).
"""
import argparse
import json
import re
import time

from common.config import DATA_DIR, EMBED_MODEL
from common.store import add_chunks, collection_name, get_collection, search

SETS = ["structured350", "structctx350", "child200", "llmctx350"]
QUERY = "What was Tesla's total gross margin in 2025?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    args = ap.parse_args()
    for name in [args.only] if args.only else SETS:
        path = DATA_DIR / "processed" / f"chunks_{name}.jsonl"
        if not path.exists():
            print(f"skip {name}: {path.name} missing (run 02_chunk / 03_contextual)")
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        chunker, size = re.fullmatch(r"([a-z]+)(\d+)", name).groups()
        col = get_collection(collection_name(chunker, int(size)), reset=True)
        t0 = time.time()
        add_chunks(col, [r["id"] for r in rows], [r["text"] for r in rows],
                   [{k: r[k] for k in ("ticker", "fiscal_year", "item", "section", "url", "kind") if k in r} for r in rows])
        print(f"\n{name}: indexed {col.count()} chunks into '{col.name}' with {EMBED_MODEL} in {time.time() - t0:.1f}s")
        print(f"  Q: {QUERY}")
        for h in search(col, QUERY, k=3):
            snippet = h["text"].split("\n", 1)[-1][:90].replace("\n", " ")  # skip the header line
            print(f"   {h['score']:.3f}  {h['id']:26s} {snippet}")


if __name__ == "__main__":
    main()
