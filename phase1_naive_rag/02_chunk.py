"""Chunk the parsed filings and compare strategies.

Run: uv run python -m phase1_naive_rag.02_chunk                      # recursive, 350 tokens (default)
     uv run python -m phase1_naive_rag.02_chunk --chunker fixed --size 200

Chunks never cross an Item boundary, and each inherits metadata (ticker, year, section) that
Phase 2 uses for filtering. Output: data/processed/chunks_<chunker><size>.jsonl
"""
import argparse
import json
import statistics

from common.chunking import CHUNKERS, count_tokens
from common.config import DATA_DIR, EMBED_MODEL, EMBED_PROVIDER

PROC_DIR = DATA_DIR / "processed"


def chunk_corpus(chunker: str, size: int, overlap: int) -> list[dict]:
    fn = CHUNKERS[chunker]
    rows = []
    for path in sorted(PROC_DIR.glob("*_FY*.json")):
        doc = json.loads(path.read_text())
        for sec in doc["sections"]:
            for i, text in enumerate(fn(sec["text"], size, overlap)):
                rows.append({
                    "id": f"{doc['ticker']}_FY{doc['fiscal_year']}_{sec['item'].replace(' ', '')}_{i:03d}",
                    "text": text,
                    "ticker": doc["ticker"],
                    "fiscal_year": doc["fiscal_year"],
                    "item": sec["item"],
                    "section": sec["title"][:120],
                    "url": doc["url"],
                })
    return rows


def embedder_max_tokens() -> tuple[int, object] | None:
    """The embedder has its OWN tokenizer and limit. Text past it is silently truncated."""
    if EMBED_PROVIDER != "hf":
        return None  # text-embedding-3: 8191 tokens, far above our chunk sizes
    from common.llm import _hf_model
    m = _hf_model(EMBED_MODEL)
    return m.max_seq_length, m.tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", choices=list(CHUNKERS), default="recursive")
    ap.add_argument("--size", type=int, default=350)
    ap.add_argument("--overlap", type=int, default=50)
    args = ap.parse_args()

    rows = chunk_corpus(args.chunker, args.size, args.overlap)
    toks = [count_tokens(r["text"]) for r in rows]
    print(f"{args.chunker} size={args.size} overlap={args.overlap}: {len(rows)} chunks, "
          f"tokens mean={statistics.mean(toks):.0f} median={statistics.median(toks):.0f} "
          f"min={min(toks)} max={max(toks)}, total={sum(toks) / 1e6:.2f}M tokens")

    if (lim := embedder_max_tokens()) is not None:
        max_len, tok = lim
        over = sum(len(tok(r["text"])["input_ids"]) > max_len for r in rows)
        print(f"embedder limit {max_len} tokens ({EMBED_MODEL}): {over} chunks would be TRUNCATED")

    # Show what a boundary looks like: the end of one chunk and the start of the next
    a, b = rows[40], rows[41]
    print(f"\n--- end of {a['id']} ---\n...{a['text'][-250:]}")
    print(f"--- start of {b['id']} ---\n{b['text'][:250]}...")

    out = PROC_DIR / f"chunks_{args.chunker}{args.size}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"\nwrote {out.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
