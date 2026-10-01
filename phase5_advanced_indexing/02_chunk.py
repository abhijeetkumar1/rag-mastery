"""Build the Phase 5 chunk sets from the structured parse (01_structure) and compare them with Phase 1's chunks.

Run: uv run python -m phase5_advanced_indexing.02_chunk
     uv run python -m phase5_advanced_indexing.02_chunk --show TSLA_FY2025 --grep "Total gross margin"

Writes data/processed/chunks_<name>.jsonl, one per index variant:
  structured350  table-aware chunking (tables split between rows, header repeated), text = body only
  structctx350   the same chunks with a contextual header line prepended  ("[Tesla, Inc. (TSLA) | Form 10-K ...]")
  parent800      parent chunks (table-aware, 800 tokens, with header): what the LLM reads
  child200       children of each parent (200 tokens, with header, parent_id): what gets searched

Every row: id, text (what is embedded, BM25-indexed and shown to the LLM), body (the filing's own text, used to
re-derive relevance labels), ctx_header, kind (text|table), caption, ticker, fiscal_year, item, section, url.
"""
import argparse
import json
import re
import statistics

from common.chunking import count_tokens, recursive_chunks, structured_chunks, table_pieces
from common.config import DATA_DIR
from common.contextual import context_header
from common.structure import render_table

PROC = DATA_DIR / "processed"
NUMERIC_ROW = re.compile(r"\|[^|\n]*\d[\d,]*")  # a table row with a number in a cell after the label


def table_headers() -> dict[tuple, list[tuple[str, ...]]]:
    """(ticker, fiscal_year, row text) -> the header(s) of the source table(s) that row belongs to, from the
    structured parse. Lets us ask of ANY chunk set: does a chunk with table rows also contain their header?"""
    out: dict[tuple, list[tuple[str, ...]]] = {}
    for path in (PROC / "structured").glob("*_FY*.json"):
        doc = json.loads(path.read_text())
        for sec in doc["sections"]:
            for b in sec["blocks"]:
                if b["kind"] == "table" and b["header"]:
                    for r in b["rows"]:
                        out.setdefault((doc["ticker"], doc["fiscal_year"], r), []).append(tuple(b["header"]))
    return out


def lost_header(r: dict, headers: dict) -> bool:
    """True if the chunk holds a numeric row of a table that HAS a header row, but none of that table's possible
    headers is in the chunk. The failure Phase 4's judge flagged: "Total gross margin | 18.0" is there, the
    "(Dollars in millions) | 2025 | 2024 | 2023" line that says it's 2025 is in the previous chunk."""
    lines = set(r.get("body", r["text"]).split("\n"))
    for ln in lines:
        cands = headers.get((r["ticker"], r["fiscal_year"], ln))
        if cands and NUMERIC_ROW.search(ln) and not any(set(h) <= lines for h in cands):
            return True
    return False


def row(doc: dict, sec: dict, i: int, kind: str, body: str, caption: str, header: bool) -> dict:
    h = context_header(doc["ticker"], doc["fiscal_year"], doc["period_end"], sec["item"])  # caption: already in the body
    return {"id": f"{doc['ticker']}_FY{doc['fiscal_year']}_{sec['item'].replace(' ', '')}_{i:03d}",
            "text": f"{h}\n{body}" if header else body, "body": body, "ctx_header": h, "kind": kind,
            "caption": caption, "ticker": doc["ticker"], "fiscal_year": doc["fiscal_year"], "item": sec["item"],
            "section": sec["title"][:120], "url": doc["url"]}


def build(size: int = 350, overlap: int = 50) -> dict[str, list[dict]]:
    out = {"structured350": [], "structctx350": [], "parent800": [], "child200": []}
    for path in sorted((PROC / "structured").glob("*_FY*.json")):
        doc = json.loads(path.read_text())
        for sec in doc["sections"]:
            for i, c in enumerate(structured_chunks(sec["blocks"], size, overlap)):
                out["structured350"].append(row(doc, sec, i, c["kind"], c["body"], c["caption"], header=False))
                out["structctx350"].append(row(doc, sec, i, c["kind"], c["body"], c["caption"], header=True))
            for i, p in enumerate(structured_chunks(sec["blocks"], 800, 100)):
                parent = row(doc, sec, i, p["kind"], p["body"], p["caption"], header=True)
                out["parent800"].append(parent)
                # children re-split the PARENT, so every child lies inside exactly one parent
                bodies = ([render_table(p["caption"], p["header"], rows) for rows in table_pieces(p["header"], p["rows"], 200, p["caption"])]
                          if p["kind"] == "table" else recursive_chunks(p["body"], 200, 30))
                for j, b in enumerate(bodies):
                    child = row(doc, sec, i, p["kind"], b, p["caption"], header=True)
                    child["id"] = f"{parent['id']}_c{j:02d}"
                    child["parent_id"] = parent["id"]
                    out["child200"].append(child)
    return out


def stats(name: str, rows: list[dict], headers: dict) -> None:
    toks = [count_tokens(r["text"]) for r in rows]
    tab = [r for r in rows if NUMERIC_ROW.search(r.get("body", r["text"]))]
    lacking = sum(lost_header(r, headers) for r in tab)
    print(f"{name:16s} {len(rows):>6} {statistics.mean(toks):>6.0f} {max(toks):>5} {sum(toks) / 1e6:>7.2f}M "
          f"{len(tab):>7} {lacking:>8} ({lacking / max(len(tab), 1):.0%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="filing, e.g. TSLA_FY2025: print the chunks containing --grep, old vs new")
    ap.add_argument("--grep", default="Total gross margin")
    args = ap.parse_args()
    variants = build()
    for name, rows in variants.items():
        (PROC / f"chunks_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    old = [json.loads(line) for line in (PROC / "chunks_recursive350.jsonl").read_text().splitlines()]
    headers = table_headers()
    print(f"{'chunk set':16s} {'chunks':>6} {'mean':>6} {'max':>5} {'tokens':>8} {'w/table':>7} {'lost header':>14}")
    stats("recursive350 (P1)", old, headers)
    for name, rows in variants.items():
        stats(name, rows, headers)
    hdr = [count_tokens(r["ctx_header"]) for r in variants["structctx350"]]
    print(f"\ncontextual header: mean {statistics.mean(hdr):.0f} tokens per chunk "
          f"(+{sum(hdr) / sum(count_tokens(r['body']) for r in variants['structctx350']):.0%} tokens to embed)")
    kids = {}
    for c in variants["child200"]:
        kids[c["parent_id"]] = kids.get(c["parent_id"], 0) + 1
    print(f"parent-child: {len(variants['parent800'])} parents, {len(variants['child200'])} children, "
          f"{statistics.mean(kids.values()):.1f} children per parent (max {max(kids.values())})")

    if args.show:
        pat = re.compile(re.escape(args.grep), re.I)
        for label, rows in [("Phase 1 recursive350", old), ("Phase 5 structctx350", variants["structctx350"])]:
            for r in rows:
                if r["id"].startswith(args.show) and pat.search(r["text"]):
                    print(f"\n===== {label}: {r['id']}  (lost header: {lost_header(r, headers)})")
                    print(r["text"][:1200])
    print(f"\nwrote {', '.join(f'chunks_{n}.jsonl' for n in variants)} to {PROC.relative_to(DATA_DIR.parent)}/")


if __name__ == "__main__":
    main()
