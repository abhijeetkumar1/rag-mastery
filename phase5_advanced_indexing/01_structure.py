"""Parse the 10-Ks again, this time KEEPING STRUCTURE: every section becomes a list of blocks (paragraphs and tables,
with each table's caption and header rows identified). Output: data/processed/structured/<TICKER>_FY<year>.json

Run: uv run python -m phase5_advanced_indexing.01_structure
     uv run python -m phase5_advanced_indexing.01_structure --show TSLA_FY2025 --item "Item 7"   # print one section's tables

Phase 1's parser is reused unchanged except for one switch (html_to_text(mark_tables=True)), so the TEXT is the
same as Phase 1's, character for character; only the table boundaries are new. That keeps the golden-set evidence
quotes valid for the new chunkings (02_chunk re-derives the relevance labels from them).
"""
import argparse
import importlib
import json

from common.config import DATA_DIR
from common.structure import parse_blocks

parse1 = importlib.import_module("phase1_naive_rag.01_parse")
OUT_DIR = DATA_DIR / "processed" / "structured"


def structure_filing(doc: dict) -> dict:
    html = (DATA_DIR / "raw" / doc["file"]).read_text(encoding="utf-8", errors="ignore")
    sections = parse1.split_sections(parse1.html_to_text(html, mark_tables=True))
    out = {k: doc[k] for k in ("ticker", "fiscal_year", "filing_date", "period_end", "url")}
    out["sections"] = [{"item": s["item"], "title": s["title"], "blocks": parse_blocks(s["text"])} for s in sections]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="e.g. TSLA_FY2025: print the tables of one filing")
    ap.add_argument("--item", default="Item 7")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA_DIR / "raw" / "manifest.json").read_text())

    print(f"{'filing':12s} {'blocks':>7s} {'tables':>7s} {'w/ header':>10s} {'w/ caption':>11s} {'rows':>6s}  no-header examples")
    totals = {"tables": 0, "header": 0, "caption": 0}
    for doc in sorted(manifest, key=lambda d: d["file"]):
        s = structure_filing(doc)
        (OUT_DIR / doc["file"].replace(".html", ".json")).write_text(json.dumps(s, indent=1))
        blocks = [b for sec in s["sections"] for b in sec["blocks"]]
        tables = [b for b in blocks if b["kind"] == "table"]
        with_h = [t for t in tables if t["header"]]
        with_c = [t for t in tables if t["caption"]]
        no_h = [t["rows"][0][:40] for t in tables if not t["header"]][:2]
        totals["tables"] += len(tables)
        totals["header"] += len(with_h)
        totals["caption"] += len(with_c)
        print(f"{doc['file'][:-5]:12s} {len(blocks):>7} {len(tables):>7} {len(with_h):>10} {len(with_c):>11} "
              f"{sum(len(t['rows']) for t in tables):>6}  {no_h}")
    print(f"\n{totals['tables']} data tables: header detected in {totals['header']} "
          f"({totals['header'] / totals['tables']:.0%}), caption in {totals['caption']} ({totals['caption'] / totals['tables']:.0%})")
    print(f"wrote {OUT_DIR.relative_to(DATA_DIR.parent)}/")

    if args.show:
        s = json.loads((OUT_DIR / f"{args.show}.json").read_text())
        for sec in s["sections"]:
            if sec["item"] != args.item:
                continue
            for t in (b for b in sec["blocks"] if b["kind"] == "table"):
                print(f"\n--- caption: {t['caption']!r}  ({len(t['rows'])} body rows)")
                for h in t["header"]:
                    print(f"  HEADER | {h}")
                for r in t["rows"][:3]:
                    print(f"  row    | {r}")


if __name__ == "__main__":
    main()
