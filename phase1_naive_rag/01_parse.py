"""Parse 10-K HTML into clean text split by SEC "Item" sections.

Run: uv run python -m phase1_naive_rag.01_parse

Garbage in, garbage out: most RAG quality problems start here, not in the vector DB.
What we fight in these files:
  - <ix:header>: hidden inline-XBRL facts (thousands of machine tags nobody should retrieve)
  - display:none blocks
  - headings split across <span>s ("ITEM 1. B" + "USINESS") -> join inline text, break only on blocks
  - a table of contents + running page headers that LOOK like section headings
  - financial tables -> flatten each row to "cell | cell | cell" so numbers stay next to their labels
"""
import json
import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from common.config import DATA_DIR
from common.structure import TABLE_END, TABLE_START  # Phase 5: sentinel lines around each <table>

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"

BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"]
# "Item 7." / "ITEM 7A. MANAGEMENT'S..." at line start, followed by a title
ITEM_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?)\s*[.:\-–—]?\s*(.*)$", re.I)
# Page furniture repeated on every page: page numbers, "Table of Contents" back-links,
# bare running headers ("PART II", "Item 8"), separator rules, and page footers like
# "Apple Inc. | 2025 Form 10-K | 48". Adds noise to chunks and vectors.
NOISE_RE = re.compile(
    r"\d{1,3}|table of contents|part\s+i{1,3}v?|item\s+\d{1,2}[a-c]?|[_\-=]{3,}"
    r"|[^|]{2,60}\|\s*\d{4}\s+form\s+10-k\s*\|\s*\d{1,3}",
    re.I,
)


def html_to_text(html: str, mark_tables: bool = False) -> str:
    """mark_tables (Phase 5): wrap every table in TABLE_START/TABLE_END lines, so a structure-aware chunker
    knows where a table begins (its header rows) and ends. Off by default: Phase 1's output is unchanged."""
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all(["ix:header", "script", "style"]):
        t.decompose()
    for t in soup.select('[style*="display:none"], [style*="display: none"]'):
        t.decompose()

    if mark_tables:
        for t in soup.find_all("table"):
            t.insert_before(f"\n\n{TABLE_START}\n")
            t.insert_after(f"\n{TABLE_END}\n\n")
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in {"$", "%", ")"}]  # drop split-off currency/paren cells
        tr.replace_with(soup.new_string(" | ".join(cells) + "\n" if cells else ""))
    for t in soup.find_all(BLOCK_TAGS):
        t.insert_before("\n\n")
        t.insert_after("\n\n")
    for br in soup.find_all("br"):
        br.replace_with("\n")

    text = soup.get_text("").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = "\n".join(ln for ln in text.split("\n") if not NOISE_RE.fullmatch(ln.strip()))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^([•●▪])\n+", r"\1 ", text, flags=re.M)  # re-attach bullets split from their text
    return text.strip()


def split_sections(text: str) -> list[dict]:
    """Walk lines; a heading is a short line 'Item N. Title'.

    Some filers (AMZN) lay headings out as a table row 'Item 1. | Business'. The table of
    contents looks identical except for a trailing page number: 'Item 1. | Business | 3'.
    """
    sections = [{"item": "Cover", "title": "Cover page", "lines": []}]
    for line in text.split("\n"):
        cells = [c.strip() for c in line.split("|")]
        is_toc_row = len(cells) > 1 and cells[-1].isdigit()
        m = None if is_toc_row or len(cells) > 2 else ITEM_RE.match(" ".join(cells))
        title = m.group(2).strip() if m else ""
        if m and len(line) < 200 and len(title) >= 4 and not title[0].islower():
            sections.append({"item": f"Item {m.group(1).upper()}", "title": title, "lines": []})
        else:
            sections[-1]["lines"].append(line)
    out = []
    for s in sections:
        body = "\n".join(s["lines"]).strip()
        if len(body) > 200:  # skip empty headings, e.g. "Item 6. [Reserved]"
            out.append({"item": s["item"], "title": s["title"], "text": body})
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RAW_DIR / "manifest.json").read_text())
    for doc in manifest:
        html = (RAW_DIR / doc["file"]).read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(html)
        sections = split_sections(text)
        out = {k: doc[k] for k in ("ticker", "fiscal_year", "filing_date", "period_end", "url")}
        out["sections"] = sections
        (OUT_DIR / doc["file"].replace(".html", ".json")).write_text(json.dumps(out, indent=1))
        items = ", ".join(s["item"].removeprefix("Item ") for s in sections)
        print(f"{doc['file']:18s} html={len(html) / 1e6:4.1f}MB -> text={len(text) / 1e3:4.0f}K chars  sections: {items}")


if __name__ == "__main__":
    main()
