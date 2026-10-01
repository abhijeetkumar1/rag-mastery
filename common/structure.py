"""Document structure for Phase 5: a section's text -> a list of BLOCKS (prose paragraphs and tables).

Phase 1 flattened every table row to "cell | cell | cell" and then chunked tables like prose. A 40-row table
was cut wherever the token budget ran out, and every chunk after the first lost the rows that say WHAT the
columns are ("(Dollars in millions) | 2025 | 2024 | 2023"). Phase 4's judge flagged exactly that: the number
is in the chunk, but the chunk can't show which year it belongs to.

Here a table stays a table:
    {"kind": "table", "caption": "Revenues", "header": [rows that label the columns], "rows": [data rows]}
    {"kind": "text",  "text": "one paragraph"}
so the chunker (common.chunking.structured_chunks) can repeat the header in every piece of a split table.

Input: section text produced by phase1_naive_rag.01_parse.html_to_text(mark_tables=True), where every <table>
is wrapped in TABLE_START / TABLE_END lines. Everything here is rules, not ML: cheap and inspectable, and
wrong in ways you can list (see the header-detection gotchas in WALKTHROUGH.md).
"""
import re

TABLE_START, TABLE_END = "⟦TABLE⟧", "⟦/TABLE⟧"

MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
          r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DATE_RE = re.compile(rf"\b{MONTHS}\s+\d{{1,2}},?(?:\s+(?:19|20)\d{{2}})?|\b\d{{1,2}}/\d{{1,2}}/(?:19|20)?\d{{2}}\b", re.I)
# Words that mark a row as describing the COLUMNS rather than holding data
HEADER_WORDS = re.compile(rf"\b(?:{MONTHS}|year|years|ended|months|weeks|quarter|fiscal|in millions|in thousands|"
                          r"in billions|dollars|except|change|% change|as of)\b", re.I)


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.split("|")]


def _cell_is_label(cell: str) -> bool:
    """A cell that names a column: no digits other than dates, years, footnote marks and durations
    ("December 31, 2025", "Jan 25, 2026", "1/31/2021", "2025 vs. 2024", "Average Price (1)", "12 Months")."""
    rest = DATE_RE.sub(" ", cell)
    rest = YEAR_RE.sub(" ", rest)
    rest = re.sub(r"\(\d\)|\b\d+\s+(?:months|weeks|days|years)\b", " ", rest, flags=re.I)
    return not re.search(r"\d", rest)


def _row_is_header(row: str) -> bool:
    return all(_cell_is_label(c) for c in _cells(row))


def _row_names_columns(row: str) -> bool:
    """Header rows carry a year, a date or a unit ("(in millions)"), or are a row of short column names
    ("Name | Age | Position"); a bare "Cost of revenues" is neither."""
    cells = _cells(row)
    return bool(YEAR_RE.search(row) or DATE_RE.search(row) or HEADER_WORDS.search(row)
                or (len(cells) >= 2 and all(0 < len(c) <= 40 for c in cells)))


def _is_period_row(row: str) -> bool:
    """A one-cell date row ("June 30, 2026"): in a multi-panel table it opens a panel, it doesn't head the table."""
    return "|" not in row and bool(DATE_RE.search(row) or YEAR_RE.fullmatch(row.strip(" ,:")))


def split_header(rows: list[str], max_rows: int = 5) -> tuple[list[str], list[str]]:
    """Header = the leading rows that only contain labels, at least one of which names the columns (a year, date or
    unit). Trailing label-only rows WITHOUT such a token are group labels ("Cost of revenues", "OPERATING
    ACTIVITIES:"), the first rows of the body, not part of the header.
    Multi-panel tables (Microsoft: "June 30, 2026" ... rows ... "June 30, 2025" ... rows): if the header ends in a
    one-cell date row and the body has another one, the date rows are panel labels, so the first one moves to the
    body (where table_chunks carries it as a group label)."""
    n = 0
    while n < min(len(rows), max_rows) and _row_is_header(rows[n]):
        n += 1
    while n and not _row_names_columns(rows[n - 1]):
        n -= 1
    if n and _is_period_row(rows[n - 1]) and any(_is_period_row(r) for r in rows[n:]):
        n -= 1
    return rows[:n], rows[n:]


def is_group_label(row: str) -> bool:
    """A one-cell row with no data that EXPLICITLY opens a group ("INVESTING ACTIVITIES:", "Current assets:"), so a
    split table can repeat it above the rows it labels. Tesla's plain "Cost of revenues" also opens a group, but
    nothing marks where that group ends (the gross-profit rows below it are not costs), so un-marked labels are
    not carried: repeating a label beyond its scope would mislabel rows, the very bug this phase fixes."""
    return "|" not in row and len(row) <= 80 and _cell_is_label(row) and (row.endswith(":") or row.isupper()
                                                                         or _is_period_row(row))


def _is_heading(par: str) -> bool:
    """A short paragraph that reads like a heading ("Revenues", "CONSOLIDATED STATEMENTS OF CASH FLOWS",
    "(in millions)"), not a sentence."""
    return len(par) <= 100 and "\n" not in par and not par.rstrip().endswith((".", ";", ",")) and "|" not in par


def parse_blocks(section_text: str, caption_paragraphs: int = 2) -> list[dict]:
    """Section text with table markers -> blocks. The caption of a table is the up-to-`caption_paragraphs`
    heading-like paragraphs right before it (they stay in the prose too: they also head the text after the table).
    A "table" with no numbers and no header is layout (bullet lists, headings laid out in a grid): it becomes text."""
    blocks: list[dict] = []
    pos = 0
    pattern = re.compile(re.escape(TABLE_START) + r"\n(.*?)\n?" + re.escape(TABLE_END), re.S)
    for m in pattern.finditer(section_text):
        blocks += _paragraphs(section_text[pos:m.start()])
        rows = [r.strip() for r in m.group(1).split("\n") if r.strip()]
        header, body = split_header(rows)
        has_numbers = any(not _row_is_header(r) for r in body)
        if not rows:
            pass
        elif not header and not has_numbers:  # layout table
            blocks += [{"kind": "text", "text": r.replace(" | ", " ")} for r in rows]
        else:
            caption = []
            for b in reversed(blocks[-caption_paragraphs:]):
                if b["kind"] != "text" or not _is_heading(b["text"]):
                    break
                caption.insert(0, b["text"])
            blocks.append({"kind": "table", "caption": " — ".join(caption), "header": header, "rows": body})
        pos = m.end()
    blocks += _paragraphs(section_text[pos:])
    return blocks


def _paragraphs(text: str) -> list[dict]:
    # unmatched markers (a section boundary fell inside a table) are dropped; the rows stay as text
    text = text.replace(TABLE_START, "").replace(TABLE_END, "")
    return [{"kind": "text", "text": p.strip()} for p in re.split(r"\n{2,}", text) if p.strip()]


def render_table(caption: str, header: list[str], rows: list[str]) -> str:
    return "\n".join(([f"Table: {caption}"] if caption else []) + header + rows)
