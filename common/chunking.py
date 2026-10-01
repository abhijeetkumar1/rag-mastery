"""Chunkers. Sizes are in TOKENS (not chars), because model limits are in tokens.

Two strategies, from dumbest to the usual default:
  fixed_token_chunks : slide a window of N tokens with overlap. Ignores structure entirely.
  recursive_chunks   : split on the coarsest separator that fits (paragraph > line > sentence > word),
                       then greedily pack pieces up to N tokens. Keeps paragraphs/table rows intact.
Phase 5 adds structure-aware chunking over parsed blocks (common.structure):
  structured_chunks  : prose as recursive_chunks, tables split between rows with the header repeated per piece.
"""
import tiktoken

from common.structure import is_group_label, render_table

_enc = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def fixed_token_chunks(text: str, size: int = 350, overlap: int = 50) -> list[str]:
    ids = _enc.encode(text)
    step = size - overlap
    return [_enc.decode(ids[i : i + size]) for i in range(0, max(len(ids) - overlap, 1), step)]


SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    """Recursively split until every piece is <= size tokens."""
    if count_tokens(text) <= size:
        return [text]
    if not seps:  # no separator left: hard cut by tokens
        return fixed_token_chunks(text, size, 0)
    sep, rest = seps[0], seps[1:]
    out = []
    for part in text.split(sep):
        if part.strip():
            out += _split(part + sep, size, rest)
    return out


def recursive_chunks(text: str, size: int = 350, overlap: int = 50) -> list[str]:
    pieces = _split(text, size, SEPARATORS)
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for p in pieces:
        n = count_tokens(p)
        if cur and cur_tokens + n > size:
            chunks.append("".join(cur).strip())
            # carry trailing pieces forward as overlap so a fact split across the boundary survives
            carry, carry_tokens = [], 0
            for q in reversed(cur):
                t = count_tokens(q)
                if carry_tokens + t > overlap:
                    break
                carry.insert(0, q)
                carry_tokens += t
            while carry and carry_tokens + n > size:  # overlap must not push the next chunk over size
                carry_tokens -= count_tokens(carry.pop(0))
            cur, cur_tokens = carry, carry_tokens
        cur.append(p)
        cur_tokens += n
    if cur:
        chunks.append("".join(cur).strip())
    return [c for c in chunks if c]


CHUNKERS = {"fixed": fixed_token_chunks, "recursive": recursive_chunks}


# ---------------------------------------------------------------- Phase 5: structure-aware chunking


def table_pieces(header: list[str], rows: list[str], size: int, caption: str = "") -> list[list[str]]:
    """The rows of each piece of a table (see table_chunks). Returned as row lists so parent-child chunking can split
    a parent table piece again into smaller children, each with the same header."""
    if count_tokens(render_table(caption, header, rows)) <= size:
        return [rows]
    fixed = count_tokens(render_table(caption, header, []))
    budget = max(size - fixed, size // 3)  # a huge header must not leave room for zero rows
    pieces, cur, cur_tokens, group = [], [], 0, None
    for row in rows:
        for part in (_split(row, budget, SEPARATORS[2:]) if count_tokens(row) > budget else [row]):
            n = count_tokens(part)
            if cur and cur_tokens + n > budget:
                pieces.append(cur)
                carry = [group] if group and part != group else []
                cur, cur_tokens = carry, sum(count_tokens(g) for g in carry)
            cur.append(part)
            cur_tokens += n
        if is_group_label(row):
            group = row
    if cur:
        pieces.append(cur)
    return pieces


def table_chunks(caption: str, header: list[str], rows: list[str], size: int) -> list[str]:
    """A table that fits stays whole. A bigger one is split BETWEEN ROWS, and every piece repeats the caption and
    the header rows (and the group label, e.g. "INVESTING ACTIVITIES:", when a piece starts inside a group), so
    each piece is self-describing: "Total gross margin | 18.0" always travels with "(Dollars in millions) | 2025".
    The repeated header costs tokens in every piece; that is the price of a chunk that can be read alone."""
    return [render_table(caption, header, p) for p in table_pieces(header, rows, size, caption)]


def structured_chunks(blocks: list[dict], size: int = 350, overlap: int = 50) -> list[dict]:
    """Blocks (common.structure.parse_blocks) -> chunks {"kind": "text"|"table", "body", "caption"} (+ "header" and
    "rows" for tables).
    Runs of prose between tables go through recursive_chunks as before; each table goes through table_chunks.
    Prose and tables are never mixed in one chunk, so a chunk is either a passage or a (piece of a) table."""
    out: list[dict] = []
    run: list[str] = []

    def flush():
        if run:
            out.extend({"kind": "text", "body": c, "caption": ""} for c in recursive_chunks("\n\n".join(run), size, overlap))
            run.clear()

    for b in blocks:
        if b["kind"] == "text":
            run.append(b["text"])
        else:
            flush()
            for rows in table_pieces(b["header"], b["rows"], size, b["caption"]):
                out.append({"kind": "table", "body": render_table(b["caption"], b["header"], rows), "caption": b["caption"],
                            "header": b["header"], "rows": rows})
    flush()
    return out
