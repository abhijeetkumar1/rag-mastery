"""Chunkers. Sizes are in TOKENS (not chars), because model limits are in tokens.

Two strategies, from dumbest to the usual default:
  fixed_token_chunks : slide a window of N tokens with overlap. Ignores structure entirely.
  recursive_chunks   : split on the coarsest separator that fits (paragraph > line > sentence > word),
                       then greedily pack pieces up to N tokens. Keeps paragraphs/table rows intact.
"""
import tiktoken

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
