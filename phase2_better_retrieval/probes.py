"""Probe queries with VERIFIED relevant chunks, to compare retrieval modes with numbers, not vibes.

A chunk is relevant to a probe if its id starts with `prefix` AND its text matches `pattern`.
Every pattern was checked against the chunks (a probe with zero relevant chunks is an error below).
This is a sanity instrument, not an evaluation set: 12 queries, one person's judgments. Phase 4
builds a proper golden set with graded relevance and more metrics.

Kinds: "keyword" = the query hinges on an exact name/term; "semantic" = paraphrase, few shared words;
"numeric" = asks for a figure that sits in a table or a sentence with numbers.
"""
import json
import re

from common.config import DATA_DIR

PROBES = [
    # (query, kind, id prefix, regex over chunk text)
    ("Vision Pro", "keyword", "AAPL_", r"vision pro"),
    ("Cybertruck", "keyword", "TSLA_", r"cybertruck"),
    ("H20 export license", "keyword", "NVDA_", r"\bh20\b"),
    ("Activision Blizzard acquisition", "keyword", "MSFT_", r"activision"),
    ("Can the company lose key people?", "semantic", "", r"key (employees|personnel)"),
    ("Is Tesla too reliant on its CEO?", "semantic", "TSLA_", r"highly dependent on the services of elon musk"),
    ("How big is Amazon's workforce?", "semantic", "AMZN_FY2025", r"full-time and part-time employees"),
    ("Which NVIDIA products drove most data center sales?", "semantic", "NVDA_FY2026", r"majority of our data center revenue"),
    ("What was Apple's total net sales in fiscal 2025?", "numeric", "AAPL_FY2025", r"total net sales \| 416,161"),
    ("How much did Apple spend on share buybacks in fiscal 2025?", "numeric", "AAPL_FY2025", r"repurchased \d+ million shares"),
    ("What was AWS operating income in 2025?", "numeric", "AMZN_FY2025", r"AWS[^\n]*\n?[^\n]*operating income|operating income[^\n]*\n[^\n]*AWS"),
    ("How much did Microsoft Cloud revenue grow in fiscal 2026?", "numeric", "MSFT_FY2026", r"microsoft cloud revenue increased"),
]


def relevant_ids(rows: list[dict]) -> list[set[str]]:
    out = []
    for q, _, prefix, pat in PROBES:
        ids = {r["id"] for r in rows if r["id"].startswith(prefix) and re.search(pat, r["text"], re.I)}
        if not ids:
            raise ValueError(f"probe has no relevant chunk, fix its pattern: {q!r}")
        out.append(ids)
    return out


def hit_and_rr(ranked_ids: list[str], relevant: set[str], k: int) -> tuple[int, float]:
    """hit@k (was ANY relevant chunk in the top k?) and reciprocal rank (1/rank of the first one)."""
    for rank, i in enumerate(ranked_ids[:k], 1):
        if i in relevant:
            return 1, 1.0 / rank
    return 0, 0.0


def load_rows(chunker: str = "recursive", size: int = 350) -> list[dict]:
    path = DATA_DIR / "processed" / f"chunks_{chunker}{size}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]
