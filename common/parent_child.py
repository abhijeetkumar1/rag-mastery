"""Parent-child retrieval ("small-to-big"): SEARCH small chunks, READ big ones.

The chunk-size dilemma: a small chunk has a sharp embedding (one topic, one table piece) and matches precisely, but
gives the LLM too little context; a big chunk gives context but its embedding averages several topics and its
BM25 score is diluted. Parent-child takes both: index children (~200 tokens), then replace each retrieved child by
its parent (~800 tokens) before generation. Several children of one parent collapse into one passage.

    children ranked by retrieval:  P7.c1  P3.c0  P7.c2  P9.c4  P3.c1  P2.c0 ...
    expand + dedupe, keep k=3:     P7 (via c1, c2)   P3 (via c0, c1)   P9 (via c4)

Cost: the prompt is ~3-4x bigger for the same k, and the LLM may get distracted by the parts of a parent that are
irrelevant (lost in the middle). LangChain calls this ParentDocumentRetriever; LlamaIndex, AutoMergingRetriever
(which only merges up when enough children of a parent are retrieved).
"""
import json

from common.chunking import count_tokens
from common.config import DATA_DIR


def load_parents(name: str = "parent800") -> dict[str, dict]:
    path = DATA_DIR / "processed" / f"chunks_{name}.jsonl"
    return {r["id"]: r for r in (json.loads(line) for line in path.read_text().splitlines())}


def expand_to_parents(hits: list[dict], children: dict[str, dict], parents: dict[str, dict], k: int,
                      span: dict | None = None) -> list[dict]:
    """Child hits (best first) -> up to k parent hits, deduplicated, in the order of their best child.
    Each parent hit keeps the best child's scores and lists all its retrieved children ("via")."""
    out: dict[str, dict] = {}
    for h in hits:
        pid = children[h["id"]]["parent_id"]
        if pid in out:
            out[pid]["via"].append(h["id"])
            continue
        if len(out) == k:
            continue  # keep scanning: later children of already-chosen parents are still recorded in "via"
        p = parents[pid]
        out[pid] = {**h, "id": pid, "text": p["text"], "via": [h["id"]], "child_text": h["text"]}
    expanded = list(out.values())
    if span is not None:  # what the trace records: which child pulled in which parent, and the token cost
        span.update(n_children=len(hits), n_parents=len(expanded),
                    mapping=[[p["id"], p["via"]] for p in expanded],
                    child_tokens=sum(count_tokens(h["text"]) for h in hits[:k]),
                    parent_tokens=sum(count_tokens(p["text"]) for p in expanded))
    return expanded
