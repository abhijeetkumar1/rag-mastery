"""Response caching (Phase 7): skip the whole pipeline for a question we've already answered.

Three layers of caching exist in this project, from safe to risky:
  1. embeddings / structured LLM calls (common.llm): same input, same model => same output. Always safe.
  2. ExactCache: the SAME question (after normalization) from the same access scope => the stored answer.
  3. SemanticCache: a SIMILAR question (cosine of question embeddings >= threshold) => the stored answer.

Every key includes the access SCOPE (which tickers the user may see) and a VERSION string (index, prompt version,
chat model). Without the scope, an answer built from documents user A may see is served to user B: a cross-tenant
leak through the cache, a classic bug. Without the version, a re-index or prompt change keeps serving old answers.

Why the semantic cache is OFF by default (measured in phase7_production/03_cache.py): for financial questions,
questions with DIFFERENT answers are closer in embedding space than true paraphrases.
  "Apple's total net sales in fiscal 2025" vs "... fiscal 2024"           cosine 0.935   (different answer)
  "Tesla's total gross margin" vs "Tesla's automotive gross margin"        cosine 0.975   (different answer)
  "Apple's total net sales in fiscal 2025" vs "How much did Apple sell in total in FY2025?"   0.801 (same answer)
An entity guard (same tickers, years and relative-time words) blocks the year/company swaps but not metric swaps.
Semantic caching is safe for FAQ-style traffic with few distinct answers; for precise numeric lookups use the exact
cache plus the lower layers.

Only answers that passed the output guardrails and were not degraded (no fallback) are stored: caching a bad answer
turns a one-off failure into a persistent one.
"""
import hashlib
import re
import threading
import time
from collections import OrderedDict

import numpy as np

from common.config import HF_EMBED_MODEL
from common.filters import extract_filters
from common.llm import embed

# relative-time words, canonicalized: "most recent" = "latest", "previous" = "prior"
_TEMPORAL = {"latest": "latest", "recent": "latest", "current": "latest", "last": "latest", "prior": "prior",
             "previous": "prior", "first": "first", "next": "next"}
_FY = re.compile(r"\b(?:fy\s?|fiscal\s+(?:year\s+)?)(20\d\d)\b", re.I)


def normalize(question: str) -> str:
    """Case, punctuation, whitespace and 'FY2025' / 'fiscal year 2025' -> 'fiscal 2025'. Only rewrites that can't
    change the answer."""
    q = _FY.sub(r"fiscal \1", question.lower().replace("’", "'"))
    q = re.sub(r"[^\w\s%$.']", " ", q)
    q = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", q)  # drop sentence dots, keep decimals
    return re.sub(r"\s+", " ", q).strip()


def signature(question: str) -> tuple:
    """What two questions must share for a semantic hit: companies, years, relative-time words."""
    q = normalize(question)  # "FY2025" -> "fiscal 2025": a bare \b20\d\d\b can't see the year inside "FY2025"
    return (tuple(extract_filters(question).get("ticker", [])), tuple(sorted(set(re.findall(r"\b20\d\d\b", q)))),
            tuple(sorted({_TEMPORAL[w] for w in re.findall(r"[a-z]+", q) if w in _TEMPORAL})))


class _TTLStore:
    """LRU + TTL in memory. Production: Redis (shared by all workers and replicas, TTL and eviction built in)."""

    def __init__(self, max_items: int, ttl_s: float):
        self.max_items, self.ttl_s = max_items, ttl_s
        self.items: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key):
        with self.lock:
            item = self.items.get(key)
            if item is None or time.time() - item[0] > self.ttl_s:
                self.items.pop(key, None)
                self.misses += 1
                return None
            self.items.move_to_end(key)
            self.hits += 1
            return item[1]

    def put(self, key, value) -> None:
        with self.lock:
            self.items[key] = (time.time(), value)
            self.items.move_to_end(key)
            while len(self.items) > self.max_items:
                self.items.popitem(last=False)


class ExactCache:
    def __init__(self, version: str, max_items: int = 10_000, ttl_s: float = 24 * 3600):
        self.version = version
        self.store = _TTLStore(max_items, ttl_s)

    def key(self, question: str, scope: tuple) -> str:
        return hashlib.sha256(f"{self.version}|{','.join(scope)}|{normalize(question)}".encode()).hexdigest()

    def get(self, question: str, scope: tuple):
        return self.store.get(self.key(question, scope))

    def put(self, question: str, scope: tuple, value) -> None:
        self.store.put(self.key(question, scope), value)


class SemanticCache:
    """Nearest stored question in the same scope (and, with guard=True, the same signature). Embeds with the LOCAL
    model (bge-small, ~5 ms on CPU): the cache must be cheaper than what it saves, and must not depend on the API
    whose outage it might be covering for. Linear scan: fine for thousands of entries; beyond that, a vector index."""

    def __init__(self, version: str, threshold: float = 0.95, guard: bool = True, max_items: int = 5_000,
                 ttl_s: float = 24 * 3600, model: str = HF_EMBED_MODEL):
        self.version, self.threshold, self.guard, self.model = version, threshold, guard, model
        self.max_items, self.ttl_s = max_items, ttl_s
        self.entries: list[dict] = []  # {vec, scope, sig, question, value, t}
        self.lock = threading.Lock()
        self.hits = self.misses = 0

    def _vec(self, question: str) -> np.ndarray:
        return embed([normalize(question)], model=self.model, provider="hf")[0]

    def lookup(self, question: str, scope: tuple) -> tuple[object | None, float, str | None]:
        """(value or None, best similarity, the stored question it matched)."""
        v, sig, now = self._vec(question), signature(question), time.time()
        with self.lock:
            cands = [e for e in self.entries if e["scope"] == scope and now - e["t"] <= self.ttl_s
                     and (not self.guard or e["sig"] == sig)]
            if not cands:
                self.misses += 1
                return None, 0.0, None
            sims = np.array([e["vec"] for e in cands]) @ v
            i = int(np.argmax(sims))
            if sims[i] >= self.threshold:
                self.hits += 1
                return cands[i]["value"], float(sims[i]), cands[i]["question"]
            self.misses += 1
            return None, float(sims[i]), cands[i]["question"]

    def put(self, question: str, scope: tuple, value) -> None:
        e = {"vec": self._vec(question), "scope": scope, "sig": signature(question), "question": question,
             "value": value, "t": time.time()}
        with self.lock:
            self.entries.append(e)
            del self.entries[: max(0, len(self.entries) - self.max_items)]
