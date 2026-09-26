"""BM25 (Okapi) from scratch: the classic sparse / keyword retriever.

score(q, d) = Σ_{t ∈ q}  idf(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )
idf(t)      = ln( 1 + (N − df(t) + 0.5) / (df(t) + 0.5) )          (Lucene/Elasticsearch variant, never negative)

  tf    term frequency in the doc   -> more mentions = more relevant, but SATURATING (k1)
  idf   inverse document frequency  -> rare terms ("Blackwell", "E-4471") count far more than common ones
  |d|   doc length vs average       -> long docs don't win just by containing more words (b)
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np

# Keep tokens like "10-k", "e-4471", "416,161", "u.s." whole: exact identifiers are BM25's strength
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,\-'][a-z0-9]+)*")
STOPWORDS = set("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it its
of on or our over such than that the their them then there these they this those to under up was we were
what when where which while who whom why will with would you your also any all may more most other some
""".split())


def tokenize(text: str) -> list[str]:
    tokens = (t.removesuffix("'s") for t in TOKEN_RE.findall(text.lower().replace("’", "'")))  # "apple's" -> "apple"
    return [t for t in tokens if t not in STOPWORDS]


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        docs = [tokenize(t) for t in texts]
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean())
        # inverted index: term -> list of (doc_idx, tf). Only docs containing the term are touched at query time.
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        # length normalization per doc is query-independent: precompute it
        self._norm = k1 * (1 - b + b * self.doc_len / self.avgdl)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score for every doc (0 for docs sharing no term with the query)."""
        s = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue  # out-of-vocabulary term: contributes nothing (dense retrieval would still "match" it)
            idx, tf = zip(*self.postings[term])
            idx, tf = np.array(idx), np.array(tf, dtype=np.float32)
            s[idx] += self.idf[term] * tf * (self.k1 + 1) / (tf + self._norm[idx])
        return s

    def search(self, query: str, k: int = 5, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Top-k (doc_idx, score). `mask` = boolean array of allowed docs (metadata filter)."""
        s = self.scores(query)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        k = min(k, int(np.isfinite(s).sum() if mask is not None else self.n))
        top = np.argpartition(-s, k - 1)[:k] if k else np.array([], dtype=int)
        top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]
