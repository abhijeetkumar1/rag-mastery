"""A 'vector database' in ~30 lines: exact (brute-force) k-NN search.

Run: uv run python -m phase0_foundations.03_tiny_vector_search
"""
import time

import numpy as np

from common.llm import embed


class TinyVectorStore:
    def __init__(self):
        self.texts: list[str] = []
        self.matrix: np.ndarray | None = None

    def add(self, texts: list[str]):
        vecs = embed(texts)
        self.texts += texts
        self.matrix = vecs if self.matrix is None else np.vstack([self.matrix, vecs])

    def search(self, query: str, k: int = 3):
        q = embed([query])[0]
        scores = self.matrix @ q                      # O(N·d): one dot product per doc
        top = np.argpartition(-scores, k)[:k]         # O(N) partial sort, not full O(N log N)
        top = top[np.argsort(-scores[top])]
        return [(float(scores[i]), self.texts[i]) for i in top]


docs = [
    "Our refund policy allows returns within 30 days of purchase with a receipt.",
    "Shipping is free for orders above $50 within the continental US.",
    "Premium members get 24/7 phone support and a dedicated account manager.",
    "To reset your password, click 'Forgot password' on the login page.",
    "We store customer data in encrypted form in AWS us-east-1.",
    "Gift cards cannot be exchanged for cash and never expire.",
    "Orders are usually dispatched within 2 business days.",
]

store = TinyVectorStore()
store.add(docs)

for q in ["Can I get my money back?", "How long until my package ships?", "Is my data safe?"]:
    print(f"\nQ: {q}")
    for score, text in store.search(q, k=2):
        print(f"  {score:.3f}  {text}")

# Why do real vector DBs exist if this works? Scale.
print("\n== Brute force at scale (random 1536-d vectors) ==")
rng = np.random.default_rng(0)
q = rng.standard_normal(1536).astype(np.float32)
for n in [10_000, 100_000, 300_000]:
    m = rng.standard_normal((n, 1536)).astype(np.float32)
    t = time.perf_counter()
    _ = m @ q
    ms = (time.perf_counter() - t) * 1000
    print(f"N={n:>7,}  memory={m.nbytes / 1e9:.2f} GB  search={ms:.1f} ms")
# Linear in N, and memory = N * d * 4 bytes. 10M chunks ≈ 61 GB and ~seconds/query.
# -> ANN indexes (HNSW, IVF, PQ) trade a little recall for sub-linear search. See NOTES.md.
