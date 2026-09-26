"""Similarity metrics from scratch. No API key needed.

Run: uv run python -m phase0_foundations.01_similarity
"""
import numpy as np


def dot(a, b):
    return float(np.dot(a, b))


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def euclidean(a, b):
    return float(np.linalg.norm(a - b))


def normalize(v):
    return v / np.linalg.norm(v)


a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 4.0, 6.0])   # same direction as a, twice the length
c = np.array([3.0, -1.0, 0.5])  # different direction

print("== Raw vectors ==")
for name, other in [("a vs b (same dir)", b), ("a vs c (diff dir)", c)]:
    print(f"{name:20s} dot={dot(a, other):7.3f}  cos={cosine(a, other):6.3f}  "
          f"euclid={euclidean(a, other):6.3f}")

# Key insight #1: cosine ignores magnitude; dot and euclidean don't.
# a and b have cosine 1.0 (identical meaning-direction) but euclidean distance > 0.

print("\n== After L2-normalizing ==")
an, bn, cn = normalize(a), normalize(b), normalize(c)
for name, other in [("a vs b", bn), ("a vs c", cn)]:
    d, cos, e = dot(an, other), cosine(an, other), euclidean(an, other)
    print(f"{name:8s} dot={d:6.3f}  cos={cos:6.3f}  euclid={e:6.3f}  "
          f"check: euclid^2 = 2 - 2*cos -> {e**2:.3f} == {2 - 2 * cos:.3f}")

# Key insight #2: on unit vectors, dot == cosine, and euclid^2 = 2 - 2*cos.
# So all three metrics give the SAME ranking. OpenAI embeddings come pre-normalized,
# which is why vector DBs can use the cheapest op (dot product) safely.

print("\n== Curse of dimensionality: random vectors become near-orthogonal ==")
rng = np.random.default_rng(0)
for d in [2, 10, 100, 1536]:
    x = rng.standard_normal((500, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    sims = (x @ x.T)[np.triu_indices(500, k=1)]
    print(f"dim={d:5d}  mean cos={sims.mean():+.3f}  std={sims.std():.3f}")

# Key insight #3: in high dimensions, unrelated vectors cluster near cos=0 with tiny spread.
# That's why real embedding similarities often live in a narrow band (e.g. 0.2–0.6) —
# absolute thresholds are fragile; rank-based top-k is more robust.
