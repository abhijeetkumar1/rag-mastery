"""Real embeddings: where semantic search shines and where it fails.

Run: uv run python -m phase0_foundations.02_embeddings
"""
import numpy as np

from common.llm import embed

sentences = [
    "The company's revenue grew 12% year over year.",         # 0
    "Sales increased by twelve percent compared to last year.",  # 1 paraphrase of 0, no shared keywords
    "Revenue declined 12% year over year.",                    # 2 opposite meaning, near-identical words
    "Apple released a new iPhone this fall.",                  # 3
    "I ate an apple and a banana for breakfast.",              # 4 same word 'apple', different sense
    "Error code E-4471 occurs when the disk is full.",         # 5 rare exact token
    "Error code E-4417 occurs when the network times out.",    # 6 near-identical rare token
]
for i, s in enumerate(sentences, start=1):
    print(f"{i:2d}: {s}")

vecs = embed(sentences)
print(f"shape={vecs.shape}  norms≈{np.linalg.norm(vecs, axis=1).round(3)[:3]}  (pre-normalized)\n")

sim = vecs @ vecs.T  # unit vectors -> dot == cosine

def show(i, j, label):
    print(f"{sim[i, j]:.3f}  {label}")

show(0, 1, "paraphrase, no keyword overlap     -> should be HIGH (semantic win)")
show(0, 2, "opposite meaning, same words       -> often ALSO high (negation blindness)")
show(3, 4, "'apple' company vs fruit           -> low-ish (context disambiguates)")
show(5, 6, "E-4471 vs E-4417, different errors -> high (exact-ID blindness)")

# Takeaways (these motivate later phases):
#  - Dense embeddings capture meaning, not exact tokens -> IDs/codes/SKUs need BM25 (Phase 2 hybrid).
#  - Negation/numbers are weakly encoded -> retrieval can fetch the "wrong but similar" chunk;
#    a reranker (Phase 2) or the LLM must do the fine-grained reading.

print("\n== Dimension truncation (Matryoshka) ==")
# text-embedding-3 models are trained so a PREFIX of the vector is itself a usable embedding.
# (bge-small / MiniLM are NOT Matryoshka-trained -> expect quality to fall off much faster.)
full = vecs.shape[1]
for d in [full, full // 6, full // 24]:
    v = vecs[:, :d] / np.linalg.norm(vecs[:, :d], axis=1, keepdims=True)
    s = v @ v.T
    print(f"dim={d:4d}  paraphrase={s[0, 1]:.3f}  opposite={s[0, 2]:.3f}  fruit-vs-company={s[3, 4]:.3f}")
# Trade-off: 6x smaller index & faster search for a small quality hit. Real interview topic.
