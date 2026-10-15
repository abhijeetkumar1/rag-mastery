"""Semantic cache: can a similarity threshold tell a paraphrase (same answer) from a near-miss (different answer)?

Run: uv run python -m phase7_production.03_cache
     uv run python -m phase7_production.03_cache --model text-embedding-3-small --provider openai   # needs the API

For each labeled pair (cache_cases.py) the base question is "in the cache" and the other is the new request:
  hit precision  of the pairs that would be served from cache, the share with the SAME answer (1 - wrong-answer rate)
  hit recall     of the SAME pairs, the share served from cache (the saving)
measured per threshold, without and with the entity guard (common.cache.signature: same companies, years and
relative-time words). Also: how many paraphrases the EXACT cache (after normalization) catches, and the closest pairs
of DIFFERENT questions in the golden dev set (real traffic shape, all different answers).
Free: local bge-small by default, no API calls.
"""
import argparse
import importlib

import numpy as np

from common.cache import normalize, signature
from common.config import HF_EMBED_MODEL
from common.llm import embed

cases = importlib.import_module("phase7_production.cache_cases")
retrieval5 = importlib.import_module("phase5_advanced_indexing.05_retrieval_eval")


def sims(pairs: list[tuple[str, str]], model: str, provider: str) -> np.ndarray:
    a = embed([normalize(x) for x, _ in pairs], model=model, provider=provider)
    b = embed([normalize(y) for _, y in pairs], model=model, provider=provider)
    return (a * b).sum(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=HF_EMBED_MODEL)
    ap.add_argument("--provider", default="hf")
    args = ap.parse_args()
    B = cases.BASE
    same = [(B[k], q) for k, q in cases.SAME]
    diff = [(B[k], q) for k, q, _ in cases.DIFF]
    kinds = [c for _, _, c in cases.DIFF]
    s_same, s_diff = sims(same, args.model, args.provider), sims(diff, args.model, args.provider)
    g_same = np.array([signature(a) == signature(b) for a, b in same])
    g_diff = np.array([signature(a) == signature(b) for a, b in diff])
    print(f"embedding model: {args.model}   pairs: {len(same)} same-answer, {len(diff)} different-answer\n")
    print(f"cosine, same-answer pairs:       min {s_same.min():.3f}  median {np.median(s_same):.3f}  max {s_same.max():.3f}")
    print(f"cosine, different-answer pairs:  min {s_diff.min():.3f}  median {np.median(s_diff):.3f}  max {s_diff.max():.3f}")
    print("\nby kind of change (different-answer pairs): median cosine, and how many the entity guard blocks")
    for k in sorted(set(kinds)):
        idx = [i for i, c in enumerate(kinds) if c == k]
        print(f"  {k:14s} n={len(idx):2d}  median {np.median(s_diff[idx]):.3f}  guard blocks {sum(not g_diff[i] for i in idx)}/{len(idx)}")
    print(f"  the guard also blocks {sum(not g for g in g_same)}/{len(same)} same-answer pairs (false blocks):")
    for (a, b), g in zip(same, g_same):
        if not g:
            print(f"    {signature(a)} vs {signature(b)}: {b}")

    print("\nthreshold   no guard: hits right/wrong  precision recall  |  guard: hits right/wrong  precision recall")
    for t in (0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98):
        row = []
        for guard in (False, True):
            ok = int(((s_same >= t) & (g_same | (not guard))).sum())
            bad = int(((s_diff >= t) & (g_diff | (not guard))).sum())
            prec = ok / (ok + bad) if ok + bad else float("nan")
            row.append(f"{ok:3d} / {bad:<3d}  {prec:5.2f}  {ok / len(same):5.2f}")
        print(f"  {t:.2f}      {row[0]}            |  {row[1]}")

    exact = sum(normalize(a) == normalize(b) for a, b in same)
    print(f"\nexact cache (normalized string match) catches {exact}/{len(same)} paraphrases, 0 wrong answers by construction")

    qs = [x["question"] for x in retrieval5.golden("dev")]
    V = embed([normalize(q) for q in qs], model=args.model, provider=args.provider)
    S = V @ V.T
    np.fill_diagonal(S, -1)
    i, j = np.unravel_index(np.argsort(-S, axis=None)[:10:2], S.shape)  # symmetric: every pair appears twice
    print(f"\nclosest pairs of DIFFERENT golden dev questions ({len(qs)} questions, all different answers):")
    for a, b in zip(i, j):
        print(f"  {S[a, b]:.3f}  guard {'passes' if signature(qs[a]) == signature(qs[b]) else 'blocks'}\n"
              f"         {qs[a][:110]}\n         {qs[b][:110]}")
    for t in (0.90, 0.95):
        n = int(((S >= t).sum(1) > 0).sum())
        print(f"  dev questions with another dev question at cosine >= {t}: {n}/{len(qs)}")


if __name__ == "__main__":
    main()
