"""Evaluation toolkit (Phase 4): retrieval metrics, statistics, deterministic checks, and LLM judges.

Retrieval metrics take a ranked list of chunk ids and a {chunk_id: grade} dict of relevant chunks
(grade 2 = the evidence in the filing the question is about, 1 = the same evidence in another filing).
"""
import math
import random
import re

from common.config import JUDGE_MODEL
from common.guardrails import CITE_RE, _norm, extract_numbers
from common.llm import chat_json

# ---------------------------------------------------------------- retrieval metrics


def recall_at(ranked: list[str], rel: dict[str, int], k: int) -> float:
    return len(set(ranked[:k]) & rel.keys()) / len(rel) if rel else 0.0


def precision_at(ranked: list[str], rel: dict[str, int], k: int) -> float:
    return len(set(ranked[:k]) & rel.keys()) / k


def hit_at(ranked: list[str], rel: dict[str, int], k: int) -> float:
    return float(bool(set(ranked[:k]) & rel.keys()))


def mrr_at(ranked: list[str], rel: dict[str, int], k: int) -> float:
    return next((1.0 / r for r, i in enumerate(ranked[:k], 1) if i in rel), 0.0)


def ndcg_at(ranked: list[str], rel: dict[str, int], k: int) -> float:
    """DCG = Σ (2^grade − 1) / log2(rank + 1), normalized by the DCG of the ideal ordering."""
    dcg = sum((2 ** rel.get(i, 0) - 1) / math.log2(r + 1) for r, i in enumerate(ranked[:k], 1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(r + 1) for r, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------- statistics


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap CI: resample the QUESTIONS with replacement."""
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return sum(values) / len(values), means[int(0.025 * n)], means[int(0.975 * n)]


def paired_bootstrap(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float, float]:
    """Difference b − a on the SAME questions: mean, 95% CI, and P(diff ≤ 0). Pairing removes question
    difficulty from the noise: a hard question is hard for both systems."""
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), boots[int(0.025 * n)], boots[int(0.975 * n)], sum(d <= 0 for d in boots) / n


# ---------------------------------------------------------------- deterministic answer checks


def number_matches(ref: str, answer: str) -> bool:
    """Does the answer state the reference number, allowing millions→billions and rounding?
    ref "416,161" (millions) matches "416,161", "$416.2 billion", "416.16 billion"; ref "18.0" (%) matches "18%"."""
    r = float(_norm(ref))
    for m in re.finditer(r"(\$?\d[\d,]*(?:\.\d+)?)\s*(billion|million|%)?", CITE_RE.sub(" ", answer), re.I):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        x = float(_norm(raw))
        dec = len(_norm(raw).split(".")[1]) if "." in _norm(raw) else 0
        tol = 0.5 * 10 ** -dec
        if abs(x - r) <= max(tol, 1e-9) or (unit == "billion" and abs(x - r / 1000) <= tol + 1e-9):
            return True
    return False


# ---------------------------------------------------------------- LLM judges

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["correct", "partially_correct", "incorrect", "refused"]},
    },
    "required": ["reasoning", "verdict"], "additionalProperties": False,
}


def judge_correctness(question: str, reference: str, answer: str, model: str = JUDGE_MODEL) -> tuple[dict, dict]:
    """Answer vs the reference answer. Reasoning is generated BEFORE the verdict (schema order)."""
    system = """You grade answers from a question-answering system over SEC 10-K filings against a reference answer.
- "correct": states the same facts as the reference (same numbers up to rounding/unit conversion, same entity,
  same period, same metric). Extra correct detail is fine.
- "partially_correct": part of the reference is right and part is missing or wrong (e.g. one of two companies).
- "incorrect": contradicts the reference, or gives a number for a different metric, entity or period
  (e.g. a segment's figure instead of the company total, guidance instead of actuals, a net figure instead of gross).
- "refused": says it doesn't know / can't answer.
Think step by step in "reasoning", then give the verdict."""
    user = f"Question: {question}\n\nReference answer: {reference}\n\nAnswer to grade: {answer}"
    return chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}],
                     CORRECTNESS_SCHEMA, name="correctness", model=model)


FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "passages": {"type": "array", "items": {"type": "integer"}},
                "check": {"type": "string"},
                "verdict": {"type": "string", "enum": ["supported", "unsupported", "contradicted"]},
            },
            "required": ["claim", "passages", "check", "verdict"], "additionalProperties": False}},
    },
    "required": ["claims"], "additionalProperties": False,
}


def judge_faithfulness(answer: str, passages: list[str], model: str = JUDGE_MODEL) -> tuple[dict, dict]:
    """Claim-level groundedness: split the answer into atomic factual claims, verify each against the passages.
    Designed to catch "right number, wrong label": the number exists, but for another metric/entity/period."""
    system = """You check whether an answer is fully supported by numbered source passages from SEC 10-K filings.
1. Split the answer into atomic factual claims (one number or fact each). Skip hedges and meta-statements.
2. For each claim, find the passage(s) that bear on it and write "check": quote the relevant passage text and
   compare EVERY attribute: entity (company vs a segment or product line), metric (total vs segment, gross vs net,
   GAAP vs non-GAAP, actuals vs guidance/forecast), period (which fiscal year / column), and unit.
3. verdict: "supported" only if the passages state it with all attributes matching (arithmetic derived from
   passage numbers, like a growth rate or a unit conversion, counts as supported if computed correctly);
   "contradicted" if a passage states something incompatible; otherwise "unsupported".
A number that appears in a passage but belongs to a different metric, entity or period is NOT supported."""
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    user = f"Passages:\n\n{ctx}\n\nAnswer to check:\n{answer}"
    return chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}],
                     FAITHFULNESS_SCHEMA, name="faithfulness", model=model)


def faithfulness_score(result: dict) -> float:
    claims = result["claims"]
    return sum(c["verdict"] == "supported" for c in claims) / len(claims) if claims else 1.0


# ---- v2: the judge EXTRACTS per-attribute matches, code DECIDES. The v1 judge (gpt-4.1) once wrote "the metric is
# 'purchases of property and equipment, net of proceeds...'" and still returned "supported" for an answer claiming
# plain purchases: reasoning and verdict disagreed. With a rubric of booleans, a mismatch can't be argued away.
ATTRS = ["value_matches", "entity_matches", "metric_matches", "period_matches", "unit_matches"]
FAITHFULNESS_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "quote": {"type": "string"},
                **{a: {"type": "boolean"} for a in ATTRS},
                "contradicted": {"type": "boolean"},
            },
            "required": ["claim", "quote", *ATTRS, "contradicted"], "additionalProperties": False}},
    },
    "required": ["claims"], "additionalProperties": False,
}


def judge_faithfulness_v2(answer: str, passages: list[str], model: str = JUDGE_MODEL) -> tuple[dict, dict]:
    system = """You check an answer against numbered source passages from SEC 10-K filings.
Split the answer into atomic factual claims. For each claim copy the most relevant passage text into "quote"
(exact words; "" if nothing relevant), then answer each question strictly FROM THE PASSAGES ALONE:
- value_matches: the passage states this value (a correct unit conversion or arithmetic from passage numbers counts)
- entity_matches: the passage's figure is for the same entity (the company, not a segment or product line,
  unless the claim names that segment)
- metric_matches: the same metric, with every qualifier ("net of ...", "non-GAAP", "segment", guidance vs actual)
- period_matches: the passage itself shows the value belongs to the claimed period (a column header, a date, or
  "fiscal year X" in the passage). If the passage doesn't show which period a number belongs to: false.
- unit_matches: same unit and scale (millions vs billions, % vs points)
- contradicted: a passage states something incompatible with the claim
For a qualitative claim, judge the attributes that apply and set the others to true."""
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    out, usage = chat_json([{"role": "system", "content": system},
                            {"role": "user", "content": f"Passages:\n\n{ctx}\n\nAnswer to check:\n{answer}"}],
                           FAITHFULNESS_SCHEMA_V2, name="faithfulness_v2", model=model)
    for c in out["claims"]:  # the verdict is computed, not generated
        failed = [a for a in ATTRS if not c[a]]
        c["verdict"] = "contradicted" if c["contradicted"] else ("unsupported" if failed else "supported")
        c["check"] = f"failed: {', '.join(failed)} | quote: {c['quote'][:120]}" if failed or c["contradicted"] else c["quote"][:150]
    return out, usage
