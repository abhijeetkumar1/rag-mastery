"""Output guardrails: cheap, deterministic checks on a generated answer before it reaches the user.

  check_citations          every [n] must point at a passage that was actually in the prompt
  check_numeric_grounding  every number in the answer must appear in a cited passage
                           (for financial Q&A, a hallucinated number is the most damaging failure)

These are rule-based on purpose: no extra LLM call, milliseconds of latency, and fully explainable.
Their limit is precision: a correctly DERIVED number (e.g. a growth % computed from two figures)
is flagged too. That's why the policy is "warn", not "block" (see 05_guardrails_demo.py).
"""
import re

REFUSAL = "i don't know based on the provided filings"
CITE_RE = re.compile(r"\[(\d+)\]")
# a number not glued to letters or hyphens (skips "FY2025", "10-K", "E-4471"); optional $, thousands commas, decimals, %
NUM_RE = re.compile(r"(?<![\w.\-])\$?\d[\d,]*(?:\.\d+)?%?(?![\w\-])")


def is_refusal(answer: str) -> bool:
    return REFUSAL in answer.lower().replace("’", "'")


def _norm(num: str) -> str:
    """'$416,161' -> '416161', '6.40%' -> '6.4'. Comparable across answer and passages."""
    s = num.lstrip("$").rstrip("%").replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def extract_numbers(text: str, skip_trivial: bool = True) -> set[str]:
    """skip_trivial drops years and bare single digits. Use it on ANSWERS only: sources must keep
    every number, e.g. the table cell "| 6 |" is the evidence for an answer's "6%"."""
    out = set()
    for m in NUM_RE.finditer(text):
        raw = m.group().rstrip(",")
        n = _norm(raw)
        is_year = len(n) == 4 and n.isdigit() and 1900 <= int(n) <= 2100 and "," not in raw
        is_tiny = n.isdigit() and len(n) == 1 and not raw.endswith("%")  # "1", "2" are rarely facts
        if n and not (skip_trivial and (is_year or is_tiny)):
            out.add(n)
    return out


def check_citations(answer: str, k: int) -> dict:
    cited = sorted({int(n) for n in CITE_RE.findall(answer)})
    return {
        "cited": cited,
        "invalid_citations": [n for n in cited if not 1 <= n <= k],  # e.g. [7] when only 5 passages
        "uncited_answer": not cited and not is_refusal(answer),       # made claims, cited nothing
    }


def check_numeric_grounding(answer: str, passages: list[str]) -> list[str]:
    """Numbers in the answer (citations stripped) that appear in none of the given passages."""
    in_answer = extract_numbers(CITE_RE.sub(" ", answer))
    in_sources = set().union(*(extract_numbers(p, skip_trivial=False) for p in passages)) if passages else set()
    return sorted(in_answer - in_sources, key=lambda x: (len(x), x))


def check_answer(answer: str, passages: list[str]) -> dict:
    """Run all output checks. `passages` are the texts of the k passages in the prompt, in order."""
    report = {"refused": is_refusal(answer), **check_citations(answer, len(passages))}
    valid = [n for n in report["cited"] if 1 <= n <= len(passages)]
    # verify numbers against what the answer CITED; if it cited nothing, against everything retrieved
    sources = [passages[n - 1] for n in valid] or passages
    report["ungrounded_numbers"] = [] if report["refused"] else check_numeric_grounding(answer, sources)
    report["passed"] = not (report["invalid_citations"] or report["uncited_answer"] or report["ungrounded_numbers"])
    return report
