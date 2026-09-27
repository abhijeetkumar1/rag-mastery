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


def explain_derived(answer: str, numbers: list[str], passages: list[str], rel_tol: float = 0.002) -> dict[str, str]:
    """For numbers not found verbatim: can they be DERIVED from the passage numbers? (Phase 3)
      "716.9 billion"  ≈ 716,924 (millions) / 1000                   unit conversion, needs "billion" in the answer
      "12.4%"          ≈ (716,924 / 637,959 − 1) · 100               growth, needs "%", both numbers on ONE line
      "50.1 billion"   ≈ (331,839 − 281,724) / 1000                  difference, needs "billion", ONE line
    A bare number with no unit must match verbatim. Returns {number: how it was derived}.

    Why so strict: the first version allowed any unit and any pair of source numbers (~10,000 pairs), and a tamper
    test showed it accepted WRONG numbers (15.3% and $736.9B passed; 12.4 was "explained" as 19817 − 7404).
    Requiring the unit AND a single table row (where growth is actually computed) removes most coincidences."""
    units = {}
    for m in re.finditer(r"(\$?\d[\d,]*(?:\.\d+)?)\s*(%|percent|billion|million)?", CITE_RE.sub(" ", answer), re.I):
        units.setdefault(_norm(m.group(1)), (m.group(2) or "").lower())
    lines = [[float(n) for n in extract_numbers(line, skip_trivial=False) if float(n)]
             for p in passages for line in p.split("\n")]
    everything = sorted({y for line in lines for y in line})
    out = {}
    for raw in numbers:
        x, unit = float(raw), units.get(raw, "")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        tol = max(rel_tol * abs(x), 0.5 * 10 ** -decimals)  # tolerate the answer's own rounding
        if unit == "billion":
            y = next((y for y in everything if abs(y / 1000 - x) <= tol), None)
            if y is not None:
                out[raw] = f"{y:g} million = {x:g} billion"
                continue
        for line in lines:
            pairs = [(a, b) for a in line for b in line if a != b]
            if unit in ("%", "percent"):
                hit = next(((a, b) for a, b in pairs if abs((a / b - 1) * 100 - x) <= tol), None)
                how = hit and f"growth ({hit[0]:g} / {hit[1]:g} − 1) · 100"
            elif unit == "billion":
                hit = next(((a, b) for a, b in pairs if a > b and abs((a - b) / 1000 - x) <= tol), None)
                how = hit and f"difference ({hit[0]:g} − {hit[1]:g}) / 1000"
            else:
                hit = how = None
            if hit:
                out[raw] = how
                break
    return out


def check_answer(answer: str, passages: list[str], tolerant: bool = False) -> dict:
    """Run all output checks. `passages` are the texts of the k passages in the prompt, in order.
    tolerant=True (Phase 3): numbers derivable from cited figures (unit conversion, growth %, difference)
    are moved from ungrounded_numbers to derived_numbers instead of failing the check."""
    report = {"refused": is_refusal(answer), **check_citations(answer, len(passages))}
    valid = [n for n in report["cited"] if 1 <= n <= len(passages)]
    # verify numbers against what the answer CITED; if it cited nothing, against everything retrieved
    sources = [passages[n - 1] for n in valid] or passages
    report["ungrounded_numbers"] = [] if report["refused"] else check_numeric_grounding(answer, sources)
    if tolerant and report["ungrounded_numbers"]:
        report["derived_numbers"] = explain_derived(answer, report["ungrounded_numbers"], sources)
        report["ungrounded_numbers"] = [n for n in report["ungrounded_numbers"] if n not in report["derived_numbers"]]
    report["passed"] = not (report["invalid_citations"] or report["uncited_answer"] or report["ungrounded_numbers"])
    return report
