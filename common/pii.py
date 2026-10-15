"""PII detection and redaction (Phase 7). Rule-based: regexes plus validity checks (Luhn for card numbers, the SSN
area/group rules), so a random 16-digit number or "000-12-3456" isn't flagged.

Where redaction runs in the service, and why:
  traces / logs / OTel export     ALWAYS. Phase 6's open issue: Phoenix stored full prompts. Observability backends
                                  are read by many people and kept for months; they must never become a PII store.
  online-eval queue, feedback     always (same reason: humans read them)
  the LLM prompt                  configurable (PII_TO_LLM=redact|allow). A question never needs a user's SSN to be
                                  answered from 10-Ks, so redacting is free here; in a support bot it would break the task.
  the cache                       keyed and stored on the redacted question: the cache must never hold raw PII.
The corpus is NOT redacted: 10-K filings are public, and they contain phone numbers (headquarters), which is why
measuring false positives on the corpus matters (phase7_production/04_guardrails.py).

Regex PII catches structured identifiers (emails, phones, SSNs, cards, IBANs, IPs). It misses names and addresses,
which need NER (Microsoft Presidio = regex recognizers + spaCy NER + context words). NER over 10-K text would flag
every executive's name: public information, a false positive for this corpus.
"""
import re

PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "SSN": r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
    "CARD": r"\b(?:\d[ -]?){12,18}\d\b",
    "PHONE": r"(?<![\w$.,])(?:\+?1[ .-]?)?\(?\b\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b(?![.,]?\d)",
    "IBAN": r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b",
    "IP": r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
}
_RX = {k: re.compile(p) for k, p in PATTERNS.items()}


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d) * (2 if alt else 1)
        total += n - 9 if n > 9 else n
        alt = not alt
    return total % 10 == 0


def find_pii(text: str) -> list[tuple[str, int, int, str]]:
    """[(type, start, end, matched text)], non-overlapping, earliest-first (a card number is not also a phone)."""
    found = []
    for kind, rx in _RX.items():
        for m in rx.finditer(text):
            s = m.group()
            if kind == "CARD" and not (13 <= len(re.sub(r"\D", "", s)) <= 19 and _luhn(re.sub(r"\D", "", s))):
                continue
            found.append((kind, m.start(), m.end(), s))
    found.sort(key=lambda f: (f[1], -(f[2] - f[1])))
    out, end = [], -1
    for f in found:
        if f[1] >= end:
            out.append(f)
            end = f[2]
    return out


def redact(text: str) -> tuple[str, list[str]]:
    """Replace each PII span by [TYPE]. Returns (redacted text, the types found). The types go in the trace; the
    values never do."""
    if not text:
        return text, []
    spans = find_pii(text)
    for kind, s, e, _ in reversed(spans):
        text = text[:s] + f"[{kind}]" + text[e:]
    return text, [k for k, *_ in spans]
