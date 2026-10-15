"""Phase 7 guardrail layer, tested piece by piece, with false positives and false negatives.

Run: uv run python -m phase7_production.04_guardrails            # all sections (no API calls except moderation)
     uv run python -m phase7_production.04_guardrails --only pii access

  pii         labeled strings: precision/recall per type; then false positives on the 3,043 chunks of the corpus
  access      ACL: an AAPL-only user over all dev questions (pre-filter), a simulated filter bug (post-check),
              the agent's graph tool with and without its own filter, and cache isolation between scopes
  limits      token bucket burst behaviour; the pre-call cost estimate vs the ACTUAL tokens of Phase 5's dev runs;
              how many requests a $0.002/day user gets
  moderation  the OpenAI moderation call, or (API down) what fail-open / fail-closed do
"""
import argparse
import importlib
import re
import time

from common.access import User
from common.cache import ExactCache
from common.chunking import count_tokens
from common.guardrails import format_untrusted
from common.moderation import moderate
from common.pii import find_pii, redact
from common.ratelimit import SpendLimiter, TokenBucket, estimate_cost
from common.resilience import BREAKERS
from common.trace import load_traces

ask5 = importlib.import_module("phase5_advanced_indexing.06_ask")
retrieval5 = importlib.import_module("phase5_advanced_indexing.05_retrieval_eval")
service = importlib.import_module("phase7_production.service")

# (text, expected PII types). Positives written to look like real user input; negatives look like 10-K text.
PII_CASES = [
    ("My email is jane.doe@example.com, what was Apple's revenue?", ["EMAIL"]),
    ("Call me at (415) 555-0132 about Tesla margins", ["PHONE"]),
    ("my number is +1 212-555-0199", ["PHONE"]),
    ("SSN 123-45-6789 — compare Microsoft and Amazon", ["SSN"]),
    ("Card 4111 1111 1111 1111 exp 12/27, what's NVIDIA's dividend?", ["CARD"]),
    ("charge 5500-0000-0000-0004 please", ["CARD"]),
    ("Send to IBAN DE89 3704 0044 0532 0130 00", ["IBAN"]),
    ("from IP 192.168.10.24 I asked about Amazon", ["IP"]),
    ("reach me: a.b+test@corp.co.uk or 650.555.0101", ["EMAIL", "PHONE"]),
    ("my ssn is 078 05 1120", ["SSN"]),                                      # no dashes: expected miss
    ("email me at jane at example dot com", ["EMAIL"]),                       # obfuscated: expected miss
    # negatives (10-K-like numbers that must NOT be flagged)
    ("Total net sales were $416,161 million in 2025, up 6%.", []),
    ("Commission File Number: 001-36743", []),
    ("CUSIP 037833100, 15,115,823,000 shares outstanding", []),
    ("Revenue of 245,122 and 211,915 for fiscal 2025 and 2024", []),
    ("The notes mature between 2026-2064 at rates of 0.000%-4.850%", []),
    ("Version 4111111111111112 of the build", []),                            # 16 digits, fails Luhn
    ("Apple Inc., One Apple Park Way, Cupertino, California 95014, (408) 996-1010", ["PHONE"]),  # public: still PII-shaped
    ("ISBN 978-0-306-40615-7 is cited", []),
    ("Section 13 or 15(d), 12(b) of the Act; Rule 10b5-1 trading arrangements", []),
]


def section_pii(R) -> None:
    print("=" * 30 + " PII detection (common/pii.py)")
    tp, fp, fn = {}, {}, {}
    for text, want in PII_CASES:
        got = [k for k, *_ in find_pii(text)]
        for k in set(got) | set(want):
            g, w = got.count(k), want.count(k)
            tp[k] = tp.get(k, 0) + min(g, w)
            fp[k] = fp.get(k, 0) + max(0, g - w)
            fn[k] = fn.get(k, 0) + max(0, w - g)
        mark = "ok " if sorted(got) == sorted(want) else "XX "
        print(f"  {mark} want {want or '-'!s:20s} got {got or '-'!s:20s} {redact(text)[0][:70]}")
    print("\n  type   precision recall   (on these labeled strings)")
    for k in sorted(tp):
        p = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else float("nan")
        r = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else float("nan")
        print(f"  {k:6s} {p:6.2f}   {r:6.2f}")
    counts, examples = {}, {}
    for row in R.rows:
        for k, s, e, m in find_pii(row["text"]):
            counts[k] = counts.get(k, 0) + 1
            examples.setdefault(k, []).append((row["id"], m))
    n_chunks = sum(bool(find_pii(r["text"])) for r in R.rows)
    print(f"\n  on the corpus ({len(R.rows)} chunks of public 10-K text): {n_chunks} chunks contain a PII-shaped match")
    for k, c in sorted(counts.items(), key=lambda x: -x[1]):
        ex = "; ".join(f"{i}: '{m}'" for i, m in examples[k][:3])
        print(f"    {k:6s} {c:4d}   e.g. {ex}")
    print("  -> redact user INPUT and logs, not the corpus: these are public (HQ phone numbers, filer details)")


def section_access(R) -> None:
    print("\n" + "=" * 30 + " access control (common/access.py, service.ScopedRetriever)")
    user = User("analyst-aapl", ("AAPL",))
    qs = [x["question"] for x in retrieval5.golden("dev")][:30]
    scoped = service.ScopedRetriever(R, user)
    leaked = empty = 0
    for q in qs:
        hits = scoped.retrieve(q, k=6, rerank=False)
        leaked += sum(h["meta"]["ticker"] != "AAPL" for h in hits)
        empty += not hits
    print(f"  pre-filter: {len(qs)} dev questions (all companies) as an AAPL-only user -> {leaked} non-AAPL chunks "
          f"returned, {empty} questions with no results")

    class Buggy(service.ScopedRetriever):  # a bug in the filter code: the scope is forgotten
        def retrieve(self, query, k=5, filters=None, **kw):
            return self.R.retrieve(query, k=k, filters=filters, **kw)
    buggy = Buggy(R, user)
    raw = [h for q in qs[:10] for h in buggy.retrieve(q, k=6, rerank=False)]
    bad = [h for h in raw if h["meta"]["ticker"] != "AAPL"]
    from common.access import violations
    caught = violations(raw, user)
    print(f"  simulated filter bug: {len(bad)}/{len(raw)} retrieved chunks are not AAPL; the post-check flags "
          f"{len(caught)} of them (dropped before generation, logged as access_violation)")

    from common.agent_tools import Evidence, make_tools
    for allowed in (None, {"AAPL"}):
        ev = Evidence()
        out = make_tools(service.ScopedRetriever(R, user), ev, allowed_tickers=allowed)["company_relations"].invoke(
            {"relation": "PARTNERS_WITH"})
        print(f"  agent graph tool, allowed_tickers={allowed}: {len(ev.hits)} evidence chunks, "
              f"{sum(h['meta']['ticker'] != 'AAPL' for h in ev.hits)} not AAPL   ({out.splitlines()[0][:60]})")

    c = ExactCache("v")
    c.put("What was Apple's revenue?", ("AAPL", "AMZN", "MSFT", "NVDA", "TSLA"), {"answer": "built from all filings"})
    print(f"  cache isolation: same question, other scope -> {c.get('What was Apple’s revenue?', ('AAPL',))}; "
          f"same scope -> {c.get('what was apple’s revenue', ('AAPL', 'AMZN', 'MSFT', 'NVDA', 'TSLA'))}")


def section_limits(R) -> None:
    print("\n" + "=" * 30 + " rate and spend limits (common/ratelimit.py)")
    for rpm in (60, 3):
        b = TokenBucket(rate=rpm / 60, capacity=max(1.0, rpm / 6))
        res = [b.take() for _ in range(15)]
        print(f"  rpm={rpm:3d}: 15 requests at once -> {sum(r == 0 for r in res)} allowed, then Retry-After "
              f"{min(r for r in res if r) if any(res) else 0:.1f} s")
    b = TokenBucket(rate=60 / 60, capacity=10)
    allowed = 0
    t_end = time.monotonic() + 3
    while time.monotonic() < t_end:  # a client hammering for 3 s: burst, then the refill rate
        allowed += b.take() == 0
        time.sleep(0.01)
    print(f"  rpm=60 for 3 s of continuous requests: {allowed} allowed (burst 10 + ~1/s refill)")

    traces = [t for t in load_traces("ask_v5") if t["attrs"].get("tag") == "eval:dev" and t["attrs"].get("index") == "structctx350"]
    errs, n = [], 0
    for t in traces:
        ctx = next((s for s in t["spans"] if s["name"] == "context"), None)
        gen = next((s for s in t["spans"] if s["name"] == "generate"), None)
        scan = next((s for s in t["spans"] if s["name"] == "injection_scan"), None)
        if not (ctx and gen and gen["attrs"].get("input_tokens")) or (scan and scan["attrs"].get("flagged")):
            continue
        hits = [{"id": i, "text": R.by_id[i]["text"], "meta": R._meta(i)} for i in ctx["attrs"]["ids"] if i in R.by_id]
        msgs = [{"role": "system", "content": ask5.SYSTEM},
                {"role": "user", "content": f"Context:\n\n{format_untrusted(hits)}\n\nQuestion: {t['attrs']['question']}"}]
        est, _ = estimate_cost("gpt-4o-mini", msgs, 0)
        errs.append((est - gen["attrs"]["input_tokens"]) / gen["attrs"]["input_tokens"])
        n += 1
    if errs:
        errs.sort()
        print(f"  pre-call prompt-token estimate vs actual (Phase 5 dev runs, n={n}): median error "
              f"{errs[len(errs) // 2] * 100:+.1f}%, range {errs[0] * 100:+.1f}% .. {errs[-1] * 100:+.1f}%")
        outs = sorted(s["attrs"]["output_tokens"] for t in traces for s in t["spans"]
                      if s["name"] == "generate" and s["attrs"].get("output_tokens"))
        print(f"  actual output tokens: median {outs[len(outs) // 2]}, p95 {outs[int(len(outs) * .95)]}, max {outs[-1]} "
              f"-> reserving max_tokens=600 overestimates the output cost ~{600 / outs[len(outs) // 2]:.0f}x (released after the call)")
    sp = SpendLimiter()
    msgs = [{"role": "user", "content": "x " * 1900}]  # ~1,900-token prompt, the typical Phase 5 size
    _, est = estimate_cost("gpt-4o-mini", msgs, 600)
    granted = 0
    while sp.reserve("trial", 0.002, est):
        sp.settle("trial", est, 0.00034)  # Phase 5's typical actual cost per answer
        granted += 1
    print(f"  $0.002/day user: estimate ${est:.5f}/request, actual ~$0.00034 -> {granted} requests granted "
          f"(the last reservation must still fit under the limit)")


def section_moderation() -> None:
    print("\n" + "=" * 30 + " moderation (common/moderation.py)")
    for text in ["What was Apple's total net sales in fiscal 2025?", "I want to hurt the people at Tesla's factory"]:
        for fail_open in (True, False):
            t = time.perf_counter()
            m = moderate(text, fail_open=fail_open)
            print(f"  fail_open={fail_open!s:5s} {time.perf_counter() - t:5.2f} s  flagged={m['flagged']!s:5s} "
                  f"source={m['source']:22s} {m['categories'][:3]}  '{text[:45]}'")
    print(f"  breaker: {BREAKERS['moderation'].status()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=["pii", "access", "limits", "moderation"])
    args = ap.parse_args()
    only = set(args.only or ["pii", "access", "limits", "moderation"])
    R = ask5.Index("structctx350").R
    if "pii" in only:
        section_pii(R)
    if "access" in only:
        section_access(R)
    if "limits" in only:
        section_limits(R)
    if "moderation" in only:
        section_moderation()


if __name__ == "__main__":
    main()
