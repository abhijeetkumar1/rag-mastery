"""Operations dashboard from the service's traces: a static HTML page (data/dashboard.html) plus a text summary.

Run: uv run python -m phase7_production.07_dashboard                    # all ask_v7 traces
     uv run python -m phase7_production.07_dashboard --exclude-tag loadtest
     uv run python -m phase7_production.07_dashboard --since 2026-10-15

What a RAG service dashboard shows, and why each one:
  volume, latency p50/p95, time to first token     is it up, is it fast (users feel p95 and TTFT, not the mean)
  latency per stage                                where the time goes (what to optimize or cache)
  cost per request, total                          is spend on budget; which stage spends it
  outcomes: answered / refused (by reason) / error a refusal spike = a broken router or retriever, not "safer"
  output guardrail pass rate                       ungrounded numbers trending up = a model or index regression
  degraded rate, by dependency                     fallbacks hide outages from users, so they must show up here
  cache hit rate                                   is the cache earning its memory
  score drift (PSI of the best rerank score)       traffic or retrieval changing vs the evaluated baseline
  feedback: thumbs-down rate + the traces          the cheapest label source: thumbs-down -> golden-set candidates
  online eval (judge scores on sampled traffic)    08_online_eval.py; faithfulness on real questions, not the golden set
Production: Prometheus (GET /metrics) + Grafana for the time series, the trace backend (Phoenix, Langfuse) for drill-
down. This page is the same numbers computed from the JSONL traces, so it works with no infrastructure.
"""
import argparse
import html
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from common.config import DATA_DIR
from common.trace import load_traces

OUT = DATA_DIR / "dashboard.html"
FEEDBACK = DATA_DIR / "feedback" / "feedback.jsonl"
SCORES = DATA_DIR / "online_eval" / "scores.jsonl"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def psi(baseline: list[float], current: list[float], bins: int = 10) -> float:
    """Population Stability Index over the baseline's deciles. < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large."""
    if len(baseline) < 20 or len(current) < 20:
        return float("nan")
    edges = np.unique(np.percentile(baseline, np.linspace(0, 100, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.histogram(baseline, edges)[0] / len(baseline)
    c = np.histogram(current, edges)[0] / len(current)
    b, c = np.clip(b, 1e-4, None), np.clip(c, 1e-4, None)
    return float(((c - b) * np.log(c / b)).sum())


def compute(traces: list[dict], baseline: list[float], feedback: list[dict], scores: list[dict]) -> dict:
    n = len(traces)
    lat = [t["duration_ms"] / 1000 for t in traces]
    ttft = [t["attrs"]["ttft_ms"] / 1000 for t in traces if t["attrs"].get("ttft_ms") is not None]
    outcomes = Counter(t["attrs"].get("outcome", "answered") for t in traces)
    answered = [t for t in traces if t["attrs"].get("outcome") == "answered"]
    fresh = [t for t in answered if (t["attrs"].get("cache") or "miss") == "miss"]
    stages: dict[str, list[float]] = {}
    for t in traces:
        for s in t["spans"]:
            stages.setdefault(s["name"], []).append(s.get("duration_ms", 0))
    degraded = Counter(d for t in traces for d in (t["attrs"].get("degraded") or []))
    cache = Counter(t["attrs"].get("cache") or "miss" for t in traces)
    best = [t["attrs"]["best_rerank"] for t in answered if isinstance(t["attrs"].get("best_rerank"), (int, float))
            and t["attrs"].get("cache") == "miss"]
    by_id = {t["trace_id"]: t for t in traces}
    fb = Counter(f["rating"] for f in feedback)
    return {
        "n": n, "days": sorted({t["started_at"][:10] for t in traces}),
        "tags": Counter(t["attrs"].get("tag") or "-" for t in traces),
        "p50": pct(lat, 50), "p95": pct(lat, 95), "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95),
        "cost_total": sum(t["attrs"].get("cost_usd") or 0 for t in traces),
        "outcomes": outcomes,
        "refusal_rate": sum(v for k, v in outcomes.items() if k.startswith("refused")) / n if n else float("nan"),
        "error_rate": outcomes.get("error", 0) / n if n else float("nan"),
        # cache replays repeat a stored verdict: count fresh answers only (old replay traces also lack the field)
        "guard_pass": (sum(bool(t["attrs"].get("guardrails_passed")) for t in fresh) / len(fresh)) if fresh else float("nan"),
        "degraded_rate": sum(bool(t["attrs"].get("degraded")) for t in traces) / n if n else float("nan"),
        "degraded": degraded, "cache": cache,
        "cache_hit": (n - cache.get("miss", 0)) / n if n else float("nan"),
        "stages": {k: (pct(v, 50), pct(v, 95), len(v)) for k, v in stages.items()},
        "best": best, "baseline": baseline, "psi": psi(baseline, best),
        "feedback": fb, "down": [(f, by_id.get(f["trace_id"])) for f in feedback if f["rating"] == "down"],
        "online": scores,
    }


# ------------------------------------------------------------------ HTML (inline SVG, no external libraries)
CSS = """
:root{--surface-0:#f4f3f0;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#77766f;
--grid:#e4e3de;--series-1:#2a78d6;--series-2:#eb6834;--border:#e4e3de}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface-0:#121211;--surface-1:#1a1a19;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#9a998f;--grid:#2c2c2a;--series-1:#3987e5;--series-2:#d95926;--border:#2c2c2a}}
:root[data-theme="dark"]{--surface-0:#121211;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;
--text-muted:#9a998f;--grid:#2c2c2a;--series-1:#3987e5;--series-2:#d95926;--border:#2c2c2a}
*{box-sizing:border-box}body{margin:0;background:var(--surface-0);color:var(--text-primary);
font:14px/1.45 system-ui,-apple-system,sans-serif}main{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 12px}.sub{color:var(--text-secondary);margin:0 0 20px;overflow-wrap:anywhere}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(150px,100%),1fr));gap:12px;margin-bottom:20px}
.tile,.card{min-width:0;background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px}
.tile .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}.tile .l{color:var(--text-secondary);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(340px,100%),1fr));gap:16px;margin-bottom:16px}
svg{width:100%;height:auto;display:block}svg text{fill:var(--text-secondary);font-size:11px}
.bar{fill:var(--series-1)}.bar2{fill:var(--series-2)}.bar:hover,.bar2:hover{opacity:.8}
table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
td,th{text-align:left;padding:4px 6px;border-bottom:1px solid var(--border)}th{color:var(--text-secondary);font-weight:500}
details{margin-top:8px;color:var(--text-secondary)}summary{cursor:pointer;font-size:12px}
.legend{display:flex;gap:14px;font-size:12px;color:var(--text-secondary);margin-bottom:6px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);
border-radius:6px;padding:6px 8px;font-size:12px;display:none;box-shadow:0 2px 8px rgba(0,0,0,.15)}
.note{color:var(--text-muted);font-size:12px;margin-top:8px}
"""
JS = """const tip=document.getElementById('tip');document.querySelectorAll('[data-tip]').forEach(el=>{
el.addEventListener('mousemove',e=>{tip.style.display='block';tip.textContent=el.dataset.tip;
tip.style.left=Math.min(e.clientX+12,innerWidth-220)+'px';tip.style.top=(e.clientY+12)+'px'});
el.addEventListener('mouseleave',()=>tip.style.display='none')});"""


def esc(x) -> str:
    return html.escape(str(x))


def hbars(rows: list[tuple[str, float, str]], unit: str, cls: str = "bar") -> str:
    """Horizontal bars: (label, value, tooltip). One series, labels left, value at the bar end."""
    if not rows:
        return "<p class='note'>no data</p>"
    w, lh, left = 560, 22, 200
    vmax = max(v for _, v, _ in rows) or 1
    h = lh * len(rows) + 6
    parts = [f"<svg viewBox='0 0 {w} {h}' role='img'>"]
    for i, (lab, v, tipt) in enumerate(rows):
        y = i * lh + 3
        bw = max(2, (w - left - 70) * v / vmax)
        short = lab if len(lab) <= 30 else lab[:29] + "…"
        parts.append(f"<text x='{left - 8}' y='{y + 13}' text-anchor='end'>{esc(short)}</text>"
                     f"<rect class='{cls}' x='{left}' y='{y + 3}' width='{bw:.1f}' height='{lh - 8}' rx='3' data-tip='{esc(tipt)}'/>"
                     f"<text x='{left + bw + 6}' y='{y + 13}'>{v:,.{0 if float(v).is_integer() or v >= 100 else 1}f}{unit}</text>")
    return "".join(parts) + "</svg>"


def histogram(a: list[float], b: list[float], labels: tuple[str, str]) -> str:
    """Two overlaid distributions as side-by-side bars per bin (shares, so different sample sizes compare)."""
    if not a or not b:
        return "<p class='note'>not enough data</p>"
    lo, hi = math.floor(min(a + b)), math.ceil(max(a + b))
    edges = np.linspace(lo, hi, 13)
    ha, hb = np.histogram(a, edges)[0] / len(a), np.histogram(b, edges)[0] / len(b)
    w, h, pad = 520, 180, 28
    bw = (w - 2 * pad) / 12
    vmax = max(ha.max(), hb.max()) or 1
    parts = [f"<svg viewBox='0 0 {w} {h + 24}' role='img'>",
             f"<line x1='{pad}' x2='{w - pad}' y1='{h}' y2='{h}' stroke='var(--grid)'/>"]
    for i in range(12):
        x = pad + i * bw
        for j, (vals, cls, name) in enumerate(((ha, "bar", labels[0]), (hb, "bar2", labels[1]))):
            bh = (h - 10) * vals[i] / vmax
            parts.append(f"<rect class='{cls}' x='{x + 2 + j * (bw - 4) / 2:.1f}' y='{h - bh:.1f}' width='{(bw - 6) / 2:.1f}' "
                         f"height='{max(bh, 0):.1f}' rx='2' data-tip='{esc(name)}: {vals[i] * 100:.0f}% of requests in "
                         f"[{edges[i]:.1f}, {edges[i + 1]:.1f})'/>")
        if i % 2 == 0:
            parts.append(f"<text x='{x}' y='{h + 16}'>{edges[i]:.0f}</text>")
    return "".join(parts) + "</svg>"


def table(head: list[str], rows: list[list]) -> str:
    return ("<details><summary>table view</summary><table><tr>" + "".join(f"<th>{esc(x)}</th>" for x in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in r) + "</tr>" for r in rows) + "</table></details>")


def render(m: dict) -> str:
    f = m["feedback"]
    n_fb = sum(f.values())
    kpis = [("requests", f"{m['n']}"), ("latency p50 / p95 (s)", f"{m['p50']:.2f} / {m['p95']:.2f}"),
            ("first token p95", f"{m['ttft_p95']:.2f} s"), ("cost total", f"${m['cost_total']:.4f}"),
            ("refusal rate", f"{m['refusal_rate'] * 100:.0f}%"), ("errors", f"{m['error_rate'] * 100:.1f}%"),
            ("guardrails passed (fresh)", f"{m['guard_pass'] * 100:.0f}%"), ("degraded", f"{m['degraded_rate'] * 100:.0f}%"),
            ("cache hit rate", f"{m['cache_hit'] * 100:.0f}%"),
            ("score drift (PSI)", "n/a" if math.isnan(m["psi"]) else f"{m['psi']:.2f}"),
            ("thumbs down", f"{f.get('down', 0)}/{n_fb}" if n_fb else "no feedback")]
    stages = sorted(m["stages"].items(), key=lambda kv: -kv[1][0])
    stage_rows = [(k, p50, f"{k}: p50 {p50:.1f} ms, p95 {p95:.1f} ms, {cnt} spans") for k, (p50, p95, cnt) in stages if p50 >= 0.05]
    out_rows = [(k, v, f"{k}: {v} requests ({v / m['n'] * 100:.0f}%)") for k, v in m["outcomes"].most_common()]
    deg_rows = [(k, v, f"{k}: {v} requests served with a fallback") for k, v in m["degraded"].most_common()]
    psi_note = ("stable" if m["psi"] < 0.1 else "moderate shift" if m["psi"] < 0.25 else "LARGE shift") if not math.isnan(m["psi"]) else ""
    down = "".join(f"<tr><td>{esc(fb['trace_id'])}</td><td>{esc((t or {}).get('attrs', {}).get('question', '?'))[:120]}</td>"
                   f"<td>{esc(fb.get('comment', ''))}</td></tr>" for fb, t in m["down"][:20])
    online = m["online"]
    onl = (f"{len(online)} sampled answers judged; mean faithfulness {np.mean([s['faithfulness'] for s in online]):.2f}"
           if online else "no online-eval scores yet (08_online_eval.py)")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG Service Dashboard</title><style>{CSS}</style></head><body><main>
<h1>RAG service dashboard</h1><p class="sub">{m['n']} requests (ask_v7) on {', '.join(m['days'])} · tags {dict(m['tags'])}</p>
<div class="kpis">{''.join(f"<div class='tile'><div class='v'>{esc(v)}</div><div class='l'>{esc(l)}</div></div>" for l, v in kpis)}</div>
<div class="grid">
<div class="card"><h2>Latency per stage (p50, ms)</h2>{hbars(stage_rows, ' ms')}
{table(['stage', 'p50 ms', 'p95 ms', 'spans'], [[k, f'{a:.1f}', f'{b:.1f}', c] for k, (a, b, c) in stages])}</div>
<div class="card"><h2>Outcomes</h2>{hbars(out_rows, '')}
{table(['outcome', 'requests'], [[k, v] for k, v, _ in out_rows])}
<h2 style="margin-top:16px">Degraded requests, by dependency</h2>{hbars(deg_rows, '') if deg_rows else "<p class='note'>none</p>"}</div>
</div>
<div class="grid">
<div class="card"><h2>Score drift: best rerank score per answered request</h2>
<div class="legend"><span><span class="sw" style="background:var(--series-1)"></span>baseline: Phase 5 dev eval (n={len(m['baseline'])})</span>
<span><span class="sw" style="background:var(--series-2)"></span>this traffic, cache misses (n={len(m['best'])})</span></div>
{histogram(m['baseline'], m['best'], ('baseline', 'this traffic'))}
<p class="note">PSI {('n/a' if math.isnan(m['psi']) else f"{m['psi']:.2f}")} {psi_note}. A shift means the questions changed, or retrieval did
(a degraded dense search shows up here before anyone complains).</p></div>
<div class="card"><h2>Feedback and online evaluation</h2><p>{n_fb} ratings: {f.get('up', 0)} up, {f.get('down', 0)} down.</p>
<p>{esc(onl)}</p><h2 style="margin-top:12px">Thumbs-down (golden-set candidates)</h2>
{f"<table><tr><th>trace</th><th>question</th><th>comment</th></tr>{down}</table>" if down else "<p class='note'>none</p>"}</div>
</div></main><div id="tip"></div><script>{JS}</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-tag")
    ap.add_argument("--since")
    args = ap.parse_args()
    traces = [t for t in load_traces("ask_v7") if (not args.exclude_tag or t["attrs"].get("tag") != args.exclude_tag)
              and (not args.since or t["started_at"] >= args.since)]
    if not traces:
        raise SystemExit("no ask_v7 traces yet: run the service first")
    base = [s["attrs"]["best_score"] for t in load_traces("ask_v5")
            if t["attrs"].get("tag") == "eval:dev" and t["attrs"].get("index") == "structctx350"
            for s in t["spans"] if s["name"] == "relevance_floor" and isinstance(s["attrs"].get("best_score"), (int, float))]
    m = compute(traces, base, jsonl(FEEDBACK), jsonl(SCORES))
    OUT.write_text(render(m))
    print(f"{m['n']} requests  latency p50 {m['p50']:.2f} s p95 {m['p95']:.2f} s  first token p50 {m['ttft_p50']:.2f} s "
          f"p95 {m['ttft_p95']:.2f} s  cost ${m['cost_total']:.4f}")
    print(f"outcomes {dict(m['outcomes'])}\nrefusal {m['refusal_rate']:.0%}  errors {m['error_rate']:.1%}  guardrails passed "
          f"{m['guard_pass']:.0%}  degraded {m['degraded_rate']:.0%} {dict(m['degraded'])}  cache hits {m['cache_hit']:.0%} {dict(m['cache'])}")
    print(f"score drift: PSI {m['psi']:.3f} (baseline n={len(base)}, current n={len(m['best'])}; best rerank median "
          f"{np.median(base):.2f} -> {np.median(m['best']) if m['best'] else float('nan'):.2f})")
    print("slowest stages (p50 ms): " + ", ".join(f"{k} {v[0]:.0f}" for k, v in sorted(m["stages"].items(), key=lambda kv: -kv[1][0])[:5]))
    print(f"feedback {dict(m['feedback'])}\nwrote {OUT}")


if __name__ == "__main__":
    main()
