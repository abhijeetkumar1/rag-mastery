"""GraphRAG, intro: extract an entity-relation graph from the filings, then answer questions that need the WHOLE corpus
("which companies name each other as competitors?"), which top-k chunk retrieval can't aggregate.

Run: uv run python -m phase5_advanced_indexing.08_graphrag            # ~370 gpt-4o-mini calls first time (~$0.10), cached
     uv run python -m phase5_advanced_indexing.08_graphrag --no-rag   # skip the vector-RAG comparison

Steps:
  1. extract   each Item 1 / 1A chunk of the latest filings -> typed triples (Microsoft, COMPETES_WITH, Google),
               with the source chunk id kept on every edge (provenance = citations for graph answers)
  2. resolve   entity resolution: "Alphabet", "Google LLC", "Google" -> one node (rules + an alias table)
  3. query     graph traversal in code: exact, exhaustive over the corpus, explainable
  4. compare   the same questions through 06_ask (vector RAG, k=6)

This is the "local" half of Microsoft's GraphRAG (2024) without its expensive part: GraphRAG also clusters the graph
into communities (Leiden) and has an LLM summarize each one, to answer "global" questions ("what are the main themes?").
"""
import argparse
import importlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from common.config import CHAT_MODEL, DATA_DIR
from common.llm import chat_json, client
from common.trace import cost_usd

ask5 = importlib.import_module("phase5_advanced_indexing.06_ask")
LATEST = {"AAPL": 2025, "MSFT": 2026, "NVDA": 2026, "TSLA": 2025, "AMZN": 2025}
COVERED = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "TSLA": "Tesla", "AMZN": "Amazon"}
RELATIONS = ["COMPETES_WITH", "SUPPLIED_BY", "CUSTOMER_IS", "PARTNERS_WITH", "ACQUIRED_OR_INVESTED_IN"]

OBJECT_TYPES = ["company", "product_or_service", "country_or_region", "generic_group", "government_or_regulator", "other"]
SCHEMA = {"type": "object", "properties": {"triples": {"type": "array", "items": {
    "type": "object", "properties": {"object": {"type": "string"}, "object_type": {"type": "string", "enum": OBJECT_TYPES},
                                     "part_of_filer": {"type": "boolean"},
                                     "relation": {"type": "string", "enum": RELATIONS}, "evidence": {"type": "string"}},
    "required": ["object", "object_type", "part_of_filer", "relation", "evidence"], "additionalProperties": False}}},
    "required": ["triples"], "additionalProperties": False}
SYSTEM = f"""You extract business relationships from a passage of a company's 10-K filing. The filer is the subject of
every relationship. Return the OTHER companies or organizations the passage explicitly names, with one relation each:
  COMPETES_WITH            the filer names it as a competitor
  SUPPLIED_BY              the filer buys from / depends on it (manufacturer, supplier, foundry)
  CUSTOMER_IS              it buys from the filer
  PARTNERS_WITH            a named partnership, collaboration or strategic agreement
  ACQUIRED_OR_INVESTED_IN  the filer acquired it or invested in it
object_type: what the object is ("company" only for a named business, e.g. not "hyperscalers", "Windows" or "Taiwan").
part_of_filer: true if the object is the filer's own subsidiary, segment, brand or product (e.g. LinkedIn for Microsoft).
evidence: the exact words that state the relationship (max 25 words). If none, return an empty list."""

# Entity resolution: the same organization under different names. Rules first (strip legal suffixes), then aliases.
ALIASES = {"alphabet": "Google", "google": "Google", "google cloud": "Google", "amazon web services": "Amazon",
           "aws": "Amazon", "amazon": "Amazon", "microsoft": "Microsoft", "azure": "Microsoft", "apple": "Apple",
           "nvidia": "NVIDIA", "tesla": "Tesla", "taiwan semiconductor manufacturing company": "TSMC", "taiwan semiconductor manufacturing": "TSMC", "tsmc": "TSMC",
           "meta platforms": "Meta", "meta": "Meta", "facebook": "Meta", "advanced micro devices": "AMD", "amd": "AMD",
           "intel": "Intel", "openai": "OpenAI", "samsung": "Samsung", "samsung electronics": "Samsung",
           "huawei": "Huawei", "byd": "BYD", "hon hai precision industry": "Foxconn", "foxconn": "Foxconn",
           "contemporary amperex technology": "CATL", "space exploration technologies": "SpaceX", "xai": "xAI", "salesforce": "Salesforce", "oracle": "Oracle", "ibm": "IBM"}


def resolve(name: str) -> str:
    n = re.sub(r"\s*\([^)]*\)", "", re.sub(r"[.,]", "", name)).strip()  # "Contemporary Amperex ... (CATL)" -> no parens
    while True:  # "Samsung Electronics Co Ltd": strip legal suffixes repeatedly
        m = re.sub(r"\s+(inc|incorporated|corp|corporation|company|co|ltd|llc|plc|limited|holdings)$", "", n, flags=re.I)
        if m == n:
            break
        n = m
    return ALIASES.get(n.lower(), n)


def extract(rows: list[dict]) -> tuple[list[dict], dict]:
    client()

    def one(r):
        return chat_json([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"Filer: {COVERED[r['ticker']]}\n\n{r['body']}"}], SCHEMA, name="triples")

    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(one, rows))
    edges, dropped = [], defaultdict(int)
    for r, (out, _) in zip(rows, results):
        for t in out["triples"]:
            # v1 of this script had no object_type / part_of_filer: the graph filled with "competitors", "Windows",
            # "Taiwan", and "Microsoft COMPETES_WITH LinkedIn" (its own subsidiary). The schema makes the LLM classify,
            # the code filters: the same "extract attributes, decide in code" pattern as Phase 4's judge v2.
            if t["object_type"] == "company" and t["object"][:1].islower():  # "contract manufacturers" typed as a company
                t["object_type"] = "generic_group"
            if t["object_type"] != "company" or t["part_of_filer"]:
                dropped[t["object_type"] if t["object_type"] != "company" else "part_of_filer"] += 1
                continue
            obj = resolve(t["object"])
            if obj != COVERED[r["ticker"]]:  # a company "competing with itself" (segment names) is noise
                edges.append({"subject": COVERED[r["ticker"]], "relation": t["relation"], "object": obj,
                              "evidence": t["evidence"], "chunk": r["id"], "raw": t["object"]})
    usage = {"in": sum(u["input_tokens"] for _, u in results), "out": sum(u["output_tokens"] for _, u in results),
             "cached": sum(u["cached"] for _, u in results), "dropped": dict(dropped)}
    return edges, usage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rag", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(line) for line in (DATA_DIR / "processed" / "chunks_structctx350.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["fiscal_year"] == LATEST[r["ticker"]] and r["item"] in ("Item 1", "Item 1A")]
    edges, u = extract(rows)
    print(f"extracted {len(edges)} edges from {len(rows)} chunks ({u['cached']} cached calls), "
          f"{u['in']} in / {u['out']} out tokens = ${cost_usd(CHAT_MODEL, u['in'], u['out']) or 0:.3f}")
    print(f"dropped (not a company, or part of the filer): {u['dropped']}")
    nodes = {e["object"] for e in edges} | {e["subject"] for e in edges}
    raw_names = {e["raw"] for e in edges}
    print(f"graph: {len(nodes)} nodes ({len(raw_names)} raw names before entity resolution), "
          + ", ".join(f"{r} {sum(e['relation'] == r for e in edges)}" for r in RELATIONS))
    (DATA_DIR / "processed" / "graph_edges.json").write_text(json.dumps(edges, indent=1))

    out = defaultdict(lambda: defaultdict(set))  # subject -> relation -> {object: chunks}
    prov = defaultdict(set)
    for e in edges:
        out[e["subject"]][e["relation"]].add(e["object"])
        prov[(e["subject"], e["relation"], e["object"])].add(e["chunk"])

    print("\nQ1. Which of the five covered companies name each other as competitors?  (needs all 5 filings)")
    names = list(COVERED.values())
    for a in names:
        for b in names:
            if a < b and (b in out[a]["COMPETES_WITH"] or a in out[b]["COMPETES_WITH"]):
                way = "both ways" if b in out[a]["COMPETES_WITH"] and a in out[b]["COMPETES_WITH"] else (
                    f"{a} names {b}" if b in out[a]["COMPETES_WITH"] else f"{b} names {a}")
                src = sorted(prov[(a, "COMPETES_WITH", b)] | prov[(b, "COMPETES_WITH", a)])
                print(f"  {a} — {b}: {way}   sources {src[:3]}")

    print("\nQ2. Which outside companies are named as competitors by 2+ of the covered companies?  (aggregation)")
    named_by = defaultdict(set)
    for s in names:
        for o in out[s]["COMPETES_WITH"]:
            if o not in names:
                named_by[o].add(s)
    for o, s in sorted(named_by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(s) >= 2:
            print(f"  {o:20s} named by {sorted(s)}")

    print("\nQ3. Which covered companies name OpenAI, and how?  (relation types across filings)")
    for s in names:
        for rel in RELATIONS:
            if "OpenAI" in out[s][rel]:
                ev = next(e["evidence"] for e in edges if e["subject"] == s and e["relation"] == rel and e["object"] == "OpenAI")
                print(f"  {s} {rel} OpenAI   \"{ev[:90]}\"")

    print("\nSuppliers / foundries named (SUPPLIED_BY):")
    for s in names:
        if out[s]["SUPPLIED_BY"]:
            print(f"  {s:10s} {sorted(out[s]['SUPPLIED_BY'])[:10]}")

    if not args.no_rag:
        print("\n--- the same questions through vector RAG (06_ask, k=6) ---")
        ix = ask5.Index()
        ix.warmup()
        for q in ["Which of Apple, Microsoft, NVIDIA, Tesla and Amazon name each other as competitors in their 10-Ks?",
                  "Which companies are named as competitors by two or more of Apple, Microsoft, NVIDIA, Tesla and Amazon?"]:
            answer, hits, _ = ask5.ask(ix, q)
            print(f"\nQ: {q}\nA: {answer[:700]}\n   passages from: {sorted({h['meta']['ticker'] for h in hits})}")


if __name__ == "__main__":
    main()
