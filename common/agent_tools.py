"""Tools for the Phase 6 agents (LangChain tools over our own retrieval code).

A tool = a Python function + a name + a description + a typed argument schema. The LLM never runs code: it emits
a structured tool call ({"name": "search_tables", "args": {...}}), OUR code validates and executes it, and the
result goes back to the LLM as a ToolMessage. So the description and the argument schema ARE the prompt for
when and how to use a tool: write them like documentation for a careful junior analyst.

Every passage a tool returns is registered in an Evidence list and numbered once ([1], [2], ...) for the whole run,
so the final answer can cite passages found by different tool calls, and Phase 1's citation validator and numeric
grounding check work on the agent's answer unchanged.

  search_filings     hybrid + rerank over the Phase 5 index (structctx350), optional company / fiscal-year filter
  search_tables      the same, restricted to table chunks (kind=table): financial statements, segment tables
  company_relations  Phase 5's entity-relation graph (competitors, suppliers, partners...), with source passages
  calculate          arithmetic on numbers from the passages (growth rates, differences), so the LLM doesn't do it
"""
import ast
import json
import operator
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from common.config import DATA_DIR

Ticker = Literal["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
K = 5  # passages per search call
RELATIONS = ["COMPETES_WITH", "SUPPLIED_BY", "CUSTOMER_IS", "PARTNERS_WITH", "ACQUIRED_OR_INVESTED_IN"]


class Evidence:
    """All passages the agent has seen in one run, numbered in order of first appearance."""

    def __init__(self):
        self.hits: list[dict] = []
        self._num: dict[str, int] = {}

    def add(self, hit: dict) -> int:
        if hit["id"] not in self._num:
            self.hits.append(hit)
            self._num[hit["id"]] = len(self.hits)
        return self._num[hit["id"]]

    def render(self, hits: list[dict], max_chars: int = 1800) -> str:
        if not hits:
            return "No passages found. Try other words, another filing year, or search_tables / search_filings."
        out = []
        for h in hits:
            n = self.add(h)
            m = h["meta"]
            out.append(f"[{n}] {m['ticker']} 10-K FY{m['fiscal_year']}, {m['item']} ({h['id']})\n{h['text'][:max_chars]}")
        return "\n\n".join(out)


class SearchArgs(BaseModel):
    query: str = Field(description="What to look for, in the words a 10-K would use (e.g. 'total net sales', "
                                   "'purchases of property and equipment', 'employees')")
    company: Ticker | None = Field(None, description="Restrict to one company's filings")
    fiscal_years: list[int] | None = Field(None, description="Restrict to these FILING fiscal years. Each 10-K also "
                                                             "reports the 1-2 prior years, so FY2024 data is in the FY2024 and FY2025 filings")
    # No `k` argument on purpose: the first version exposed k (1-8) and gpt-4o-mini chose k=1 on every call, got the
    # wrong table for NVIDIA, then fed invented numbers to calculate. Don't give the LLM a knob it will misuse.


class RelationArgs(BaseModel):
    company: Literal["Apple", "Microsoft", "NVIDIA", "Tesla", "Amazon"] | None = Field(
        None, description="The filer whose relationships to list; omit for all five")
    relation: Literal["COMPETES_WITH", "SUPPLIED_BY", "CUSTOMER_IS", "PARTNERS_WITH", "ACQUIRED_OR_INVESTED_IN"] | None = None
    other: str | None = Field(None, description="Only edges whose other party matches this name (e.g. 'OpenAI', 'TSMC')")


class CalcArgs(BaseModel):
    expression: str = Field(description="Arithmetic only: numbers, + - * / ** ( ), e.g. '(716924 / 637959 - 1) * 100'")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic by walking the AST: never eval(), which would run any Python the LLM (or an injected
    passage) wrote."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:  # 9**9**9 would hang the process: a DoS via the LLM
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"not allowed: {ast.dump(node)[:60]}")
    return ev(ast.parse(expr.replace(",", ""), mode="eval").body)


def make_tools(R, evidence: Evidence, trace=None) -> dict[str, StructuredTool]:
    """Tools bound to one retriever and one run's Evidence. `trace` (common.trace.Trace) gets a span per call."""
    from contextlib import nullcontext
    span = trace.span if trace else (lambda *a, **kw: nullcontext({}))
    edges = json.loads((DATA_DIR / "processed" / "graph_edges.json").read_text())

    def _search(query: str, company=None, fiscal_years=None, k: int = 5, tables_only: bool = False) -> str:
        f = {**({"ticker": [company]} if company else {}), **({"fiscal_year": fiscal_years} if fiscal_years else {}),
             **({"kind": ["table"]} if tables_only else {})} or None
        name = "search_tables" if tables_only else "search_filings"
        with span(f"tool:{name}", query=query, filters=f, k=k) as sp:
            hits = R.retrieve(query, k=k, filters=f)
            out = evidence.render(hits)
            sp.update(ids=[h["id"] for h in hits], top_score=round(hits[0]["score"], 2) if hits else None)
        return out

    def search_filings(query: str, company=None, fiscal_years=None) -> str:
        return _search(query, company, fiscal_years, K)

    def search_tables(query: str, company=None, fiscal_years=None) -> str:
        return _search(query, company, fiscal_years, K, tables_only=True)

    def company_relations(company=None, relation=None, other=None) -> str:
        with span("tool:company_relations", company=company, relation=relation, other=other) as sp:
            found = [e for e in edges if (not company or e["subject"] == company) and (not relation or e["relation"] == relation)
                     and (not other or other.lower() in (e["object"] + " " + e["raw"]).lower())]
            sp["n_edges"] = len(found)
            if not found:
                return "No matching relationships in the graph (it covers Item 1 / 1A of each company's latest 10-K only)."
            lines = []
            for e in found[:25]:
                n = evidence.add({"id": e["chunk"], "text": R.by_id[e["chunk"]]["text"], "meta": R._meta(e["chunk"]), "score": 0.0})
                lines.append(f"{e['subject']} {e['relation']} {e['object']}   source [{n}]: \"{e['evidence']}\"")
            more = f"\n(+{len(found) - 25} more; filter by company or relation)" if len(found) > 25 else ""
            return "Relationships extracted from the filings (each with its source passage):\n" + "\n".join(lines) + more

    def calculate(expression: str) -> str:
        with span("tool:calculate", expression=expression) as sp:
            try:
                v = safe_eval(expression)
                sp["result"] = v
                return f"{expression} = {v:,.4f}".rstrip("0").rstrip(".")
            except (ValueError, SyntaxError, ZeroDivisionError) as e:
                sp["error"] = str(e)
                return f"Error: {e}. Use plain arithmetic on numbers."

    return {
        "search_filings": StructuredTool.from_function(
            search_filings, name="search_filings", args_schema=SearchArgs,
            description="Search the 10-K filings (Apple, Microsoft, NVIDIA, Tesla, Amazon; two fiscal years each) for "
                        "passages and tables. Returns numbered passages [n] to cite. Use one company per call."),
        "search_tables": StructuredTool.from_function(
            search_tables, name="search_tables", args_schema=SearchArgs,
            description="Like search_filings but returns TABLES only (income statements, balance sheets, cash flows, "
                        "segment tables). Use it for exact reported figures, e.g. when prose only gives a rounded or "
                        "differently defined number (net vs gross, segment vs total)."),
        "company_relations": StructuredTool.from_function(
            company_relations, name="company_relations", args_schema=RelationArgs,
            description="Look up relationships the filings name between the five companies and other organizations "
                        "(competitors, suppliers, customers, partners, investments), across ALL five filings at once. "
                        "Use it for 'which companies ...' questions; every edge comes with its source passage [n]."),
        "calculate": StructuredTool.from_function(
            calculate, name="calculate", args_schema=CalcArgs,
            description="Compute arithmetic exactly (growth %, differences, ratios). Always use it instead of doing math yourself."),
    }
