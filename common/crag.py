"""Corrective RAG (CRAG, Yan et al., 2024) + a Self-RAG-style answer check, as a LangGraph StateGraph (Phase 6).

Unlike ReAct, the control flow is FIXED by us; the LLM only fills in judgments at fixed points (grade, rewrite,
verify). It's predictable and cheaper than an open-ended agent, and each step can be evaluated on its own.

    START ─► plan ──(refuse)──► END
               │
               ▼
           retrieve ─► grade ──(every sub-question has an answering passage)──► generate ─► verify ──(ok)──► END
               ▲         │                                                        ▲          │
               │         └──(some missing, rounds left)──► rewrite ─┐             └(fix, 1×)─┘
               └────────────────────────────────────────────────────┘

  plan      Phase 3's router + decomposition (sub-questions with ticker / filing-year filters)
  retrieve  round 0: exactly Phase 5's retrieval (structctx350). Later rounds: only the sub-questions still missing
            evidence, with the rewritten query, tables-only if the rewriter asks, filing years widened by one
  grade     one LLM call per sub-question: per passage, is it relevant, and does it ANSWER the sub-question exactly
            (same entity, metric, period)? plus what's missing. The original CRAG uses a small fine-tuned T5 evaluator
            and falls back to WEB search; here the fallback is a different search of the same corpus
  rewrite   new query + table/text choice from what the grader said is missing
  generate  Phase 5's generator over the passages that passed grading (answering ones first)
  verify    deterministic output checks + an LLM check of the answer against the question (metric, entity, period);
            on failure, ONE regeneration with the reviewer's feedback
"""
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.config import GRADER_MODEL
from common.guardrails import check_answer, is_refusal
from common.trace import cost_usd

MAX_ROUNDS = 2      # corrective retrieval rounds after the first
MAX_FIXES = 1       # regenerations after a failed verification
REFUSAL = "I don't know based on the provided filings."


class PassageGrade(BaseModel):
    n: int
    relevant: bool = Field(description="about the sub-question's company, topic and period")
    answers: bool = Field(description="states the exact fact or figure asked: same entity (company total vs segment), "
                                      "same metric and qualifiers (gross vs net), and the asked period is shown")


class Grades(BaseModel):
    passages: list[PassageGrade]
    missing: str = Field(description="if no passage answers: what exactly is missing; else ''")


class Rewrite(BaseModel):
    query: str = Field(description="a new search query in 10-K wording, different from the ones tried")
    tables_only: bool = Field(description="true if the answer is a reported figure likely in a financial table")


class Verdict(BaseModel):
    answers_question: bool = Field(description="the answer addresses what was asked (every company / part)")
    metric_matches: bool = Field(description="each figure is the exact metric asked (total, not a segment; gross vs net as asked)")
    period_matches: bool = Field(description="each figure is for the period asked, as shown in the cited passage")
    supported: bool = Field(description="each claim is stated in, or correctly computed from, the cited passages")
    feedback: str = Field(description="if anything is false: what to fix and which passage [n] to use; else ''")


class CragState(TypedDict, total=False):
    question: str
    subs: list[dict]
    hits: dict            # sub-question -> list of hits
    grades: dict          # sub-question -> Grades (as dict)
    tried: dict           # sub-question -> queries tried
    rewrites: dict        # sub-question -> Rewrite (as dict)
    rounds: int
    fixes: int
    answer: str
    passages: list        # the hits given to the generator, in prompt order
    verdict: dict
    feedback: str
    refused_by: str


def _structured(schema, system: str, user: str, run: dict, span_name: str, **attrs):
    """One structured LLM call (strict JSON schema), with tokens and cost in our trace."""
    model = run.get("grader_model", GRADER_MODEL)
    llm = ChatOpenAI(model=model, temperature=0).with_structured_output(schema, method="json_schema", include_raw=True)
    with run["trace"].span(span_name, model=model, **attrs) as sp:
        out = llm.invoke([SystemMessage(system), HumanMessage(user)])
        u = out["raw"].usage_metadata or {}
        c = cost_usd(model, u.get("input_tokens", 0), u.get("output_tokens", 0)) or 0.0
        run["cost"] += c
        sp.update(input_tokens=u.get("input_tokens"), output_tokens=u.get("output_tokens"), cost_usd=c)
        sp["result"] = out["parsed"].model_dump()
    return out["parsed"]


def _numbered(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {h['meta']['ticker']} 10-K FY{h['meta']['fiscal_year']}, {h['meta']['item']}\n{h['text'][:1800]}"
                       for i, h in enumerate(hits, 1))


def plan_node(state: CragState, config) -> CragState:
    run = config["configurable"]["run"]
    refusal = run["route"](state["question"], run["trace"])  # Phase 3 router (input guardrail)
    if refusal:
        return {"answer": refusal, "refused_by": "router"}
    from common.query import plan
    p, _, _ = plan(state["question"])
    return {"subs": p["sub_questions"], "hits": {}, "grades": {}, "tried": {}, "rewrites": {}, "rounds": 0, "fixes": 0}


def retrieve_node(state: CragState, config) -> CragState:
    run = config["configurable"]["run"]
    hits, tried = dict(state["hits"]), dict(state["tried"])
    if state["rounds"] == 0:  # round 0 = Phase 5 exactly
        for h in run["retrieve"](state["question"], state["subs"], run["trace"]):
            hits.setdefault(h["sub_question"], []).append(h)
        for s in state["subs"]:
            hits.setdefault(s["question"], [])
            tried[s["question"]] = [s["question"]]
        return {"hits": hits, "tried": tried}
    R = run["R"]
    for s in state["subs"]:
        q = s["question"]
        if q not in state["rewrites"]:
            continue
        rw = state["rewrites"][q]
        years = sorted(set(s["filing_years"]) | {y + 1 for y in s["filing_years"]}) if s["filing_years"] else None
        f = {**({"ticker": [s["ticker"]]} if s["ticker"] else {}), **({"fiscal_year": years} if years else {}),
             **({"kind": ["table"]} if rw["tables_only"] else {})} or None
        with run["trace"].span("corrective_search", sub_question=q, query=rw["query"], filters=f) as sp:
            new = R.retrieve(rw["query"], k=4, filters=f)
            seen = {h["id"] for h in hits[q]}
            new = [{**h, "sub_question": q} for h in new if h["id"] not in seen]
            sp["new_ids"] = [h["id"] for h in new]
        hits[q] = hits[q] + new
        tried[q] = tried[q] + [rw["query"]]
    return {"hits": hits, "tried": tried}


GRADE_SYSTEM = """You grade search results for a question-answering system over SEC 10-K filings. For EACH numbered
passage decide: relevant (about the sub-question's company, topic and period) and answers (states the exact fact or
figure asked, with the same entity, metric and qualifiers, and shows the asked period, e.g. as a table column or
"fiscal year 2025"). Be strict about what a figure IS:
- a segment's figure does not answer a question about the company total;
- a figure with qualifiers the question doesn't have ("net of proceeds", "excluding", "adjusted") does not answer it;
- a figure that is PART of the asked item ("$X of employee termination expenses in restructuring and other") does not
  answer a question about the item itself; a differently named measure ("cash capital expenditures") does not answer a
  question about a specific line item ("purchases of property and equipment").
If no passage answers, say precisely what is missing (e.g. "the total 'Restructuring and other' line for 2024, likely
in an income statement or MD&A table")."""


def grade_node(state: CragState, config) -> CragState:
    run = config["configurable"]["run"]
    grades = dict(state["grades"])
    for s in state["subs"]:
        q = s["question"]
        if q in grades and any(p["answers"] for p in grades[q]["passages"]) and state["rounds"] > 0:
            continue  # already answered in an earlier round
        hs = state["hits"][q]
        if not hs:
            grades[q] = {"passages": [], "missing": "no passages retrieved"}
            continue
        g = _structured(Grades, GRADE_SYSTEM, f"Sub-question: {q}\n\nPassages:\n\n{_numbered(hs)}", run, "grade",
                        sub_question=q, n=len(hs), round=state["rounds"])
        grades[q] = g.model_dump()
    return {"grades": grades}


def _missing(state: CragState) -> list[str]:
    return [s["question"] for s in state["subs"] if not any(p["answers"] for p in state["grades"][s["question"]]["passages"])]


def after_grade(state: CragState) -> str:
    return "rewrite" if _missing(state) and state["rounds"] < MAX_ROUNDS else "generate"


REWRITE_SYSTEM = """The search results did not answer a sub-question about SEC 10-K filings. Write ONE new search query
(10-K wording, e.g. "net sales" for Apple/Amazon, "total revenues" for Tesla, "purchases of property and equipment"),
different from the queries already tried, aimed at what is missing. Set tables_only when the answer is a reported figure."""


def rewrite_node(state: CragState, config) -> CragState:
    run = config["configurable"]["run"]
    rewrites = {}
    for q in _missing(state):
        r = _structured(Rewrite, REWRITE_SYSTEM, f"Sub-question: {q}\nMissing: {state['grades'][q]['missing']}\n"
                                                 f"Queries tried: {state['tried'][q]}", run, "rewrite", sub_question=q)
        rewrites[q] = r.model_dump()
    return {"rewrites": rewrites, "rounds": state["rounds"] + 1}


def generate_node(state: CragState, config) -> CragState:
    """Passages that answer first, then relevant ones; irrelevant ones are dropped (CRAG's 'knowledge refinement')."""
    run = config["configurable"]["run"]
    passages = []
    for s in state["subs"]:
        q = s["question"]
        g = {p["n"]: p for p in state["grades"][q]["passages"]}
        hs = state["hits"][q]
        order = [h for i, h in enumerate(hs, 1) if g.get(i, {}).get("answers")] + \
                [h for i, h in enumerate(hs, 1) if g.get(i, {}).get("relevant") and not g.get(i, {}).get("answers")]
        passages += [h for h in order if h["id"] not in {p["id"] for p in passages}][:4]
    if not passages:
        return {"answer": REFUSAL, "passages": [], "refused_by": "crag:no_relevant_passages"}
    question = state["question"]
    if state.get("feedback"):
        question += f"\n\n(A reviewer checked a previous draft answer and said: {state['feedback']} Fix this.)"
    answer = run["generate"](question, passages, run["trace"])
    return {"answer": answer, "passages": passages}


VERIFY_SYSTEM = """You review an answer from a question-answering system over SEC 10-K filings, using the numbered
passages it cites. Check strictly: does it answer every part of the question; is each figure the EXACT metric asked
(company total vs a segment, gross vs net, the named line item); is each figure for the asked period (check the column
headers / dates in the passage); is each claim stated in, or correctly computed from, the passages? If anything fails,
say briefly what to fix and which passage to use."""


def verify_node(state: CragState, config) -> CragState:
    run = config["configurable"]["run"]
    if is_refusal(state["answer"]) or state.get("refused_by"):
        return {"verdict": {"skipped": "refusal"}}
    report = check_answer(state["answer"], [h["text"] for h in state["passages"]], tolerant=True)
    v = _structured(Verdict, VERIFY_SYSTEM, f"Question: {state['question']}\n\nPassages:\n\n{_numbered(state['passages'])}"
                                            f"\n\nAnswer to review:\n{state['answer']}", run, "verify", fix=state["fixes"])
    verdict = {**v.model_dump(), "guardrails_passed": report["passed"], "ungrounded": report["ungrounded_numbers"]}
    ok = v.answers_question and v.metric_matches and v.period_matches and v.supported and report["passed"]
    verdict["ok"] = ok
    feedback = v.feedback or (f"these numbers are not in the passages: {report['ungrounded_numbers']}" if not report["passed"] else "")
    return {"verdict": verdict, "feedback": "" if ok else feedback}


def after_verify(state: CragState) -> str:
    v = state.get("verdict", {})
    return "fix" if v.get("ok") is False and state["fixes"] < MAX_FIXES else END


def fix_node(state: CragState) -> CragState:
    return {"fixes": state["fixes"] + 1}


def build_graph():
    g = StateGraph(CragState)
    for name, fn in [("plan", plan_node), ("retrieve", retrieve_node), ("grade", grade_node), ("rewrite", rewrite_node),
                     ("generate", generate_node), ("verify", verify_node), ("fix", fix_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", lambda s: END if s.get("refused_by") else "retrieve", {"retrieve": "retrieve", END: END})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"rewrite": "rewrite", "generate": "generate"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify, {"fix": "fix", END: END})
    g.add_edge("fix", "generate")
    return g.compile()
