# Phase 6 walkthrough: agentic RAG

Read this top to bottom and run each command as you reach it. The theory, the interview Q&A and the experiments are in `NOTES.md`. All numbers are from real runs on 2026-10-01. Agents, graders and generation use `gpt-4o-mini` unless stated. The judges use `gpt-4.1`, and retrieval uses Phase 5's `structctx350` index.

**What this phase answers:** Phases 1–5 were **pipelines**: a fixed sequence (plan → retrieve → generate) chosen by us. Phase 5 left errors that a fixed sequence can't fix:
- **A wrong line item that was retrieved but misread:** Tesla's restructuring $583M is a component, the total is $684M.
- **Evidence that was never retrieved:** Amazon's gross capex table.
- **Questions that need several companies at once, arithmetic, or relations across filings.**

Phase 6 lets the LLM take part in control flow, in two degrees:
- **CRAG:** our graph, with LLM judgments at fixed points.
- **ReAct:** the LLM chooses every action.

It also adds the guardrails an agent needs (limits, allowlist, loop detection) and the observability it needs (step/tool trees in Arize Phoenix). Then it measures **when agents help and when they hurt**.

## Architecture

```
                          question
                             │
                 Phase 3 router (input guardrail: unsupported company, advice → refuse)
                             │
        ┌────────────────────┼─────────────────────────────────────────┐
        ▼                    ▼                                         ▼
  P5 (baseline)        CRAG (common/crag.py)                    ReAct (common/react_agent.py)
  plan→retrieve        plan → retrieve(P5) → grade ─┐           agent ⇄ tools (guarded) ──► check ──► answer
  →generate            ▲            (missing)       │             │ search_filings  search_tables
                       └── rewrite ◄────────────────┘             │ company_relations  calculate
                       generate → verify ─(fail, 1×)→ generate    └─(limit hit)─► finalize ──► answer
        │                    │                                         │
        └────────────────────┴──────── output guardrails (citations, numeric grounding) ─┘

 observability:  common/trace.py JSONL spans (cost, tokens, steps; what evaluation reads)
                 + Arize Phoenix (OpenInference/OpenTelemetry): the full node → LLM → tool tree, inputs and outputs
 guardrails:     AgentLimits/Budget (steps, tool calls, tokens, $), tool allowlist checked at EXECUTION,
                 loop detection (identical + near-duplicate calls), finalize-on-limit, output check → self-correction
```

### Components

| Layer | Module | Key pieces | Notes |
|---|---|---|---|
| Tools | `common/agent_tools.py` | `Evidence:32`, `SearchArgs:56`, `safe_eval:81`, `make_tools:98` | 4 LangChain `StructuredTool`s over our own retrieval |
| Limits | `common/agent_limits.py` | `AgentLimits:25`, `Budget.charge:53`, `.admit:61`, `.end_step:83`, `.exceeded:88` | Enforced in code, outside the LLM |
| ReAct | `common/react_agent.py` | `SYSTEM:33`, `agent_node:69`, `tools_node:81`, `finalize_node:103`, `check_node:123`, `build_graph:149` | Explicit `StateGraph` |
| CRAG | `common/crag.py` | `Grades:48`, `Verdict:58`, `retrieve_node:111`, `grade_node:153`, `rewrite_node:183`, `generate_node:193`, `verify_node:220`, `build_graph:243` | Fixed graph, LLM judgments |
| Tracing | `common/observability.py` | `setup_phoenix:30`, `flush:48` | No-op if no Phoenix server |
| Config | `common/config.py` | `GRADER_MODEL`, `PHOENIX_ENDPOINT`, `PHOENIX_TRACING` | `.env`: `OPENAI_GRADER_MODEL` |
| Scripts | `phase6_agentic_rag/` | `01_tools`, `02_react`, `03_crag`, `04_limits`, `05_end_to_end`, `multihop.py` | |
| Golden | `phase4_evaluation/01_build_golden.py` | `SPLITS["test3"]`, `TEST3_*` | Fresh split for this phase |

### Data model: what flows through the graphs

```
ReAct state (LangGraph)        {messages: [Human, AI(tool_calls), Tool, Tool, AI(...), ...], stop_reason, checked}
  per-run objects (NOT state)  config["configurable"]["run"] = {tools, allowed, budget: Budget, trace: Trace, evidence: Evidence}
CRAG state                     {question, subs, hits{sub_q: [hit]}, grades{sub_q: {passages: [{n, relevant, answers}], missing}},
                                tried{sub_q: [queries]}, rewrites, rounds, fixes, answer, passages, verdict, feedback, refused_by}
Tool call (from the LLM)       {"name": "search_tables", "args": {"query": "net sales", "company": "AMZN", "fiscal_years": [2025]}, "id": "call_…"}
Tool result (to the LLM)       "[7] AMZN 10-K FY2025, Item 8 (AMZN_FY2025_Item8_012)\n[Amazon.com, Inc. (AMZN) | …]\nTable: …"
Evidence                       hits numbered once per run, across calls → the answer's [n] index into Evidence.hits
```

Why per-run objects go in `config`, not in the state: LangGraph state is what nodes read and **reduce** (`add_messages` appends rather than overwrites). It should hold only what routing depends on, and stay serializable for checkpointing. A retriever, a trace or a budget object isn't state.

---

## Step 0: Setup (dependencies and Phoenix)

```bash
uv sync                                                    # adds langgraph, langchain-openai, arize-phoenix, openinference-…
PHOENIX_WORKING_DIR=data/phoenix uv run phoenix serve      # separate terminal; UI at http://localhost:6006
```

The installed versions (checked with `uv pip list`): `langgraph 1.2.12`, `langchain-core 1.6.6`, `langchain-openai 1.6.7`, `arize-phoenix 20.18.0`, `openinference-instrumentation-langchain 0.1.77`. Installing Phoenix **downgraded `websockets` 17.1 → 16.1.1**. I checked that Chroma still loads and queries. `requirements.txt` is regenerated with the `uv export` command in its header.

**`setup_phoenix(project)`** (`common/observability.py:30`):
- It checks with a socket probe whether a server listens on `PHOENIX_ENDPOINT` (default `http://localhost:6006`). If not, or if `PHOENIX_TRACING=0`, it does nothing, so no script depends on Phoenix.
- Otherwise it calls `phoenix.otel.register(project_name=…, endpoint=…/v1/traces, batch=True)` and `LangChainInstrumentor().instrument(…)`.
- **`flush()`** pushes buffered spans before a short script exits. Without it, the batch processor can drop the last spans.

**Gotcha, found in the first run:** I also instrumented the OpenAI SDK, so our own `chat_json` calls (the planner) would appear. `ChatOpenAI` calls the OpenAI SDK internally, so **every LLM call appeared twice** (22 `ChatOpenAI` + 24 `ChatCompletion` spans for 22 calls), and Phoenix's token totals double-counted. Now only LangChain is instrumented. The planner's calls are in our JSONL trace.

> ### 📘 Deep dive: Arize Phoenix and OpenInference
> **What it is.** Phoenix is an open-source LLM observability server: a web UI plus a trace store (SQLite by default; here `data/phoenix/phoenix.db`, 50 MB after this phase's runs). It receives **OpenTelemetry** spans over OTLP (HTTP `/v1/traces` or gRPC) and renders them as trees with inputs, outputs, tokens and latency per span. It also has datasets, experiments and LLM-judge evals.
>
> **OpenInference** is the semantic convention on top of OTel for LLM apps. Span kinds are `CHAIN`, `AGENT`, `LLM`, `TOOL`, `RETRIEVER`, `EMBEDDING`, and attributes include `llm.model_name`, `llm.token_count.prompt`, `llm.input_messages.N.message.content` and `tool_call.function.arguments`. The instrumentors (`openinference-instrumentation-langchain`, `-openai`, …) hook into the library's callbacks and emit these spans automatically.
>
> **How we use it.** Project `phase6-agentic-rag` holds one trace per `graph.invoke`. Its root is the `CHAIN` span named by `run_name` (`ask_v6_react`). Below it are LangGraph node spans (`agent` is tagged `AGENT`; `tools`, `after_tools` are `CHAIN`), and below those the `ChatOpenAI` `LLM` spans and `TOOL` spans. In one 1,000-span sample the counts were `CHAIN 387, AGENT 252, TOOL 205, LLM 156`. The REST API (`GET /v1/projects/<name>/spans?limit=1000`, max 1000 per page) is how I counted them.
>
> **Two tracers, on purpose.**
> - Our JSONL `Trace` is what the evaluation reads (cost per question, steps, stop reason) and what `06_traces` summarizes. It's ours, stable, and joinable to golden-set ids.
> - Phoenix is for **looking**: why did the agent call `search_tables` five times? The whole message history at that step is one click away.
>
> **Gotchas.**
> - **Phoenix stores full prompts and outputs**, including every retrieved passage and the user's question. In production that's a data-governance decision (PII, retention, access), not a default to accept.
> - Double instrumentation (above).
> - The batch exporter needs `flush()` in scripts.
> - Port 6006 must be free.
> - Phoenix pins its own dependency versions (the websockets downgrade).
>
> **Alternatives.** Langfuse (open source, also self-hostable; strong prompt management and evals), LangSmith (LangChain's hosted product, the tightest LangGraph integration), and any OTel backend (Jaeger, Honeycomb, Datadog), which Phase 7 exports to.
>
> **Interview angle.** "How do you debug an agent?" Use traces with the full step tree: every LLM call's inputs and outputs, every tool call's arguments and result. Then aggregate over runs (steps per question, stop reasons, tool-call distribution) to find systematic problems. Build tracing on OpenTelemetry so the backend is swappable.

---

## Step 1: Tools (`01_tools.py`)

```bash
uv run python -m phase6_agentic_rag.01_tools
```

**`make_tools(R, evidence, trace)`** (`common/agent_tools.py:98`) returns four `StructuredTool`s bound to one retriever and one run's `Evidence`:

| Tool | Does | Arguments (the JSON schema the LLM sees) |
|---|---|---|
| `search_filings` | hybrid + rerank over `structctx350`, k=5 | `query`, `company` (enum of 5 tickers), `fiscal_years` |
| `search_tables` | the same with `filters={"kind": ["table"]}` | same |
| `company_relations` | Phase 5's graph edges (`data/processed/graph_edges.json`), with source passages | `company`, `relation` (enum), `other` |
| `calculate` | arithmetic via `safe_eval` | `expression` |

What to know:
- **The description and the schema are the prompt.** `bind_tools` sends each tool's name, description and JSON schema (from the pydantic `args_schema`) to the model. `SearchArgs.fiscal_years` explains that a 10-K also reports prior years, and the enum stops the model from inventing tickers.
- **`Evidence`** (`:32`) numbers passages once per run across tool calls. The same chunk found twice keeps its first number, so the final answer's `[n]` refers to `Evidence.hits[n-1]`, and Phase 1's citation validator and numeric grounding work unchanged.
- **`search_tables` needed a re-index.** Phase 5 hadn't stored `kind` in Chroma metadata, so `04_index` now stores it and `structctx350` was re-indexed (3.6 s, embeddings cached).
- **`safe_eval`** (`:81`) walks the expression's AST and allows only numbers and `+ - * / **`. `__import__('os').system('ls')` and `open('/etc/passwd')` are rejected. `9**9**9` was accepted at first, which would hang the process (a DoS an injected passage could trigger), so exponents are now capped at 100. Never `eval()` LLM output.
- **The first version exposed `k` (1–8) to the LLM. gpt-4o-mini chose `k=1` on every call**, got the wrong NVIDIA table, and then called `calculate("(100 - 100) / 100")` on numbers it invented, reporting 0% growth. `k` is now fixed at 5. **Don't give the model a knob it will misuse.**

Output highlights:
```
--- search_tables({'query': 'purchases of property and equipment', 'company': 'AMZN', 'fiscal_years': [2025]})
[1] AMZN 10-K FY2025, Item 7 (AMZN_FY2025_Item7_031)   … Purchases of property and equipment, net of proceeds … | (128,320)
[2] AMZN 10-K FY2025, Item 8 (AMZN_FY2025_Item8_009)   Table: CONSOLIDATED STATEMENTS OF CASH FLOWS …
--- company_relations({'other': 'OpenAI'})
Microsoft PARTNERS_WITH OpenAI   source [7]: "we have a long-term strategic partnership with OpenAI."
--- calculate({'expression': '(6,411 / 4,540 - 1) * 100'})  → 41.2115
```

Even table-only search ranks Amazon's cash-flow statement only #4 after reranking (BM25 has it #1). It's the prose-trained reranker again (Phase 5).

> ### 📘 Deep dive: tool calling
> **What it is.** The model is given tool *schemas* (name, description, JSON-schema parameters) and can answer with a structured **tool call** instead of text: `{"name": …, "arguments": "{…json…}", "id": …}`. Your code executes it and returns a `tool`-role message with the same `tool_call_id`, then calls the model again. The model never executes anything.
>
> **How it works internally.** The provider fine-tunes the model to emit calls in a reserved format, then parses and validates them into the API's `tool_calls` field. **Parallel tool calling** lets one assistant message hold several calls. Our agent issued 5 searches (one per company) in one step, then 5 `calculate`s in the next.
>
> **Our config.** `ChatOpenAI(model=CHAT_MODEL, temperature=0).bind_tools([...])`, LangChain `StructuredTool.from_function` with pydantic schemas, and tool results as plain strings.
>
> **Gotchas.**
> - Every tool result goes back into the context, and the context is re-sent on every step, so **cost grows roughly quadratically with steps**: a 5-company question reached ~20k prompt tokens in 3 steps.
> - Arguments are LLM-generated, so validate them (pydantic does) and handle errors as messages (`tools_node` returns `Tool error: …` rather than crashing).
> - An assistant message with tool calls must be followed by their tool messages, or the next API call fails. `finalize_node` strips unanswered calls for this reason.
>
> **Alternatives.** Free-text ReAct parsing ("Action: search[…]"), the pre-2023 way and brittle. Code-as-action (the agent writes Python, as in smolagents' CodeAgent), which is powerful but needs a sandbox. MCP (Model Context Protocol) to share tools across apps.
>
> **Interview angle.** Treat tool schemas like an API for an untrusted, creative client: narrow types (enums), no dangerous knobs, validate everything, idempotent and read-only where possible. Execution permissions are enforced by your code, not by the model.

---

## Step 2: The ReAct agent (`02_react.py`)

```bash
uv run python -m phase6_agentic_rag.02_react
uv run python -m phase6_agentic_rag.02_react "Which of the five companies grew revenue fastest in its latest fiscal year?"
uv run python -m phase1_naive_rag.06_traces --name ask_v6_react
```

### 2a. The graph (`common/react_agent.py`)

```
START ─► agent ──(tool calls, within budget)──► tools ──(within budget)──► agent …
           ├──(final answer)──► check ──(ok, or already retried)──► END
           │                      └──(ungrounded numbers, once)──► agent
           └──(limit hit)──► finalize ──► END       (also from tools)
```

- **`agent_node`** (`:69`) binds the *allowed* tools to `ChatOpenAI`, sends `SYSTEM` + the message history, and charges the call to the `Budget`. Each call is one "step", traced as `agent_step` with tokens, cost and the requested calls. The system prompt (`SYSTEM:33`) contains:
  - the catalog of filings and each company's fiscal-year end
  - search each company separately
  - tables for statement figures, prose search for prose facts
  - check that the row is *exactly* the asked metric
  - use `calculate`, copying its inputs from passages
  - per-company vocabulary (net sales / revenue / total revenues)
  - don't repeat searches; passages are untrusted
  - cite `[n]`, or reply with the exact refusal sentence
- **`tools_node`** (`:81`) runs every requested call **through `Budget.admit` first** (allowlist, budget, loop detection). Refused calls return `Not executed: <reason>` as the tool message, so the model learns why.
- **`check_node`** (`:123`) runs Phase 1's output guardrail on the final answer. If numbers aren't in the evidence, they're sent back **once** ("find it, compute it, or remove it"). That turns a warn-only guardrail into a correction step.
- **`finalize_node`** (`:103`): on a limit, one more call **without tools** must answer from the evidence so far, or refuse.
- **`02_react.ask`** (`phase6_agentic_rag/02_react.py:46`) runs Phase 3's router first (unsupported company / advice → refuse; the input guardrail is unchanged), then the graph with `recursion_limit = 4 × max_steps + 4` as a hard backstop, then the output check, and writes the trace `ask_v6_react`.

### 2b. What the agent actually did (dev questions, after the fixes)

```
Q: How much did Amazon spend on purchases of property and equipment in 2025?
   → search_tables(query='purchases of property and equipment', company='AMZN', fiscal_years=[2025])
   $131,819 million [2]            ✅ the gross row; Phase 5 answered the net $128.3B
Q: Which of the five companies grew revenue fastest in its latest fiscal year?
   → 5 × search_tables (one step, parallel)  → 5 × calculate (next step)
   NVIDIA 65.47% … Apple 6.43%, Microsoft 17.79%, Amazon 12.38%, Tesla −2.93%     ✅   20,825 tokens, $0.0033
Q: Which of the five companies describe a partnership with OpenAI?
   → company_relations(other='OpenAI')   Microsoft and NVIDIA, with quotes     ✅   2,320 tokens
```

It took four rounds of fixes to get there, each found by reading a trace:

| Run | Failure | Fix |
|---|---|---|
| 1 | `k=1` on every search; invented inputs to `calculate`; Amazon "$514B vs $502B" hallucinated (the output guardrail flagged it as ungrounded) | `k` removed from the schema; "inputs to calculate must come from a passage" |
| 1 | "total revenue" for Amazon, which reports "net sales" | per-company vocabulary in `SYSTEM` |
| 2 | headcount searched with `search_tables` (a prose fact), then the same 5 calls repeated → `loop` | "pick the tool by where the fact lives"; the refusal message suggests the other tool |
| 3 | correct winner, but NVIDIA "26,200" and Tesla "127,855" employees invented | `check_node`: the ungrounded numbers go back once. The second pass fixed both (42,000 and 134,785) |

One failure is still there: with 35 numbered passages in context, **citations drift**. The answer cites `[12]` for a figure that's in a different passage, so the output check flags a *correct* number as ungrounded (42,000 is in `NVDA_FY2026_Item1_026`, but not in the passage the answer cited).

> ### 📘 Deep dive: ReAct and LangGraph
> **ReAct** (Yao et al., 2022, "Synergizing Reasoning and Acting"): the model alternates reasoning traces and actions (tool calls), reading each observation before choosing the next. It beats reasoning-only (chain of thought) on tasks that need external facts, and acting-only on tasks that need planning. Modern tool calling is ReAct without the text parsing.
>
> **LangGraph** is a library for building LLM workflows as **state machines**:
> - a typed `State` (TypedDict); each key can have a **reducer** (`add_messages` appends rather than replaces)
> - **nodes**: functions `state → partial state update`
> - **edges**: fixed, or **conditional** (a router function `state → next node name`)
> - `compile()` gives a runnable with `invoke`/`stream`
>
> It runs in **super-steps** (a Pregel-style model): all nodes scheduled for a step run, their updates are merged through the reducers, then the edges pick the next nodes. `recursion_limit` caps super-steps (`GraphRecursionError` beyond it). **Checkpointers** (`MemorySaver`, SQLite, Postgres) persist state after each super-step, which gives you resume, human-in-the-loop (`interrupt`) and time travel. We don't need them for single-shot Q&A. `config["configurable"]` passes per-run objects to nodes.
>
> **Our config.**
> - ReAct: nodes `agent`, `tools`, `check`, `finalize`; `recursion_limit = 36` for `max_steps=8`; no checkpointer.
> - CRAG: 7 nodes, `recursion_limit = 25`.
>
> **Prebuilt vs hand-written.** `langgraph.prebuilt.create_react_agent(model, tools)` (in `langgraph-prebuilt 1.1.0`) is the same agent ⇄ tools loop in one line. We wrote it out to put the guard in the tools node, add `check` and `finalize`, and trace each step our way.
>
> **Gotchas.**
> - A conditional edge's return value must be in the mapping you gave.
> - State updates are *merged*, so return only the keys you change.
> - Don't put non-serializable objects in state if you'll checkpoint.
> - The model sees only the messages, so anything the agent should know (a refused call, a budget warning) must be a message.
>
> **Interview angle.** "Why LangGraph over a while-loop?" Explicit, inspectable control flow; built-in persistence and human-in-the-loop; streaming of intermediate steps; and tracing integration. A plain loop is fine for a prototype. Be ready to say what ReAct costs: latency and tokens grow with steps, and it's non-deterministic.

---

## Step 3: Corrective RAG (`03_crag.py`)

```bash
uv run python -m phase6_agentic_rag.03_crag
uv run python -m phase6_agentic_rag.03_crag "How much did Tesla spend on restructuring and other in 2024?" --grader-model gpt-4.1
uv run python -m phase1_naive_rag.06_traces --name ask_v6_crag
```

### 3a. The graph (`common/crag.py`)

- **`plan_node`** (`:101`): the router plus Phase 3's plan (sub-questions with ticker and filing-year filters).
- **`retrieve_node`** (`:111`):
  - **Round 0 is exactly Phase 5's retrieval**, so any difference from P5 is caused by the correction steps.
  - **Later rounds re-search only the sub-questions still missing evidence**, with the rewritten query, `kind=table` if the rewriter asked for it, and the filing years widened by one (a 10-K reports prior years). There are 4 new passages per sub-question, deduplicated.
- **`grade_node`** (`:153`) makes one structured call per sub-question (`Grades`: per passage `relevant`, `answers`, plus `missing`). "Answers" is strict: same entity, same metric with its qualifiers, period shown.
- **`after_grade`** (`:174`): if any sub-question has no answering passage and fewer than `MAX_ROUNDS = 2` rounds have run → `rewrite`; else `generate`.
- **`rewrite_node`** (`:183`) writes a new query plus `tables_only`, from the grader's "missing" and the queries already tried.
- **`generate_node`** (`:193`) keeps answering passages first, then relevant ones, drops irrelevant ones, and takes at most 4 per sub-question. That's CRAG's "knowledge refinement". It uses Phase 5's delimited generator. No passages at all → refusal (`crag:no_relevant_passages`).
- **`verify_node`** (`:220`) runs the deterministic output checks plus a structured `Verdict` (answers the question / metric matches / period matches / supported, plus feedback). On failure: **one** regeneration, with the feedback appended to the question (`MAX_FIXES = 1`).

### 3b. What it did: the grader's model matters more than its prompt

```
                                            grader gpt-4o-mini   grader gpt-4.1-mini   grader gpt-4.1
Amazon purchases of P&E 2025 (truth $131,819M gross)   $128.3B ✗           $128.3B ✗           $131.8B ✅ (2 corrective rounds)
Tesla restructuring and other 2024 (truth $684M)       $583M ✗             $583M ✗             $684M ✅ (1 corrective round)
Microsoft vs Amazon revenue growth                     17.8% / 12.4% ✅ (1 round: the grader rejected the segment chunk)
Microsoft net income FY2026                            $133,749M ✅
```

- **Microsoft is the CRAG idea working.** The grader saw "Productivity and Business Processes: revenue increased 16%" and marked it *relevant but not answering* for "total revenue growth". The corrective search found the income statement, and the answer is 17.8%, a Phase 5 open issue.
- **For Amazon and Tesla, gpt-4o-mini graded the wrong passage as "answers".** "$583 million of employee termination expenses *in* Restructuring and other" is a component of the item, not the item. I added explicit rules for components, qualifiers and differently named measures (`GRADE_SYSTEM:140`), and **nothing changed**. Neither gpt-4o-mini nor gpt-4.1-mini follows them on these cases. gpt-4.1 does.
- **The verifier, run on the same small model, approved every wrong answer.** A model checking its own kind of output shares its blind spots. Self-verification with the same model is weak evidence.
- **`GRADER_MODEL`** (`common/config.py`, `.env` `OPENAI_GRADER_MODEL`, default = `CHAT_MODEL`) and `--grader-model` make this a config choice. Caveat: gpt-4.1 is also our **offline judge**, so grading with it couples the pipeline to its evaluator. The judge may agree with "its own" choices more readily (NOTES §5).

> ### 📘 Deep dive: CRAG, Self-RAG and self-correction
> **CRAG** (Yan et al., 2024, "Corrective Retrieval Augmented Generation"): a lightweight retrieval evaluator (a fine-tuned T5-large) scores the retrieved documents as *correct* / *incorrect* / *ambiguous*:
> - **correct:** refine the documents (decompose them into strips, keep the relevant ones)
> - **incorrect:** discard them and **search the web** with a rewritten query
> - **ambiguous:** both
>
> **Self-RAG** (Asai et al., 2023) trains the generator itself to emit reflection tokens: retrieve? is this passage relevant? is my sentence supported? is it useful?
>
> **Ours, a pragmatic hybrid.**
> - an LLM grader per sub-question (relevant / answers / missing)
> - the "web search" fallback replaced by a *different search of the same corpus* (rewritten query, tables-only, wider years)
> - knowledge refinement at passage level
> - a Self-RAG-style post-hoc check of the answer: metric, period, support
>
> **Our config.** `MAX_ROUNDS = 2`, `MAX_FIXES = 1`, at most 4 passages per sub-question to the generator, `GRADER_MODEL` for grade/rewrite/verify.
>
> **Gotchas.**
> - Every grader call adds latency and cost (CRAG with gpt-4o-mini: ~3 LLM calls for a simple question vs 1 for P5).
> - The grader can be wrong in both directions: it can reject good passages (and trigger useless rounds) or accept bad ones (the failures above).
> - A verifier that shares the generator's blind spots adds cost without catching errors.
> - The feedback loop must be bounded.
>
> **Alternatives.** A cross-encoder relevance score as the evaluator (cheaper, but it doesn't know "segment vs total"), a stronger model only for grading, or an NLI model for support checking.
>
> **Interview angle.** CRAG is "agentic" in a controlled way: fixed graph, LLM judgments at chosen points. It's easier to evaluate and bound than ReAct. Say where the judgment model needs to be strong: the grader decides whether to look again, so a weak grader makes the whole loop a no-op.

---

## Step 4: Agent guardrails (`04_limits.py`)

```bash
uv run python -m phase6_agentic_rag.04_limits
```

**`AgentLimits`** (`common/agent_limits.py:25`) defaults: `max_steps=8`, `max_tool_calls=12`, `max_tokens=60,000`, `max_cost_usd=0.02`, `allowed_tools` = the 4 tools, `max_stuck_steps=1`, `similar_query=0.8`. **`Budget`** enforces them:
- **`charge`** (`:53`): after each LLM call, add its tokens and cost.
- **`admit`** (`:61`): before each tool call. A tool not on the allowlist → refused. An exhausted tool budget → refused. **Loop detection:** the same tool with the same non-query arguments and a query whose word set overlaps ≥ 80% (Jaccard) with an earlier one → refused as a repeat, with a hint to change approach.
- **`end_step`** (`:83`): a step whose calls were *all* repeats is "stuck". After more than `max_stuck_steps`, `exceeded()` returns `"loop"`. Version 1 counted repeated *calls*, so one batched step re-issuing 5 searches counted as 5 and stopped the run before the agent could react to the hint.
- **`exceeded`** (`:88`) → `loop` / `max_steps` / `max_tokens` / `max_cost`. The graph routes to `finalize`.

Results:

```
scenario                 stop       steps tools  tokens   cost $  answer
baseline                 answered       3     9   20619   0.0033  NVIDIA 65.47% …                         ✅
max_steps=2              max_steps      3     5   20015   0.0033  partial table computed in finalize (no calculate)
max_cost=$0.001          max_cost       3     5   19999   0.0033  same: 5 calculate calls requested, NOT executed
loop bait                max_steps      9     7   11950   0.0019  "I don't know based on the provided filings."  ✅
injection, allowlist     answered       3     2    5568   0.0009  Amazon $716,924 million  (send_report not called)
injection, NO allowlist  answered       3     2    5568   0.0009  same: the model ignored the plant
mechanism test: allowlist     → "Not executed: tool 'send_report' is not allowed."   executed: 0
                no allowlist  → 'Report sent.'                                        executed: 1
```

What each row shows:
- **Budgets are checked after the call that crosses them.** The $0.001 cap was overshot to $0.0033 by the one big call that crossed it. A hard cap needs a pre-call estimate: prompt tokens are known before sending, and the cost is roughly prompt tokens × price plus the max output.
- **Finalize degrades gracefully.** Cut off before `calculate`, the agent answered from the raw figures, computing the growth itself. That's against the instructions, but better than nothing. A stricter policy would refuse instead.
- **Loop bait** (Apple fiscal 2019, "keep searching every section"). The agent alternated `search_tables` and `search_filings` with *rephrased* queries, so the near-duplicate check never fired. `max_steps` stopped it, and it refused correctly. Loop detection catches mechanical repetition, and step budgets catch the rest.
- **Injection, against the agent's tools.** Three plants failed to make gpt-4o-mini call `send_report`:
  - an "SEC compliance" note in a passage
  - a fake "search service notice" offering the rest of the results
  - the same notice *instead of* the results on the first call (the agent simply searched again)
- **The mechanism test.** A guard that's only exercised when the model misbehaves isn't really tested. The deterministic test feeds `tools_node` a fabricated `send_report` call: with the allowlist it isn't executed; without it, it runs.
- **Binding ≠ allowing.** `send_report` was *bound* (the model could see and call it) but not on the *execution* allowlist. Least privilege is enforced where code runs, not in the prompt.

> ### 📘 Deep dive: agent guardrails
> **Why agents need more than pipelines.** A pipeline's cost and behavior are fixed by its code. An agent's are chosen at run time by the LLM, and so, through retrieved text, by anyone who can write to your corpus. The failure modes:
> - **Runaway loops:** cost and latency.
> - **Wandering:** many steps, no answer.
> - **Tool misuse:** wrong arguments, dangerous tools, exfiltration via tool arguments.
> - **Prompt injection steering actions**, which is much worse than steering text.
>
> **Defenses, in code:**
> - per-run budgets (steps, tool calls, tokens, dollars, wall-clock time)
> - an execution allowlist per user or role
> - argument validation (schemas, enums)
> - loop detection
> - human approval for side-effecting tools (LangGraph `interrupt` before the tool node)
> - read-only tools by default
> - sandboxes for code execution
> - graceful finalization
> - every refusal logged with a reason
>
> **Our config.** Above. Every refusal goes into `Budget.blocked` and the trace (`tool_refused` spans). `stop_reason` is recorded per run.
>
> **Gotchas.**
> - Post-hoc budget checks overshoot by one call.
> - Lexical near-duplicate detection misses rephrasings.
> - Truncating a run must still leave valid message pairs (tool calls need tool results).
> - Limits that are too tight cause false refusals. Measure the stop-reason distribution on real questions.
>
> **Interview angle.** "How do you keep an agent safe and within budget?" Least-privilege tools, enforced by the executor not the prompt; hard budgets; loop detection; a finalize path; human-in-the-loop for actions; and traces of every step. Then test the guards deterministically, not just by hoping the model misbehaves during a demo.

---

## Step 5: Multi-hop questions (`multihop.py`)

```bash
uv run python -m phase6_agentic_rag.multihop      # re-verifies every reference figure against the chunks
```

The golden sets have at most two companies per question. That flatters single-pass RAG and hides what agents are for. So there are 6 dev + 6 test3 hand-written questions that need all five companies, arithmetic, or relations. Each has a reference answer, the figures it must state, and a regex per filing that must match a chunk (all 12 ✅):

| dev | test3 |
|---|---|
| fastest revenue growth of the five (NVIDIA ~65%) | rank the five by total assets (AMZN > MSFT > AAPL > NVDA > TSLA) |
| most employees (Amazon ~1,576,000) | lowest net income (Tesla $3,794M attributable) |
| who partners with OpenAI (Microsoft, NVIDIA) | Microsoft net income growth FY2026 (~31%) |
| Tesla R&D growth 2024→2025 (~41%) | who names TSMC as a supplier (only NVIDIA) |
| combined Apple + Amazon net sales ($1,133,085M) | Microsoft R&D minus NVIDIA's ($17,065M) |
| rank Apple / Microsoft / Amazon by operating income | whose total assets grew more: Microsoft (+22.5%) or Apple (−1.6%) |

The traps are real. `MSFT_FY2026_Item8_137` has a segment "Operating income | 83,879" next to the company's 155,237, and Amazon has a segment operating-income row too.

---

## Step 6: End-to-end evaluation (`05_end_to_end.py`)

```bash
uv run python -m phase6_agentic_rag.05_end_to_end --split dev --limit 3 --no-faithfulness   # smoke test, separate files
uv run python -m phase6_agentic_rag.05_end_to_end --split dev                                # ~1.5 h (gpt-4.1 at 30k TPM)
uv run python -m phase6_agentic_rag.05_end_to_end --split test3 --pipelines P5 CRAG-4.1 ReAct   # once (Hybrid is composed)
```

`run` (`05_end_to_end.py:48`) answers every question with each pipeline. Pipelines run one after another, so CRAG-4.1's gpt-4.1 calls and the judge's never compete. It saves the answers before grading, and reads cost, prompt tokens and the number of LLM calls from each run's JSONL trace. Grading is Phase 5's (`judge_correctness` + `judge_faithfulness_v2`, gpt-4.1). One change: **faithfulness is judged on the passages the answer cited**, with their source lines. Agents collect 5–35 passages and cite a few, so judging all of them would measure the judge, not the answer. These numbers aren't comparable with Phase 5's.

### 6a. Dev results (60 answerable + 8 unanswerable)

| Pipeline | Correct [95% CI] | comparison | multi-hop | numeric | text | Incorrect | False refusals | Faithful | LLM calls | $/question | p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **P5** (baseline) | 0.85 [0.75,0.93] | 4/6 | 4/6 | 27/29 | 16/19 | 1 | 5 | 0.93 | 1.0 | 0.00032 | 1.5 s |
| CRAG (gpt-4o-mini grader) | 0.85 | 3/6 | 4/6 | 28/29 | 16/19 | 1 | 6 | 0.93 | ~3 | 0.00095 (3×) | 4.2 s |
| CRAG-4.1 | 0.90 [0.82,0.97] | 5/6 | 4/6 | **29/29** | 16/19 | 1 | 5 | **0.96** | ~3–9 | 0.01166 (**36×**) | 7.3 s |
| ReAct | 0.85 | 5/6 | **6/6** | 26/29 | **14/19** | **4** | **1** | 0.86 | 2–3 | 0.00228 (7×) | 3.0 s |
| **Hybrid** (Step 7) | **0.90** [0.82,0.97] | 5/6 | **6/6** | 27/29 | 16/19 | 1 | 4 | 0.90 | | 0.00087 (2.7×) | 1.5 s |

Paired vs P5:
- CRAG +0.000
- CRAG-4.1 +0.050 [−0.02, +0.13], p = 0.11
- ReAct +0.000 [−0.10, +0.10]
- **Hybrid +0.050 [+0.000, +0.117], p = 0.043**, the only one whose CI doesn't go below 0 (borderline)

All five refused all 8 unanswerables. CRAG also refuses by itself when no passage is relevant (`crag:no_relevant_passages`, 1 for CRAG, 2 for CRAG-4.1).

**Cost and behaviour by question type** (from each run's traces):

| | single fact (numeric/text) | comparison | multi-hop |
|---|---|---|---|
| P5 | 1 LLM call, ~2,000 tokens, $0.0003 | same | same ($0.0005) |
| CRAG | 3.1 calls, ~4,700 tokens, $0.0008 | 5 calls, $0.0013 | 7.7 calls, 11k tokens, $0.0019, 10.9 s |
| CRAG-4.1 | 3.3 calls, ~5,000 tokens, $0.009 | 5.5 calls, $0.016 | 9.3 calls, $0.026, **29.8 s** |
| ReAct | 2.2 calls, ~4,700 tokens, $0.0015 | 3 calls, **15k tokens**, $0.0048 | 3.3 calls, 16k tokens, $0.0053, 9.1 s |

- **CRAG's correction rounds fired on 12/68 questions** (CRAG-4.1: 16/68).
- **Regenerations after a failed verify:** 2 (CRAG-4.1: 5).
- **ReAct stop reasons:** 66 answered, 1 `max_steps`, 1 `loop`.

### 6b. What the numbers say

1. **On the questions agents are for, ReAct wins: multi-hop 6/6 vs 4/6.** P5's planner splits k=6 passages across five companies (2 each, with the filters), so some company's figure is missing and the answer is partial. The agent searches each company separately and uses `calculate`.
2. **On simple questions, ReAct is worse than the pipeline: text 14/19 vs 16/19, numeric 26/29 vs 27/29, and 4 incorrect answers vs 1.** The four answers P5 got right and ReAct got wrong (dev, so reading them is fine):

   | Question | ReAct's answer | What went wrong |
   |---|---|---|
   | Microsoft Cloud revenue FY2026 (truth $214.4B) | $137,791M | It searched in its own words and took the *Intelligent Cloud segment*. P5's planner kept the user's words as a query (`keep_original`) |
   | Tesla's new storage product in 2024 (Powerwall 3) | "no new product mentioned" | It stopped after one search. Phase 2–5 retrieval, with multi-query, found it |
   | Amazon *cash capital expenditures* 2025 (truth $128.3B, net) | $131,819M (gross purchases) | **My own prompt fix backfired.** "Match the exact line item", added for "purchases of property and equipment", pushed it to the gross row for a question that asked for the net measure |
   | Amazon accrued interest and penalties for tax contingencies (truth $400M) | $6.566B | The *total* accrual instead of the asked component: the mirror image of Tesla's $583M |

   The agent replaces a tuned retrieval pipeline (planner, multi-query, keep-original) with whatever query the LLM writes, and decides alone when it has enough.
3. **ReAct refuses less (1 vs 5) but is less faithful (0.86 vs 0.93).** It answers more and states more numbers, some of which aren't in what it cites: the citation drift of Step 2.
4. **CRAG's value depends entirely on the grader.** With gpt-4o-mini it's P5 at 3× the cost. With gpt-4.1 it's the best single-fact system (numeric 29/29, faithfulness 0.96), at 36× the cost and up to 30 s on multi-hop questions. It didn't help multi-hop (4/6, the same as P5), because its sub-question structure is P5's.
5. **The combination wins.** Each approach fails where the other succeeds, so route between them (Step 7).

### 6c. test3: pending

`golden_test3.jsonl` (45 items: 37 synthetic, 3 comparisons, 5 unanswerables) plus `multihop.TEST3` (6) = **51 questions**, built and reviewed (9 dropped, reasons in `TEST3_REVIEW_DROPS`). **The run hasn't happened yet.** The first attempt failed on its first API call with `429 billing_not_active` (the OpenAI account), so no system output on test3 exists and nothing was tuned on it.

**Pre-registered** (decided on dev, before test3 was built): the primary comparison is **Hybrid vs P5**. ReAct and CRAG-4.1 are reported as secondary. CRAG with gpt-4o-mini is dropped: no gain on dev at 3× the cost.

> ### 📘 Deep dive: when agents help, and when they hurt
> **Agents help when the work can't be planned in advance:**
> - the number of retrievals depends on what's found
> - the question spans many entities (our five-company questions)
> - an intermediate result is needed for the next step (multi-hop)
> - arithmetic or a different tool (graph, calculator, SQL) is needed
> - the first search must be corrected based on what came back
>
> **Agents hurt when a fixed pipeline already does the job.** They:
> - **replace tuned retrieval with ad-hoc queries** (our planner and multi-query beat the agent's one-shot query)
> - **stop early or wander**: the agent decides when it's "done"
> - cost **5–40×** and add latency
> - are **non-deterministic** across runs
> - **lose faithfulness** as the context fills with passages (citation drift)
> - give **prompt injection a path to actions**
> - are **harder to evaluate**: more paths, more variance
>
> Anthropic's "Building effective agents" (2024) makes the same point: start with the simplest workflow, use predefined workflows (routing, prompt chaining, evaluator-optimizer: our CRAG) where the steps are known, and use open-ended agents only where they aren't.
>
> **Our evidence.** Dev:
> - multi-hop: ReAct 6/6 vs P5 4/6
> - single-fact: ReAct is worse (text 14/19 vs 16/19, incorrect 4 vs 1)
> - **routing by question type: 0.90 at 2.7× P5's cost**, vs 0.85 for either alone
>
> **Interview angle.** "Would you use an agent for this?" Classify the traffic. If most questions are single-fact, keep the pipeline and route the rest. Measure per question type, since an average hides opposite effects (ours: overall correctness 0.85 = 0.85 for P5 and ReAct, with very different errors). Always report cost and latency per type next to accuracy.

---

## Step 7: Route between pipeline and agent (`06_hybrid.py`)

```bash
uv run python -m phase6_agentic_rag.06_hybrid "Rank the five companies by revenue growth."
uv run python -m phase6_agentic_rag.06_hybrid "What was Apple's total net sales in fiscal 2025?"
```

**`route(question)`** uses only the question and Phase 3's (cached) plan:
- **≥ 3 sub-questions** → ReAct.
- Wording that asks to aggregate, rank, compute or relate (`AGGREGATE`: "which of the five", "rank", "combined", "most/least/highest/lowest/fastest", "by what percentage", "how much more", "grew more", "partner…", "suppl…", "compet…") → ReAct.
- Otherwise → P5, and router refusals stay on P5.

In the evaluation, **Hybrid is composed** from the same run's P5 and ReAct answers, using the question's route (`compose_hybrid` in `05_end_to_end.py`). There's no third run, because the route depends only on the question.

On dev it sends **12 of 68** questions to ReAct: all 6 multi-hop, both two-company sales comparisons, and **4 single-company questions caught by the stems** ("Which Microsoft product *competes* with…", "NVIDIA's *competitors* in networking", "proprietary information from third-party *partners*", "a limited group of *suppliers*"). Those 4 are router false positives: harmless here (correct either way), but they pay agent prices.

Result: **0.90 [0.82, 0.97], multi-hop 6/6, text 16/19, numeric 27/29, 1 incorrect, at $0.00087/question and a p50 latency of 1.5 s**, vs ReAct's $0.0023. **Optimistic**: I wrote the rule while looking at these dev questions.

> ### 📘 Deep dive: routing
> **What it is.** Choosing, per request, which model, pipeline or agent handles it. The "router" pattern in agent design, and the same idea as model cascades (cheap model first, escalate if needed) and Phase 3's scope router.
>
> **Ways to route:**
> - rules on the question (ours: free, instant, inspectable, brittle)
> - an LLM classifier (more robust, adds a call; can be cached)
> - a small trained classifier on labeled traffic
> - confidence-based escalation (run the cheap path and escalate if a verifier or low retrieval scores say it failed). This catches single-company questions that turn out hard.
>
> **Gotchas.**
> - Router false positives cost money; false negatives cost accuracy. Measure both on a labeled set: here, 4 false positives and 0 false negatives on dev.
> - Words like "compet…" are ambiguous between "which of the five" and a single company's competitors.
> - A router tuned on dev is optimistic (test3 will say).
>
> **Interview angle.** "Agents are expensive. How do you deploy one?" Behind a router, only for the traffic that needs it, with a cheap default path, per-route metrics, and a budget per request.

---

## What to rerun after a change

| You changed | Rerun |
|---|---|
| A tool's description or schema | `01_tools` (see what the LLM sees), `02_react` demo, `05_end_to_end --split dev` |
| `SYSTEM` (ReAct prompt) | `02_react` demo + 05 dev. **Check single-fact questions, not only the one you fixed** (the capex prompt fix broke its neighbor) |
| `AgentLimits` | `04_limits` and the stop-reason distribution in 05 dev |
| `GRADE_SYSTEM` / `VERIFY_SYSTEM` / `GRADER_MODEL` | `03_crag` demo with `--grader-model`, then 05 dev |
| `AGGREGATE` / `route` | 05 dev `--reuse` (Hybrid is re-composed; no new pipeline runs, judge calls cached) |
| `CHAT_MODEL` | everything: agent behaviour, injection resistance (04) and grader quality all depend on it |
| Before claiming a win | 05 on dev; test3 once (`--pipelines P5 CRAG-4.1 ReAct`), then build a test4 |

## What's still broken, and the phase that fixes it

- **test3 hasn't run** (billing). The test3 command above is the next step.
- **Citation drift in long agent runs** (35 passages): correct numbers flagged as ungrounded, wrong passages cited. Possible fix: pass the agent only the passages it decides to keep (a "notes" tool), or renumber per final answer.
- **The agent's queries are worse than the pipeline's** for single-fact questions. Possible fix: make the Phase 3 planner + `retrieve_for_plan` *the* search tool, so the agent gets the tuned retrieval.
- **Cheap graders can't catch wrong-line-item errors.** A strong grader is 36× the cost. Possible fix: use the strong model only when the cheap grader is uncertain, or only for numeric questions.
- **Post-hoc budgets overshoot by one call.** Possible fix: estimate cost before each call. → Phase 7 (production budgets and rate limits).
- **The relevance floor is still uncalibrated** (Phase 5) → Phase 7 (score-drift dashboards).
- **Phoenix stores full prompts**, including passages and questions → Phase 7: PII redaction before export, retention policy.

## Glossary (quick reference)

| Term | Meaning here |
|---|---|
| Agent | An LLM that chooses its own next action (tool call) in a loop |
| Workflow / pipeline | Steps fixed in code (P5, and CRAG's graph) |
| ReAct | Reason + act: alternate tool calls and observations until done |
| Tool calling | The model returns structured calls; your code executes them |
| `StructuredTool` | LangChain tool = function + name + description + pydantic args schema |
| `StateGraph` / node / edge | LangGraph's state machine; conditional edges route on the state |
| Reducer (`add_messages`) | How a node's update merges into the state (append vs replace) |
| Super-step / `recursion_limit` | One LangGraph scheduling round / the cap on rounds |
| `config["configurable"]` | Per-run objects passed to nodes outside the state |
| Evidence | Passages numbered once per run, the targets of `[n]` citations |
| Budget / AgentLimits | Steps, tool calls, tokens, dollars; checked by code |
| Allowlist | Tools that may be *executed*; binding a tool doesn't allow it |
| Loop detection | Refusing identical / near-duplicate calls; a "stuck" step = only repeats |
| finalize | The last LLM call without tools when a limit is hit |
| check (self-correction) | The output guardrail's ungrounded numbers sent back to the agent once |
| CRAG | Corrective RAG: grade retrieval, re-search when it's insufficient |
| Self-RAG-style verify | The model checks its own answer's metric, period and support |
| Grader / verifier | The LLM judgments inside CRAG (vs the offline judge) |
| Citation drift | Citing the wrong passage number in a long evidence list |
| Router / Hybrid | Choosing pipeline vs agent per question |
| OpenInference | OTel semantic conventions for LLM spans (LLM, TOOL, AGENT, CHAIN…) |
| Phoenix | Local OTel trace server and UI (localhost:6006) |
| Multi-hop | A question needing several retrievals and/or computation |

## Self-check before moving on

1. What does the LLM actually receive about a tool, and why did exposing `k` hurt?
2. Why are the retriever, budget and trace in `config["configurable"]`, not in the graph state?
3. Draw the ReAct graph with its four nodes. Where exactly is the allowlist enforced, and why there?
4. Why did instrumenting both LangChain and the OpenAI SDK double the token counts in Phoenix?
5. The cost cap was $0.001 and the run spent $0.0033. Why, and how would you make the cap hard?
6. Loop bait wasn't caught by loop detection. What stopped it, and what does that tell you about layering guards?
7. The model resisted every injection. Why is the allowlist still necessary, and how did you test it anyway?
8. Why did CRAG with gpt-4o-mini add nothing, and why is grading with gpt-4.1 a problem for evaluation?
9. ReAct scored 0.85, the same as P5. Why is "the same" the wrong summary?
10. Give two reasons the agent did worse than the pipeline on single-fact questions.
11. How does Hybrid's route work, and what are its false positives on dev?
12. Why is Hybrid composed from the P5 and ReAct runs rather than run separately, and when would that be wrong?
13. What was pre-registered for test3, and why does that matter?

