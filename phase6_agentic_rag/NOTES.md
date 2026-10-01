# Phase 6: Agentic RAG

```
01_tools        the 4 tools (search_filings, search_tables, company_relations, calculate) called by hand
02_react        ReAct agent (LangGraph StateGraph): agent ⇄ guarded tools → output check → answer; limits → finalize
03_crag         Corrective RAG: plan → retrieve → grade → (rewrite → re-search)* → generate → verify (→ fix once)
04_limits       step / cost budgets, loop bait, injection vs the tool allowlist, a deterministic allowlist test
05_end_to_end   P5 vs CRAG vs CRAG-4.1 vs ReAct vs Hybrid on golden + multi-hop questions (dev; test3 once)
06_hybrid       route: ReAct for aggregate / multi-company questions, the Phase 5 pipeline otherwise
multihop.py     12 hand-verified questions (6 dev, 6 test3) across 3–5 companies, arithmetic, relations
```
Start with `WALKTHROUGH.md` for the step-by-step explanation and the deep dives.

```bash
PHOENIX_WORKING_DIR=data/phoenix uv run phoenix serve       # other terminal; http://localhost:6006
uv run python -m phase6_agentic_rag.01_tools
uv run python -m phase6_agentic_rag.02_react
uv run python -m phase6_agentic_rag.03_crag
uv run python -m phase6_agentic_rag.04_limits
uv run python -m phase6_agentic_rag.multihop
uv run python -m phase6_agentic_rag.05_end_to_end --split dev
uv run python -m phase4_evaluation.01_build_golden --split test3
uv run python -m phase6_agentic_rag.05_end_to_end --split test3 --pipelines P5 CRAG-4.1 ReAct   # NOT RUN YET (billing)
```

## 1. The headline (dev: 60 answerable + 8 unanswerable; test3 pending)

| Pipeline | Correct | Multi-hop | Single-fact incorrect | False refusals | $/question | p50 latency |
|---|---|---|---|---|---|---|
| P5 (Phase 5) | 0.85 | 4/6 | 1 | 5 | 0.0003 | 1.5 s |
| CRAG, gpt-4o-mini grader | 0.85 | 4/6 | 1 | 6 | 0.0010 | 4.2 s |
| CRAG, gpt-4.1 grader | 0.90 | 4/6 | 1 | 5 | 0.0117 | 7.3 s |
| ReAct | 0.85 | **6/6** | **4** | **1** | 0.0023 | 3.0 s |
| **Hybrid (route)** | **0.90** | **6/6** | 1 | 4 | **0.0009** | 1.5 s |

- **ReAct and P5 have the same average with opposite errors.** ReAct wins every multi-hop question and refuses less. It loses single-fact questions because its own queries and stopping decisions are worse than the tuned pipeline's.
- **CRAG only works with a strong grader.** gpt-4o-mini approved the same wrong line items it would have produced itself. gpt-4.1 caught them, at 36× the cost.
- **Routing combines the strengths:** +0.05 vs P5 (CI [+0.00, +0.12], borderline) at 2.7× the cost. It's optimistic, because the rule was written on dev.
- **test3 is built, reviewed and pre-registered (Hybrid vs P5), but hasn't run.** The OpenAI account returned `billing_not_active` on the first call.

## 2. Workflow or agent: the design space

| | Fixed pipeline (P5) | Workflow with LLM judgments (CRAG) | Agent (ReAct) |
|---|---|---|---|
| Who decides the next step | code | code, with LLM yes/no at fixed points | the LLM |
| Cost / latency | fixed, lowest | bounded (~3–9 calls) | variable (2–9 calls, tokens grow quadratically with steps) |
| Evaluability | easy | per-node (grader, verifier) | hard: many paths |
| Strength here | single-fact questions | catching wrong-line-item answers (with a strong grader) | multi-company, arithmetic, relations |
| Failure here | multi-hop (k split across 5 companies) | weak grader = no-op | ad-hoc queries, early stopping, citation drift |

The order to try things in: pipeline → workflow (routing, chaining, evaluator–optimizer) → agent, and only where the earlier one measurably fails.

## 3. Things that surprised me (and are good interview stories)

1. **The LLM sets bad parameters if you let it.** With `k` exposed, gpt-4o-mini chose `k=1` on every search, got NVIDIA's wrong table and then fed invented numbers to `calculate`. Removing one schema field fixed it.
2. **Self-verification by the same model approved its own mistakes.** The gpt-4o-mini verifier said "metric matches, period matches, supported" for $583M (a component) and $128.3B (net, not gross). Explicit rules didn't help, gpt-4.1-mini didn't help either, and gpt-4.1 did.
3. **A prompt fix for one question broke its neighbor.** "Match the exact line item" fixed *purchases of property and equipment* and broke *cash capital expenditures*. A prompt change is a code change: it needs the full eval, not the one example.
4. **The agent made up its supporting numbers while getting the answer right.** It named Amazon as the largest employer, with NVIDIA at "26,200" and Tesla at "127,855". The output guardrail flagged them, and sending them back once (`check_node`) fixed both. A warn-only guardrail became a correction loop.
5. **The model resisted every injection I tried**, so the allowlist was never exercised. That's not evidence the allowlist is unnecessary: a deterministic test (a fabricated tool call through the executor) is how you test a guard.
6. **Double instrumentation double-counts.** LangChain and OpenAI SDK instrumentors both produced LLM spans for the same calls.
7. **Average accuracy hid everything.** 0.85 for both P5 and ReAct, with 4/6 vs 6/6 on multi-hop and 1 vs 4 wrong answers on simple questions. Always slice by question type.

## 4. Guardrails this phase: agent limits

| Guard | Where | Tested by | Result | Where it fails |
|---|---|---|---|---|
| Step budget (`max_steps=8`) | `Budget.exceeded` → finalize | `04_limits`: max_steps=2, loop bait | stops; finalize answers or refuses | – |
| Cost / token budget | `Budget.charge` (after each call) | max_cost=$0.001 | stopped, but at $0.0033 | **overshoots by one call** (no pre-call estimate) |
| Tool-call budget (12) | `Budget.admit` | – | – | – |
| Loop detection (identical / Jaccard ≥ 0.8; "stuck" steps) | `Budget.admit`, `end_step` | dev run (headcount via search_tables, repeated) | caught, with a hint to switch tools | **rephrased loops** (loop bait alternated tools and wording; max_steps caught it) |
| Execution allowlist | `tools_node` → `Budget.admit` | 3 injection plants + a deterministic test | refused when called; executed without it | none in the test; it only covers tools, not what an allowed tool does with its arguments |
| Finalize on limit | `finalize_node` | max_steps / max_cost | best-effort answer | it computed growth itself, against the instructions: a stricter policy would refuse |
| Output check → self-correction | `check_node` | employees question | fixed 2 invented figures | **citation drift** flags correct numbers cited to the wrong passage |
| Input router (Phase 3) | `02_react.route` | unanswerables | 8/8 refused (all pipelines) | – |
| `safe_eval` | `calculate` | `01_tools` | code rejected; `9**9**9` (DoS) rejected after the fix | – |

## 5. Observability this phase

- **Phoenix (OpenInference over OpenTelemetry):** one trace per `graph.invoke`, with node, LLM and tool spans and their full inputs and outputs. 1,000-span sample: CHAIN 387, AGENT 252, TOOL 205, LLM 156. The REST API is `GET /v1/projects/phase6-agentic-rag/spans` (max 1,000 per page).
- **JSONL trace** (`ask_v6_react`, `ask_v6_crag`, `ask_v6_crag41`):
  - spans `agent_step` (tokens, cost, requested calls), `tool:<name>`, `tool_refused`, `output_check`, `finalize`, `grade`, `rewrite`, `corrective_search`, `verify`
  - attributes `stop_reason`, `steps`, `tool_calls`, `rounds`, `fixes`, `verified`
  - this is what the evaluation reads for cost per question
- **What to monitor for an agent in production:**
  - steps per request (distribution, not mean)
  - stop-reason rates (`max_steps`/`loop` = stuck agents)
  - refused tool calls (an injection or prompt regression shows up here first)
  - tokens per step (context growth)
  - cost per route
- **A judge-coupling caveat for CRAG-4.1:** the pipeline's grader and the offline judge are the same model, so the judge may favor answers its own model selected. A different judge family would make that number cleaner.

---

## Interview Q&A

**Q: What is agentic RAG, and when would you use it?**
RAG where an LLM decides the retrieval steps: what to search, with which tool, whether the evidence is enough, when to stop. Use it when the steps can't be fixed in advance: questions across many entities, multi-hop, arithmetic, or retrieval that needs correcting. Not for single-fact lookups: in my project a ReAct agent got every multi-hop question right (6/6 vs 4/6) but made four times as many wrong answers on simple questions as the fixed pipeline, at 7× the cost. Routing between them gave the best accuracy at 2.7× the cost.

**Q: Explain ReAct.**
The model alternates reasoning and actions: it emits a tool call, reads the result, decides the next call, and repeats until it answers (Yao et al., 2022). With native tool calling there's no text parsing; the model returns structured calls and your code executes them. The context grows every step, so cost grows roughly quadratically, and you need a step limit, loop detection and a way to stop gracefully.

**Q: What's CRAG? How is it different from Self-RAG?**
Corrective RAG grades the retrieved documents. If they're insufficient it rewrites the query and searches again (the original paper falls back to web search), and it filters the irrelevant parts before generating. Self-RAG trains the generator to emit reflection tokens (retrieve? relevant? supported?) itself. I built a CRAG-style workflow with an LLM grader and a post-hoc verifier. Its effect depended entirely on the grader: gpt-4o-mini approved the same wrong line items it would have produced, while gpt-4.1 caught them, at 36× the cost.

**Q: How do you stop an agent from looping or overspending?**
Enforce limits in code, not in the prompt:
- step, tool-call, token and dollar budgets
- detection of identical and near-duplicate calls (refuse them with a hint, and stop after repeated stuck steps)
- a recursion limit as a backstop
- a finalize step that answers from the evidence so far, or refuses

Check budgets *before* expensive calls, or you overshoot by one: my $0.001 cap ended at $0.0033. Lexical loop detection misses rephrased loops; the step budget catches those.

**Q: How do you make tool use safe?**
- **Least privilege:** read-only tools by default.
- **An execution allowlist** checked by the executor, not the prompt. A tool can be visible to the model and still not be allowed to run.
- **Narrow argument schemas** (enums, no dangerous knobs), validated before execution, with errors returned as messages.
- **No `eval`:** I parse arithmetic as an AST and cap exponents.
- **Human approval** for side effects.

Test guards deterministically: my model resisted every injection attempt, so I fed the executor a fabricated malicious call to prove the allowlist blocks it.

**Q: Can an LLM verify its own answers?**
Partly, and less than you'd hope. When the verifier is the same (or a similar small) model, it shares the generator's blind spots. Mine approved a component figure as the total and a net figure as gross, even with explicit rules. A deterministic check (numbers must appear in the cited passages) caught different errors, and turning its findings into one feedback round fixed invented figures. Use a stronger or different model for verification, or a deterministic check, and measure the verifier like any judge.

**Q: How do you trace and debug an agent?**
Trace every node, LLM call and tool call with inputs and outputs. I used Arize Phoenix with OpenInference auto-instrumentation over OpenTelemetry, plus my own JSONL spans for cost and stop reasons that the evaluation reads. Debug individual runs in the tree, and aggregate across runs: steps per question, stop-reason rates, refused calls, tokens per step. Watch for double instrumentation (double-counted tokens) and for traces storing full prompts with user data.

**Q: When do agents hurt?**
When a fixed pipeline already solves the task. The agent then:
- replaces tuned retrieval with its own ad-hoc queries
- decides alone when it's "done" (stopping early)
- costs more and is slower
- is non-deterministic
- adds paths for prompt injection to reach actions

In my eval the agent was worse on single-fact text questions (14/19 vs 16/19) and less faithful (0.86 vs 0.93), because its long contexts caused citation drift.

**Q: How would you route between a pipeline and an agent?**
Classify the request, using rules, an LLM classifier, or confidence-based escalation. Send aggregate, multi-entity or computational questions to the agent and everything else to the cheap pipeline. Measure router false positives (cost) and false negatives (accuracy) on labeled traffic, and report metrics per route. My rule routed 12 of 68 dev questions (all six multi-hop ones, plus four single-company false positives caught by words like "competes") and scored +0.05 over the pipeline at 2.7× its cost.

## Experiments to try
- [ ] **Run test3** (`--pipelines P5 CRAG-4.1 ReAct`) and report Hybrid vs P5, as pre-registered.
- [ ] Make Phase 3's planner + `retrieve_for_plan` the agent's search tool. Does ReAct stop losing single-fact questions?
- [ ] Pre-call cost estimate in `Budget` (prompt tokens × price + max output), then rerun max_cost=$0.001. Does it now stop *before* the big call?
- [ ] An LLM router (gpt-4o-mini, cached) vs the regex rule: false positives and negatives on dev.
- [ ] Escalation instead of routing: run P5, and call ReAct only if P5 refuses or the output check fails. What's the cost and accuracy?
- [ ] CRAG with gpt-4.1 as grader *only for numeric questions* (most of the gain was numeric 29/29).
- [ ] A different judge family for CRAG-4.1 (to remove the grader/judge coupling), or judge v2 with gpt-4o: how much does CRAG-4.1's lead move?
- [ ] Human-in-the-loop: `interrupt_before=["tools"]` with a checkpointer for a side-effecting tool; resume after approval.
- [ ] Stream the agent's steps (`graph.stream(..., stream_mode="updates")`) to show progress in Phase 7's API.
- [ ] Repeat the injection test with a weaker or older model, or with "passages are untrusted" removed from `SYSTEM`. When does the agent call `send_report`?
