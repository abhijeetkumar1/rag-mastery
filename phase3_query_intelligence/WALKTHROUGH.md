# Phase 3 walkthrough: query intelligence

Read this top to bottom and run each command as you reach it. The theory, the interview Q&A and the experiments are in `NOTES.md`. All numbers are from real runs (`gpt-4o-mini` for query understanding and generation, `text-embedding-3-small`, the MiniLM reranker).

**What this phase fixes:**
- **Phase 2's wrong comparison answer.** Both halves were wrong: Microsoft's number was a *segment's* growth, and Amazon's came from *guidance*.
- **The "fiscal 2023" filter that returned nothing.**
- **Company names the regex can't resolve** ("the Redmond software giant").
- **Questions that shouldn't reach retrieval at all:** Google, investment advice, off-topic.

**How:** understand and reshape the question *before* retrieval.

## Architecture

### The Phase 3 pipeline

```
question
   │
   ▼
┌──────────────────────────────── plan() ───────────────────────────────────┐
│ analyze(): ONE structured LLM call (JSON schema), given                    │
│   • a catalog of what's indexed (tickers × filing years, "latest")         │
│   • today's date                                                           │
│ → route, companies, sub-questions (each: text, ticker, filing_years)       │
│                                                                            │
│ + deterministic policy (code):                                             │
│   • covered company named → never refuse as out_of_scope                   │
│   • every named company gets a sub-question                                │
│   • vocabulary expansion per company (Amazon: + "net sales")               │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │ route?
     ┌─────────────┬────────────┼──────────────────┬─────────────────┐
unsupported     advice      out_of_scope        answer
  company                                          │
     └── refuse immediately (no retrieval, no generation) ──┘   │ for each sub-question:
                                                                ▼
                               Phase 2 Retriever with filter {ticker, filing_years}
                               queries = [sub-question, original question] → RRF → rerank vs sub-question
                               (optional: + multi-query phrasings, or HyDE on the dense side)
                                                                │
                                                                ▼
                               relevance floor (best rerank < −3 → refuse)
                                                                │
                                                                ▼
                               Phase 1 generator (prompt UNCHANGED) → tolerant output guardrails
```

**The design principle: the LLM proposes, code enforces.**
- **The LLM** does what only a language model can: resolve "the iPhone maker", split a comparison into parts, recognise investment advice.
- **Code** does what shouldn't be a judgement call:
  - computing "latest fiscal year" from the catalog,
  - refusing to refuse when a covered company is named,
  - adding each company's own vocabulary,
  - keeping the user's original words as a query.

Every one of these rules exists because a measured LLM error needed it (Steps 3–6).

### Components

| Layer | Module | Key functions | Tech |
|---|---|---|---|
| Structured LLM calls | `common/llm.py` | `chat_json` (strict JSON schema, disk-cached) | OpenAI structured outputs |
| Query rewriting | `common/query.py` | `rewrite`, `multi_query`, `VOCAB` | LLM + domain glossary |
| HyDE | `common/query.py` | `hyde` | LLM-written hypothetical passage |
| Query understanding | `common/query.py` | `catalog`, `analyze` (v1/v2/v3), `ANALYZE_SCHEMA(_V3)` | One structured call: route + self-query + decomposition |
| Policy | `common/query.py` | `plan`, `expand_terms`, `TICKER_TERMS`, `GENERIC_TERMS` | Deterministic rules on top of `analyze` |
| Retrieval | `common/retriever.py` | `retrieve(hyde_passage=)`, `retrieve_multi` | Phase 2 pipeline + multi-query fusion |
| Output guardrail | `common/guardrails.py` | `check_answer(tolerant=True)`, `explain_derived` | Unit conversions and growth rates from one table row |
| Labeled cases | `phase3_query_intelligence/cases.py` | `ANALYZER_CASES` (22), `COMPARISON_PROBES` (4), `score_analysis`, `run_probes` | Regression tests for prompts |

### The plan: what the analyzer returns

```json
{"reason": "...",
 "companies": ["MSFT", "AMZN"],
 "unsupported_companies": [],
 "route": "answer",
 "sub_questions": [
   {"question": "What is the total revenue growth of Microsoft in fiscal 2026?", "ticker": "MSFT", "filing_years": [2026]},
   {"question": "What is the total revenue growth of Amazon in fiscal 2025? (net sales)", "ticker": "AMZN", "filing_years": [2025]}]}
```
**Notice:** "latest fiscal year" resolves to **2026 for Microsoft** and **2025 for Amazon**, because each company's newest filing differs. "(net sales)" was added by the policy, not the LLM.

### Scripts

| Script | Question it answers | LLM calls (cached after the first run) |
|---|---|---|
| `01_rewrite.py` | Does rewriting or multi-query improve retrieval? | rewrite, multi_query |
| `02_hyde.py` | Does searching with a fake answer passage help? | hyde |
| `03_self_query.py` | Can an LLM extract filters better than regex rules? | analyze |
| `04_decompose.py` | Do per-company sub-questions fix comparisons? | analyze, generation |
| `05_route.py` | How accurate is the router and input guardrail, per prompt version? | analyze ×4 versions |
| `06_ask.py` | The full pipeline, traced | analyze, generation |

---

## Step 0: Structured outputs and caching (`common/llm.py`)

Every query-understanding call goes through **`chat_json(messages, schema)`** (`llm.py:102`):
```
key = sha256(model + messages + schema)          .cache/chat_json/<key>.json exists? → return it (0 tokens, ~0 ms)
else: chat.completions.create(..., response_format={"type": "json_schema", "json_schema": {schema, "strict": True}})
      → json.loads → write cache → return (parsed, usage)
```

### 📘 Deep dive: structured outputs

**What it is:** OpenAI's `response_format={"type": "json_schema", "strict": True}` **constrains decoding**. At each step the model may only produce tokens that keep the output valid against your JSON schema: required fields, types, `enum` values, no extra keys. The output always parses.

**Why it matters for query understanding:**
- **The plan is consumed by code.** A free-text "the companies are Microsoft and Amazon…" would need fragile parsing.
- **`enum` constrains values:**
  - `"route": {"enum": ["answer", "unsupported_company", "investment_advice", "out_of_scope"]}`: the router can't invent a fifth route.
  - `"ticker": {"enum": ["AAPL", ..., null]}`: it can't return "Apple Inc." or "GOOGL" as a covered ticker.
- **Strict mode's rules:** every property must be listed in `required`, and `additionalProperties: false`. Optional fields are expressed as `["string", "null"]`.

**Field order matters.** The model writes JSON **left to right**, so a field can only use reasoning written *before* it. Our `ANALYZE_SCHEMA_V3` (`query.py:112`) puts `reason → companies → route` so the model names the companies and explains *before* deciding. (Here v3 didn't change the score, 19 vs 19, but the reasons it wrote exposed *why* the router was wrong. See Step 5.)

**Why cache LLM calls:**
- **Same input → same plan.** The cache makes experiment reruns free and **deterministic**. Temperature 0 isn't fully deterministic with the API; the cache is.
- **In production** the same idea is a query cache: exact-match first, then semantic caching (Phase 7).
- **Invalidation:** the key includes the full prompt, so any prompt edit misses the cache automatically. That's what makes prompt versions comparable (Step 5).

---

## Step 1: Rewriting and multi-query (`01_rewrite.py`)

```bash
uv run python -m phase3_query_intelligence.01_rewrite
```

### 1a. The functions

- **`rewrite(question, domain)`** (`query.py:41`) returns one search query.
  - `domain=False` asks for a generic "search query".
  - `domain=True` adds `VOCAB` (`query.py:32`), a glossary of how 10-Ks phrase things: revenue → "net sales" for Amazon and Apple, buybacks → "repurchased shares", workforce → "employees", capex → "purchases of property and equipment".
- **`multi_query(question, n=3)`** (`query.py:55`) returns 3 *different* phrasings: close to the question, formal 10-K language, and the table or section wording.
- **`Retriever.retrieve_multi(queries, rerank_query)`** (`retriever.py:95`) runs hybrid retrieval for *each* query, fuses all the lists with RRF, then runs **one** rerank against `rerank_query`.

Example:
```
question        How much did Apple spend on share buybacks in fiscal 2025?
generic rewrite 'Apple share buybacks fiscal 2025'
domain rewrite  'Apple share repurchase program fiscal 2025'
multi-query     ['Apple share buybacks fiscal 2025', 'repurchased shares fiscal year 2025',
                 'Apple 10-K share repurchase program details fiscal 2025']
```

### 1b. Results (12 Phase 2 probes)

```
strategy                hit@5  MRR@5  hit@50
original                 10/12   0.69    11/12
generic (rr=rewrite)      7/12   0.43    10/12   ← reranked against the REWRITE
generic (rr=orig)        10/12   0.69    10/12   ← same rewrite, reranked against the user's question
domain rewrite           10/12   0.69    11/12
multi-query (3+orig)     10/12   0.69    12/12   ← best candidate pool: every probe's answer is in the 50
```
First-run cost: ~1.0k / 2.3k / 3.1k tokens for the 12 probes, i.e. about 85–255 tokens per question.

**Three lessons:**
1. **Rewrite for retrieval, but rerank against the user's question.** The generic rewrite dropped from 10 to 7 only because the reranker judged relevance to the *rewrite*. Rewrites lose nuance: "Is Tesla too reliant on its CEO?" becomes a keyword string, and the reranker then scores chunks against the wrong question.
2. **Query techniques improve recall.** Multi-query took hit@50 to 12/12. They **can't fix ordering failures.** The two misses (Amazon workforce, Apple buybacks) survive every variant, because the reranker scores their mixed-topic chunks low, as diagnosed in Phase 2. That's a chunking problem (Phase 5).
3. **The vocabulary hints were written while looking at these probes,** so "domain rewrite = original" is an *optimistic* result, not evidence of a held-out gain. Phase 4 evaluates on questions the prompts never saw.

### 📘 Deep dive: query rewriting and multi-query (RAG-Fusion)

**Rewriting** turns a conversational question into a better *search* query:
- remove filler ("I was wondering…"),
- resolve references ("its CEO" → "Tesla's CEO"; in chat, rewrite with the conversation history),
- **translate into the documents' vocabulary**.

That last one is the **vocabulary mismatch problem**: users say "revenue", Amazon says "net sales". Neither dense retrieval nor BM25 fully bridges it (Phase 2's Amazon failure).

**Multi-query / RAG-Fusion** (Raudaschl, 2023): generate N phrasings, retrieve each, and fuse the results with RRF. Each phrasing reaches different chunks, which **raises recall** at the cost of N× retrieval work plus one LLM call.

**Risks:**
- **Semantic drift:** the rewrite changes *what* is asked. We saw both directions:
  - "deliveries" → "total revenues": a vocabulary hint applied where it didn't belong.
  - the user's exact 10-K phrase "purchases of property and equipment" → "capital expenditures", which lost the answer (Step 6).
- **Rerank target:** always rerank against the original intent, not the rewrite.
- **Cost and latency:** every rewrite is an LLM call on the critical path (Step 6: ~1.3–1.8 s uncached).

**Mitigations used here:**
- Retrieve with **both** the rewrite and the original question (`keep_original`, Step 6).
- Rerank against the user's question.
- Cache the rewrites.

**Interview angle:** *"Rewriting fixes vocabulary mismatch and conversational references, and multi-query raises recall. But rewrites drift, so I keep the original query in the fused set, rerank against the user's intent, and measure recall@N separately from top-k precision. In my tests multi-query improved the candidate pool from 11/12 to 12/12 but not the top 5, because the bottleneck was the reranker on mixed-topic chunks."*

---

## Step 2: HyDE (`02_hyde.py`)

```bash
uv run python -m phase3_query_intelligence.02_hyde
```

**`hyde(question)`** (`query.py:68`) asks the LLM for a 3–5 sentence passage *as it would appear in a 10-K*. `Retriever.retrieve(q, hyde_passage=p)` then embeds **the passage** (as a document, with no query prefix: `dense(..., as_document=True)`, `retriever.py:38`) for the dense side. BM25 and the reranker still use the question.

**Results:**
```
dense(question)                 8/12   MRR 0.60   hit@50 10/12
dense(HyDE)                     9/12   MRR 0.64   hit@50 11/12   ← fixed "Which NVIDIA products drove most data center sales?"
hybrid+rerank (question)       10/12   MRR 0.69   hit@50 11/12
hybrid+rerank (HyDE dense)     10/12   MRR 0.69   hit@50 11/12   ← no gain: the reranker is the bottleneck again
```

**The risk, visible in its output:**
```
Q: What was Apple's total net sales in fiscal 2025?
HyDE: "In fiscal year 2025, Apple Inc. reported total net sales of $400.5 billion, ..."   (truth: $416.2B)
Q: How big is Amazon's workforce?
HyDE: "As of December 31, 2022, Amazon.com, Inc. ... employed approximately 1,540,000 ..."  (wrong year, wrong figure)
```
The fake passage is only a search key; it's never shown to the user or the generator. But an invented **year** can pull the search toward the wrong filing.

### 📘 Deep dive: HyDE (Hypothetical Document Embeddings)

**Paper:** Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022).

**The idea:**
- A question ("Is Tesla too reliant on its CEO?") and its answer passage ("We are highly dependent on the services of Elon Musk…") look quite different: short versus long, question versus statement, casual versus formal. That's the **query/document asymmetry** from Phase 0.
- An LLM-written answer **has the answer's shape and vocabulary**, so its embedding lands near real answer passages, *even if its facts are wrong*.

**When it helps:**
- **Dense-only systems** (our dense-only probe score rose 8 → 9).
- **Zero-shot domains** with no fine-tuned retriever.
- **Short or vague questions.**

**When it doesn't:**
- **When a reranker already reads the question and chunk together:** our full pipeline showed no gain.
- **Exact-identifier queries:** BM25 handles those.
- **When the LLM knows nothing about the domain:** the fake passage is off-topic.
- **Numeric or temporal questions:** made-up figures and dates bias the search.

**Cost:** one LLM generation (~130 output tokens) plus one extra embedding per question, on the critical path.

**Variants:**
- Several hypothetical passages, averaged.
- HyDE for the dense side only (ours) vs for everything.
- Contrast with **HyQE / "hypothetical questions"**, the reverse trick at *indexing* time: generate the questions each chunk answers and index those (Phase 5 territory).

---

## Step 3: Self-query, LLM filter extraction (`03_self_query.py`)

```bash
uv run python -m phase3_query_intelligence.03_self_query
```

`analyze()` (`query.py:117`) extracts companies and **which filings to search**, given:
- **`catalog()`** (`query.py:18`): built from `manifest.json`, e.g. `AAPL: filings [2024, 2025] (latest: FY2025)`. The model knows what exists.
- **Today's date** in the prompt, so it can resolve "last year".
- **The 10-K rule:** "a filing for fiscal year X reports X, X−1 and X−2". That fixes Phase 2's fiscal-2023 pitfall at *query* time.

**Results** on 10 questions that name companies indirectly or use relative time:
```
question                                                   rules                 llm / plan
What does the iPhone maker say about tariffs?              ✅ AAPL (alias)        ✅ AAPL
How did the Redmond software giant's cloud business grow?  ❌ -                   ✅ MSFT [2026]
What risks does Jensen Huang's company see in China?       ❌ -                   ✅ NVDA
How many vehicles did the EV maker deliver last year?      ❌ -                   ✅ TSLA [2025]
What did the Cupertino company spend on R&D in fiscal 2023? ❌ -                  ✅ AAPL [2024]   ← next filing reports FY2023
Compare ... Microsoft and Amazon in their latest fiscal year ❌ (no years)        ✅ MSFT [2026], AMZN [2025]
How did GPU sales at the Blackwell maker change in fiscal 2026? ❌ -              ✅ NVDA [2026]
...
rules: companies 5/10, filings 6/10      llm: companies 10/10, filings 10/10
```

### 📘 Deep dive: self-query and temporal grounding

**Self-query retrieval** (LangChain's term): an LLM translates a natural-language question into (a) a semantic search string and (b) a **structured filter** over metadata fields. We extend it: the filter is **per sub-question**, and filing years are *reasoned* from the catalog ("latest" per company, and "FY2023 lives in the FY2024 filing").

**Rules vs LLM:**

| | Regex and alias rules (Phase 2) | LLM self-query (Phase 3) |
|---|---|---|
| "Apple", "AWS" | ✅ | ✅ |
| "the iPhone maker", "Jensen Huang's company" | Only if in the alias table | ✅ World knowledge |
| "latest", "last year" | ❌ | ✅ With the catalog and today's date |
| Cost and latency | 0 | ~950 tokens, ~1.3–1.8 s (uncached) |
| Failure mode | Misses (silently no filter) | Wrong filter (silently excludes the answer) |
| Auditable | Fully | Only via logs and labeled tests |

**Three hard-won rules** (each from a measured error in this phase):
1. **Give the model today's date.** Without it, "last year" became FY2024 for Tesla. The model has no clock, and its training cutoff isn't "now".
2. **Compute in code what code can compute.** One prompt edit flipped Microsoft's "latest fiscal year" from 2026 to 2025. Now "latest: FY2026" is computed from the catalog and handed to the model as a fact.
3. **Don't let a filter's year leak into the question text.** "Fiscal 2023" was once rewritten as "fiscal 2024" because the filing to search was FY2024. That changes what's being asked. Now the prompt says to keep the period the user asked about in the question text.

**A wrong filter is worse than no filter:** it silently removes the evidence (Phase 2's fiscal-2023 query returned *nothing*). That's why filters come with a fallback in production: if a filtered search returns weak results, retry without the filter.

---

## Step 4: Decomposition (`04_decompose.py`)

```bash
uv run python -m phase3_query_intelligence.04_decompose
```

**`decompose_retrieve()`** (`04_decompose.py:24`): for each sub-question from `plan()`, retrieve with its own filter `{ticker, filing_years}` and its own wording, then concatenate the results (k_each per company).

**Entity coverage**, meaning in how many companies' cases the evidence reaches the top 6, on 4 comparison probes with text-verified ground truth per company:
```
question                                              single  fan-out  decomp
Compare revenue growth, Microsoft vs Amazon             1/2      0/2     2/2
Which has more employees, NVIDIA or Microsoft?          0/2      1/2     1/2
Compare Apple's and NVIDIA's R&D spending               0/2      0/2     0/2
Does Amazon or Apple have more employees?               0/2      0/2     0/2
TOTAL                                                   1/8      1/8     3/8
```
**Decomposition triples coverage.** The misses are the **same chunking failures as before**: NVIDIA's "42,000 employees" sentence sits in a chunk that starts with supply-chain resilience (it doesn't reach the 50 candidates), Amazon's in a chunk about competition (rerank −11), and Apple's R&D *table* is fused #3 but the reranker scores it −2.19. **Query intelligence can't retrieve what chunking has diluted** (Phase 5).

**A labeling lesson:** the first Microsoft label accepted only the MD&A sentence "Revenue increased $50.1 billion or 18%". The answer used the **income statement** (`MSFT_FY2026_Item8_080`, total revenue 331,839), which is equally valid evidence, so decomposition was scored 2/8 when it deserved 3/8. **Incomplete relevance labels make good systems look bad.**

### End to end: the comparison question, finally right

```
Microsoft's total revenue for fiscal year 2026 was $331.8 billion, a growth of approximately 17.8% from $281.7 billion [3].
Amazon's total revenue for fiscal year 2025 was $716.9 billion, a growth of approximately 12.4% from $638.0 billion [4].
```
Ground truth: Microsoft +$50.1B / +18%, Amazon +12.4%. **Both correct.**

| Phase | Microsoft | Amazon |
|---|---|---|
| 1 | not retrieved | not retrieved → "I don't know" |
| 2 | "+16%, $19.2B": a **segment** (Productivity and Business Processes) | "+15%, $36.6B": **Q1 2026 guidance** plus an invented figure |
| 3 | ✅ 17.8% from the income statement | ✅ 12.4% from the net sales table |

**What fixed it:** a per-company sub-question in *that company's* vocabulary ("net sales" for Amazon), with a per-company filing filter (Microsoft FY2026, Amazon FY2025).

### The numeric guardrail had to change: strict vs tolerant

The *correct* answer failed the Phase 1 numeric check (**strict**): `ungrounded = ['638', '12.4', '17.8', '281.7', '331.8', '716.9']`. The LLM converted millions to billions and computed the percentages itself. In Phase 2 a *wrong* answer half-passed this check; here a *right* answer fails it. **A string-matching guardrail can't handle arithmetic.**

**`explain_derived()`** (`guardrails.py:61`), used by `check_answer(tolerant=True)`:
```
638    = 637959 million = 638 billion              unit conversion, needs "billion" in the answer
12.4   = growth (716924 / 637959 − 1) · 100        growth, needs "%", BOTH numbers on ONE table row
17.8   = growth (331839 / 281724 − 1) · 100
tamper 12.4 → 15.3:  passed=False  ungrounded=['15.3']     ✅ still catches wrong numbers
tamper 716.9 → 736.9: passed=False ungrounded=['736.9']    ✅
```
**The first version failed its tamper test.** It allowed any pair of source numbers (~100 numbers → ~10,000 pairs) and any unit, so it **accepted the wrong numbers 15.3% and $736.9B**, and "explained" 12.4 as `19817 − 7404` (two unrelated numbers). Requiring the unit word *and* a single table row removed the coincidences. On the Phase 1 demo, tolerant mode scores **8/10** (both false positives fixed; the two wrong-meaning false negatives remain).

### 📘 Deep dive: query decomposition

**What it is:** break a complex question into simpler sub-questions, retrieve for each, and answer from the combined evidence.

**Two shapes:**
- **Parallel** (ours): independent sub-questions. Comparisons ("A vs B"), multi-part questions ("revenue and headcount"). Retrieve them concurrently, then merge.
- **Sequential / multi-hop:** a later sub-question depends on an earlier answer. For example, "What's the revenue of the company that makes Blackwell?" needs "Blackwell → NVIDIA" first, then "NVIDIA revenue". That needs a loop: answer, then decide the next query. That's **agentic RAG** (Phase 6: ReAct, IRCoT, Self-Ask).

**Why it fixes comparisons:**
1. **Coverage:** each company gets its own top-k (Phase 2's fan-out did this too).
2. **Per-company vocabulary:** Amazon's sub-question says "net sales". Fan-out reused the *same* question text for both companies, which is why fan-out alone got the wrong Amazon chunk.
3. **Per-company filters:** each company's own "latest" filing.

**Costs and risks:**
- **N retrievals and one planning call.** Rerank cost grows linearly with the number of sub-questions (Step 6).
- **The planner can drop a company.** The policy adds a sub-question for every named company.
- **Sub-questions can drift** (Step 1).
- **The generator must recombine the evidence.** It still sees one flat context. A stronger design gives it the sub-questions explicitly, or answers each one and then synthesizes (map-reduce), which costs more LLM calls.

**Interview angle:** *"For comparative or multi-part questions I decompose into per-entity sub-questions, each with its own metadata filter and vocabulary, and retrieve them in parallel. Multi-hop questions need sequential decomposition, which is an agent loop. I measure it with entity coverage, the share of entities whose evidence reaches the context, because hit@k can't see that one side is missing."*

---

## Step 5: Routing and input guardrails (`05_route.py`)

```bash
uv run python -m phase3_query_intelligence.05_route
```

### 5a. Prompt versions, measured on 22 labeled questions (`cases.py`)

Each case lists the expected route, companies, allowed filing years per company, and a regex each sub-question must match (e.g. Tesla's must still say "deliver"; Amazon's must say "net sales").

```
version                    route  companies  filings  keeps_meaning   all-correct
v1 prompt                  20/22    20/22     20/22       17/22          17/22
v2 (+scope/wording rules)  20/22    21/22     20/22       19/22          19/22
v3 (+reason before route)  20/22    21/22     20/22       19/22          19/22
plan = v3 + code policy    21/22    22/22     21/22       22/22          20/22
```

**What happened between versions** (why prompt changes need regression tests):
- **v1** refused **"Is Tesla too reliant on its CEO?"** and **"What did Microsoft say about the Activision Blizzard acquisition?"** as `out_of_scope`. It turned "deliveries" into "total revenues", and left Amazon with "revenue".
- **v2** added scope rules ("10-Ks cover risks incl. dependence on key people, acquisitions…") and wording rules. That fixed three cases and **broke one that v1 got right**: "What was AWS operating income in 2025?" became `out_of_scope`.
- **v3** changed only the field order. Same score, but its `reason` field exposed the root cause: *"AWS operating income … is not a term used in the 10-K filings"* and *"[Activision] is not related to the 10-K filings of Microsoft"*. Both are false. **The router was guessing about content it had never seen.**
- **The fix was a policy, not another prompt:** if a covered company is named (by the LLM *or* by Phase 2's alias rules), never refuse as `out_of_scope`. Let retrieval evidence decide (the relevance floor).

### 5b. The input guardrail as a classifier

```
refusals: precision 8/8, recall 8/9   (false refusals of answerable questions: 0)
MISS: 'How do I reset my iPhone?'  expected=out_of_scope got=answer   ← the policy let it through
```
**That miss is intentional.** It's refused one layer later, by the relevance floor (best rerank −9.46; Step 6). A **false refusal can't be recovered downstream**, whereas a false accept still meets the floor, the LLM's refusal rule and the output checks. So the router is tuned for refusal *precision*, not recall.

### 📘 Deep dive: routing and input guardrails

**Routing** decides, per question, which path to take:
- which **index or tool** (10-Ks vs a SQL table vs web search),
- which **strategy** (simple retrieval, decomposition, HyDE),
- or **refuse** (out of scope, unsupported entity, policy violation).

It can be rules, a classifier, embeddings (nearest "route example"), or an LLM with structured output (ours).

**Input guardrails** are routes that end in a refusal:

| Guardrail | Here | Why |
|---|---|---|
| Unsupported entity | "Google", "Netflix", "Meta" → `unsupported_company` | The Phase 2 relevance floor **couldn't** catch these (Google's best chunk scored 1.83): on-topic chunks about *other* companies exist |
| Policy: investment advice | "Should I buy NVIDIA stock?" → decline | A financial assistant must not give recommendations (regulatory and liability risk) |
| Out of scope | "Capital of France", "Write a poem" | Saves cost and avoids off-topic answers |
| (Phase 7) Prompt injection, PII, abuse | | |

**The LLM proposes, code enforces:**
- An LLM router is a **classifier with errors**. Measure it on labeled cases, per prompt version, and watch for regressions (v2 fixed three cases and broke one).
- **Decide which error direction is cheaper.** Here a false refusal is permanent and a false accept is recoverable, so code overrides refusals whenever a covered company is named.
- **Don't let the router refuse based on beliefs about content.** It hasn't seen the corpus; retrieval has. "Retrieve, then decide" (the relevance floor) beats "guess, then refuse".

**Prompt versioning:**
- **`PROMPT_VERSION` is recorded in every trace** (`06_ask`), so any answer can be traced to the prompt that planned it.
- **Old versions stay runnable** (`analyze(q, "v1")`) to compare them.
- **The cache key includes the prompt**, so versions never mix.

**Interview angle:** *"I route with a structured-output LLM call that also extracts filters and sub-questions, measured as a classifier on labeled cases with precision and recall of refusals. Prompt edits are versioned and regression-tested: one of mine fixed three cases and broke a fourth. And refusals that depend on facts about the corpus are left to retrieval evidence, not to the router's guesses."*

---

## Step 6: The full pipeline (`06_ask.py`)

```bash
uv run python -m phase3_query_intelligence.06_ask                      # 6 demo questions
uv run python -m phase3_query_intelligence.06_ask "your question" --show-context
uv run python -m phase3_query_intelligence.06_ask --multi-query "..."    # + multi-query per sub-question
uv run python -m phase3_query_intelligence.06_ask --hyde "..."           # + HyDE on the dense side
uv run python -m phase3_query_intelligence.06_ask --no-plan "..."        # ablation: no analyzer
uv run python -m phase3_query_intelligence.06_ask --no-original "..."    # ablation: rewrite-only retrieval
uv run python -m phase1_naive_rag.06_traces --name ask_v3
```

### 6a. `ask()` (`06_ask.py:42`)

```
plan(question)                                  → route, sub-questions        (span "plan": tokens, cost, overrides, version)
route != answer → REFUSALS[route], return        (no retrieval, no generation)
for each sub-question:
    filter = {ticker, filing_years}
    hits += retrieve_multi([sub_question, ORIGINAL question], rerank vs sub_question, k_each)   # keep_original
relevance floor over all hits                                                  (span "relevance_floor")
Phase 1 SYSTEM + format_context(hits) → chat_completion                        (span "generate")
check_answer(answer, passages, tolerant=True)                                  (span "guardrails")
```

**`keep_original`: why the original question is always a query.**
- **The failure:** "How much did Amazon spend on **purchases of property and equipment** in 2025?" uses the 10-K's exact phrase. The planner rewrote it to "What were Amazon's **capital expenditures** in fiscal 2025?", retrieval missed the cash-flow line, and the answer was **"I don't know"**.
- **The fix:** retrieving with both the rewrite and the original (RRF-fused), plus a vocabulary rule mapping "capital expenditures" to "purchases of property and equipment". The answer came back.
- **Rule: a rewrite may add words; it must never be the only query.**

### 6b. Demo results

| Question | Outcome | Layer that decided |
|---|---|---|
| MSFT vs AMZN revenue growth | ✅ MSFT +17.8% ($331.8B), AMZN +12.4% ($716.9B), 6 numbers verified as derived | Decomposition + tolerant guardrail |
| Apple net sales **fiscal 2023** | ✅ **$383,285 million** (Phase 2: nothing retrieved) | Self-query: search the FY2025 filing for FY2023 |
| EV maker's deliveries "last year" | Refused (rerank −4.06). **Correct:** only the FY2024 10-K states delivery totals (1,789,000 in 2024), so 2025 isn't in the corpus | Relevance floor |
| Google advertising revenue | "I can only answer from the 10-K filings of Apple, Microsoft, NVIDIA, Tesla, Amazon. GOOGL is not covered." | Router (input guardrail) |
| Should I buy NVIDIA stock? | "…I can't give investment advice or recommendations." | Router (input guardrail) |
| How do I reset my iPhone? | Passed the router (policy), refused by the floor (−9.46) | Defense in depth |

**New answers, verified against the filings:**

| Question | Answer | Truth | Verdict |
|---|---|---|---|
| Amazon purchases of property and equipment 2025 | $128.3B | Gross (cash-flow line): **$131.8B**. $128.3B is the non-GAAP "net of proceeds from sales and incentives" line | ⚠️ A number that exists in the filing, attached to a slightly different metric; the qualifier was dropped |
| Tesla vs Apple gross margin | Tesla "16.2%" (automotive segment), energy 29.8%; Apple 46.9% ✅ | Tesla **total gross margin 18.0%** | ⚠️ Segment instead of total, the same pattern as Phase 2's Microsoft error |

**A recurring failure: the right number with the wrong label.** It appeared three times (Microsoft segment revenue, Amazon net capex, Tesla segment margin). The numeric guardrail can't see it, because the numbers are real. The fixes are semantic faithfulness checks (**Phase 4**) and chunks that carry their segment and metric context (**Phase 5**).

### 6c. Observability: what query intelligence costs

Uncached requests (new questions), from `06_traces --name ask_v3`:
```
plan            1840.6 ms   tokens in/out 944/77    cost $0.00019     ← a second LLM call on the critical path
generate         975.5 ms   tokens in/out 2129/45   cost $0.00035
total           3727 ms
```

Measured over all traced requests (answered questions only):

| | Phase 2 (`ask_v2`, n=11) | Phase 3, plan uncached (n=3) | Phase 3, plan cached (n=6) |
|---|---|---|---|
| LLM calls per answered question | 1 | 2 | 1 |
| Latency, median (range) | 2.1 s (1.4–15.1 s*) | **3.7 s** (3.2–4.7 s) | 2.8 s (1.2–3.8 s) |
| Cost per answered question | $0.00031 | **$0.00054** | $0.00036 |
| Cost of a router refusal | n/a (Phase 2 pays for generation) | ≈ $0.0002 (plan only, no retrieval or generation) | ≈ $0 |

\* The 15.1 s is Phase 2's cold start, before `warmup()` existed. Phase 3 cached-plan requests still cost more than Phase 2 because each sub-question retrieves with two queries (`keep_original`) and reranks separately.

**Levers:**
- **Cache plans** (exact match, then semantic caching in Phase 7).
- **A fast path:** skip the analyzer when the rules find exactly one company and no relative time.
- **A smaller or faster model** for planning.
- **Run planning and a speculative plain retrieval in parallel.**
- **Stream the generation** (Phase 7).

The trace records `prompt_version`, the route, every sub-question, which policy rules fired, and whether the plan was cached. Without those, a wrong answer can't be traced to a planning mistake.

---

## What to rerun after a change

| You changed... | Rerun |
|---|---|
| `analyze()` prompt or schema (`common/query.py`) | Bump `PROMPT_VERSION`, run `05_route` (regressions?), `03_self_query`, then `06_ask` |
| Policy (`plan`, `TICKER_TERMS`, `GENERIC_TERMS`) | `05_route`, `04_decompose`, `06_ask` |
| `VOCAB` | `01_rewrite` (and `05_route`: the analyzer prompt includes it) |
| Numeric guardrail (`explain_derived`) | `04_decompose` (the tamper test must still fail the tampered numbers), Phase 1 `05_guardrails_demo` |
| Labeled cases (`cases.py`) | `04`, `05` (they raise an error if a comparison probe has no relevant chunk) |
| Anything in Phases 1–2 | Their own "what to rerun" tables. The chat_json cache stays valid because it's keyed by prompt |

## Debugging a Phase 3 answer

1. **`06_traces --name ask_v3 --last 1`.**
2. **The `plan` span:** is the route right? Are the companies and each sub-question's `filing_years` right? Did the sub-question keep the user's meaning? Which policy overrides fired?
3. **Per sub-question retrieval:** `dense_search` / `bm25_search` / `multi_query_fusion` / `rerank` spans (the Phase 2 debugging order).
4. **Generation:** a correct number with the wrong label (segment vs total, net vs gross)? Check the cited chunk's headings.
5. **Guardrails:** `derived_numbers` shows *how* each number was verified. Check that the derivation is the intended one.

## What's still broken, and the phase that fixes it

| Failure | Seen in | Fix | Phase |
|---|---|---|---|
| Right number, wrong label (segment vs total, net vs gross) | MSFT (Phase 2), Tesla margin, Amazon capex | Faithfulness / LLM-as-judge | 4 |
| Are these gains real? Prompts and vocabulary were tuned on the same cases they're scored on | Every table here | A held-out golden set, and a harness that measures answers, not just retrieval | 4 |
| Evidence diluted in mixed-topic chunks | Employees (NVDA, AMZN), buybacks, workforce | Smaller or parent-child chunks, contextual headers | 5 |
| The reranker is weak on tables | Apple R&D table (−2.19) | Table-aware chunking, structured (XBRL) data | 5 |
| Multi-hop questions ("revenue of the Blackwell maker") | Not handled: parallel decomposition only | Agent loop | 6 |
| The planner adds ~1.5 s per question | Traces | Plan caching, a fast path, streaming | 7 |

## Glossary (quick reference)

| Term | One-line meaning |
|---|---|
| **Query rewriting** | LLM rewrites the question into a better search query (vocabulary, references) |
| **Vocabulary mismatch** | The user's words ≠ the document's words ("revenue" vs "net sales") |
| **Semantic drift** | A rewrite that changes what's being asked |
| **Multi-query / RAG-Fusion** | N phrasings → N retrievals → RRF → one rerank |
| **HyDE** | Embed an LLM-written hypothetical answer passage instead of the question |
| **Self-query** | LLM turns the question into a search string plus a metadata filter |
| **Catalog** | What's indexed (companies × filings), given to the planner |
| **Temporal grounding** | Giving the model today's date so relative time resolves correctly |
| **Decomposition (parallel / sequential)** | Split into independent sub-questions / chained ones (multi-hop) |
| **Entity coverage** | Share of entities in a question whose evidence reaches the context |
| **Routing** | Choosing a path per question: strategy, index, or refusal |
| **Input guardrail** | A route that refuses before retrieval (unsupported entity, advice, off-topic) |
| **Structured outputs** | JSON-schema-constrained decoding (strict mode, enums) |
| **Prompt versioning / regression test** | A named prompt version, plus labeled cases re-run on every change |
| **LLM proposes, code enforces** | The LLM makes judgment calls; deterministic rules guard its known failure modes |
| **Tolerant numeric check / tamper test** | Accept unit conversions and growth rates / verify it still rejects wrong numbers |

## Self-check before moving on
- [ ] Why did the generic rewrite score 7/12 when reranked against the rewrite but 10/12 against the original?
- [ ] Multi-query raised hit@50 to 12/12, but hit@5 stayed at 10/12. Where is the bottleneck, and which phase fixes it?
- [ ] Why can HyDE help a dense-only system but not ours? What's the risk in its fake passages?
- [ ] Give three planner errors from this phase that code now prevents, and the rule for each.
- [ ] Why does the router refuse with high precision but not full recall, and why is that the right trade-off here?
- [ ] Walk the comparison question through Phases 1, 2 and 3: what was wrong each time, and what fixed it?
- [ ] The first tolerant guardrail "explained" 12.4 as 19817 − 7404. Why is that dangerous, and what two constraints fixed it?
- [ ] "Right number, wrong label" happened three times. Why can't the numeric guardrail catch it?
- [ ] What does planning cost per question (latency, tokens), and name three ways to reduce it.
