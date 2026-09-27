# Phase 4 walkthrough: evaluation

Read this top to bottom and run each command as you reach it. The theory, the interview Q&A and the experiments are in `NOTES.md`. All numbers are from real runs. The pipelines under test use `gpt-4o-mini`; the judge and the golden-set generator use **`gpt-4.1`**.

**What this phase answers:** "Are Phases 2 and 3 actually better, or did they only look better on questions I tuned them on?" Every earlier table used probes I wrote *while* building the thing being measured. Phase 4 builds questions nothing was tuned on, measures all three pipelines on them with confidence intervals, and measures the **judges** too.

## Architecture

```
                       ┌──────────────────────── 01_build_golden ──────────────────────────┐
 chunks ── stratified ─► gpt-4.1 writes Q + A + VERBATIM evidence quote                     │
 random     sample      │ code: quote must be in the chunk; answer figures must be in the quote│
 (seed)                 │ gpt-4.1 critic: standalone? one answer?                            │
                        │ HUMAN review (reasons recorded in code)                           │
                        │ + hand-verified comparisons + unanswerables                       │
                        └──► golden_dev.jsonl (62)      golden_test.jsonl (52, disjoint chunks)
                                   │                               │
                  diagnose & fix ◄─┘                               └─► report ONCE
                                   │                                         │
         ┌─────────────────────────┴─────────────────────────────────────────┤
         ▼                                                                   ▼
 02_retrieval_eval                                             04_end_to_end
 6 retrieval configs × recall/precision/MRR/nDCG@6             P1 vs P2 vs P3 pipelines, answer by answer
 bootstrap CIs, paired tests                                   correctness judge + faithfulness judge (v2)
                                                               refusals, latency, cost (from tagged traces)
         ▲                                                                   │
 03_judges: meta-evaluation (do the judges work?) ◄── judge choice ──────────┤
 16 labeled answers incl. "right number, wrong label"                        ▼
                                                               05_regression: gate vs baseline.json (exit 1)
```

**The principle: every measurement is itself measured.** The golden set is reviewed, the relevance labels are audited, the judges are tested against labeled cases, and the numbers come with confidence intervals. Each layer turned out to have real errors (sections below).

### Components

| Layer | Module | Key functions | Notes |
|---|---|---|---|
| Metrics | `common/evaluation.py` | `recall_at`, `precision_at`, `hit_at`, `mrr_at`, `ndcg_at` | Graded relevance {chunk: 1 or 2} |
| Statistics | `common/evaluation.py` | `bootstrap_ci`, `paired_bootstrap` | Resample questions, 2,000 draws |
| Deterministic check | `common/evaluation.py` | `number_matches` | Reference figure in the answer (millions ↔ billions, rounding) |
| Judges | `common/evaluation.py` | `judge_correctness`, `judge_faithfulness` (v1), `judge_faithfulness_v2` | gpt-4.1, strict JSON, cached |
| Golden sets | `phase4_evaluation/01_build_golden.py` | `SPLITS`, `REVIEW_DROPS`, `TEST_REVIEW_DROPS`, `relevant_for`, `answer_bearing` | Committed `golden_{dev,test}.jsonl` |
| Pipelines under test | `04_end_to_end.py` | `run_pipelines` (P1 `04_ask`, P2 `05_ask`, P3 `06_ask`), `grade`, `report` | Traces tagged `eval:<split>` |
| Gate | `05_regression.py` | `RULES`, `baseline.json` | Tolerances for noise |
| Engineering | `common/llm.py`, `common/trace.py` | `OpenAI(max_retries=8)`, `TRACE_TAG` | Rate limits, eval traffic separated |

### Configuration

`.env` → `OPENAI_JUDGE_MODEL=gpt-4.1` → `common.config.JUDGE_MODEL`. The judge should be **different from, and stronger than,** the model it grades (see Step 3). gpt-5 models were available but don't accept `temperature=0`, which the judge needs for repeatable verdicts.

---

## Step 1: Golden sets (`01_build_golden.py`)

```bash
uv run python -m phase4_evaluation.01_build_golden --split dev
uv run python -m phase4_evaluation.01_build_golden --split test
```

### 1a. Synthetic questions, and the checks that keep them honest

For each company (5) × section group (`STRATA`: business, risk, MD&A, financials) × 4, a random chunk is sampled (seeded):
```
chunk ──► gpt-4.1 (GEN_SYSTEM, :48): one question, answer, VERBATIM evidence quote, numeric|text, usable?
      ──► code: evidence (whitespace-normalized) ⊂ chunk?             else drop ("bad_evidence")
      ──► code: numeric answer's figures ⊂ evidence's figures?        else drop ("bad_numbers")
      ──► gpt-4.1 critic (CRITIC_SYSTEM, :66): standalone, one answer, genuine question?
      ──► human review: REVIEW_DROPS / TEST_REVIEW_DROPS with a reason each
```
- **Standalone questions:** the generator must name the company and the *fact's* fiscal year, which may differ from the filing's year (a 10-K table shows three years).
- **Paraphrase:** it's told to phrase questions like a person would, not by copying the passage. Mean **lexical overlap** is still **0.61–0.64** (the share of question words found in the source chunk). That's a known bias of synthetic sets: questions resemble their source, which favors keyword search (BM25 is suspiciously strong on dev).

**Yield:**

| | dev | test |
|---|---|---|
| Sampled chunks | 80 | 80 |
| "Unusable" (no askable fact) | 22 | 22 |
| Bad evidence / bad numbers | 1 / 0 | 3 / 2 |
| Critic rejected | **0** | 2 |
| **Human dropped** | **9** | **8** |
| Kept | 48 | 43 |

**What human review caught that the critic didn't:**
- **Open-ended questions** ("What is *one* competitive advantage of Azure…"): many answers are valid, and the judge knows only one reference.
- **Yes/no questions:** 50% guessable.
- **Near-duplicates across filings.**
- **A wrong reference answer:** "What was Tesla's net income for fiscal year **2023**?" → $7,091M. That's **FY2024**; the generator misread the column order. The code check passed it (7,091 *is* in the evidence), and the critic passed it too.

**Lesson: synthetic evaluation sets need human review, and the reference answers can be wrong.** I checked every table-based numeric answer's column reading; this was the only error.

### 1b. Hand-written comparisons and unanswerables

- **Comparisons** (6 dev, 4 test): two-company questions with ground truth verified by text search, e.g. "Compare Tesla's and NVIDIA's overall gross margin" (Tesla **total** 18.0%, not a segment) and "Apple's vs Tesla's revenue growth" (+6% vs **−3%**).
- **Unanswerables** (8 dev, 5 test): companies we don't cover, years not indexed (fiscal 2019), future values, stock prices, executive pay (**Items 10–14 are incorporated by reference, so not in the 10-K text**), investment advice.

### 1c. Relevance labels (for retrieval metrics)

Same company only:
- **grade 2:** chunks containing the evidence quote, in the source chunk's filing.
- **grade 1:** the same quote in the other filing (10-Ks repeat text), or, for numeric answers, any chunk containing **all** of the answer's distinctive figures (≥ 4 significant digits; `answer_bearing()`, `:156`).

**The answer-bearing rule was added after a labeling error:** "Apple's Americas net sales, fiscal 2024" = $167,045M also appears in the **FY2025** 10-K as the prior-year column. The quote-only labels called that chunk irrelevant, and made Phase 3's *correct* retrieval look like a miss (P3 hit@6 went 0.61 → 0.70 after the fix). **Incomplete labels make good systems look bad.** This is the second time in the project (see Phase 3).

### 1d. dev vs test: why two splits

- **dev (62):** used to **diagnose and fix**. Phase 4 did fix things on dev: a router gap (Activision) and a planner filing-year rule (see Step 2). Scores on dev are optimistic *for the parts that were tuned*.
- **test (52):** built afterwards from **chunks dev never used** (seed 7), reviewed for **question quality only**, and **reported once**. Once you tune on it, it becomes dev and you need a new test set.

### 📘 Deep dive: building a RAG evaluation set

**Sources of questions, and their trade-offs:**

| Source | Pros | Cons |
|---|---|---|
| Real user queries (logs) | The true distribution | Need answers labeled; privacy; cold start |
| Expert-written | High quality, targeted (comparisons, traps) | Slow, small, author bias |
| **Synthetic from chunks** (ours; Ragas testset generation is similar) | Cheap, scalable, grounded by construction | **Lexical overlap bias**, easy "lookup" questions, wrong references, no multi-hop |
| Adversarial / unanswerable | Tests refusals and robustness | Must be written deliberately |

**The checks that matter:**
- **Verifiable evidence:** a verbatim quote checked in code.
- **Cross-checks:** answer figures must appear in the quote.
- **An LLM critic** (useful but lenient: 0 rejections on dev).
- **Human review with recorded reasons.**
- **Stratification** (company × section) so the set isn't all financial tables.
- **Dedupe.**
- **Measure the lexical overlap** so you know how easy the set is.

**Labels:** retrieval metrics need relevance judgments, and you can never label every chunk. Use **answer-bearing** definitions where you can (does the chunk contain the answer?), audit the misses, and prefer **end-to-end metrics** as the ground truth for decisions.

**Size and statistics:** with ~50 questions, 95% CIs on accuracy are about ±0.13–0.15. Separating a 2-point difference would take thousands of questions. Plan the size for the difference you need to detect.

---

## Step 2: Retrieval evaluation (`02_retrieval_eval.py`)

```bash
uv run python -m phase4_evaluation.02_retrieval_eval               # dev
uv run python -m phase4_evaluation.02_retrieval_eval --split test  # test (once)
```

Six configurations (`configs()`, `:43`), each returning a top-6 list. Phase 3's sub-question blocks are **interleaved** (`interleave()`, `:32`) so rank positions are comparable. Metrics come from `common/evaluation.py` with graded relevance, with bootstrap CIs and paired differences against P1.

### 2a. Results

nDCG@6, mean [95% CI]:

| Config | dev (after fixes) | **test** |
|---|---|---|
| P1 dense | 0.37 [0.27, 0.47] | 0.33 [0.23, 0.44] |
| P2 BM25 | 0.53 | 0.48 |
| P2 hybrid (RRF) | 0.45 | **0.50** [0.40, 0.60], best on test |
| P2 hybrid + rerank | 0.49 | 0.38 |
| P2 + auto filters | 0.52 | 0.45 |
| P3 plan | **0.55**, best on dev | 0.42 |

**Test, paired vs P1:**
- **Significant:** BM25 (+0.14), RRF (+0.16), auto filters (+0.12).
- **Not significant:** hybrid + rerank (+0.05, p = 0.14) and P3 plan (+0.09, p = 0.05).

### 2b. What was fixed on dev (Phase 3 code, found here)

- **Filing years for explicit years:** the planner sent "fiscal year 2024" questions to the FY2025 filing only, missing facts stated only in the FY2024 10-K (the Apple Watch lineup, Tesla's graduate hires). The fix, in code (`query.py:199`): year X → search the X and X+1 filings. Dev nDCG for P3 went 0.47 → 0.55.
- **Router gap:** "Which trustee is named in the 2017 base indenture for **Activision Blizzard**'s senior notes, as referenced in **Microsoft's** 10-K?" was routed `unsupported_company` (Activision isn't covered). The Phase 3 override rule only fired when *no* uncovered company was listed. The fix (`query.py:186`): a covered company named → never refuse.
- **Both fixes pass the Phase 3 regression cases** (05_route improved to 21/22).

### 2c. On test: two surprises

1. **The reranker lowered retrieval nDCG** (RRF 0.50 → reranked 0.38). Of the 8 test questions where reranking pushed the labeled chunk out of the top 6, 2 were still answered correctly (the answer was in another chunk, so the labels are incomplete), and **6 got worse end to end**. It also pulled the right chunk *in* for 4. `ms-marco-MiniLM` is a web-search model, and on held-out financial questions it's roughly neutral to slightly harmful. **Hypothesis for the next iteration:** P2 without reranking, or a finance-capable reranker. It must be checked on dev and a fresh test set, **not chosen because of this test result.**
2. **The dev→test drop isn't all overfitting.** P3 dropped 0.55 → 0.42, but P2 hybrid + rerank, which Phase 4 never changed, dropped 0.49 → 0.38 too. **Compare against an untuned control before blaming tuning.** Here most of the gap is the test questions being harder, plus sampling noise (the CIs are ±0.1).

### 📘 Deep dive: retrieval metrics and their statistics

**The metrics** (definitions in `common/evaluation.py`, worked example in the Phase 2 walkthrough):
- **hit@k:** any relevant chunk in the top k.
- **recall@k:** share of all relevant chunks in the top k. Our duplicate-filing labels make it **penalize year filters by design**, since duplicates in the other filing are filtered out.
- **precision@k:** share of the top k that is relevant. Capped when few chunks are relevant (~2 per question here).
- **MRR@k:** 1/rank of the first relevant chunk.
- **nDCG@k:** Σ (2^grade − 1)/log₂(rank + 1), normalized by the ideal ordering. **Graded** relevance (2 > 1), with a rank discount. The best single number for ranked retrieval with graded labels, and the one used for decisions here.

**Bootstrap confidence intervals** (`bootstrap_ci`, `:44`): resample the *questions* with replacement 2,000 times and take the 2.5th and 97.5th percentiles of the mean. No distributional assumptions. It answers "if I'd drawn a different 50 questions, how much would this number move?"

**Paired comparisons** (`paired_bootstrap`, `:51`): compare two systems **on the same questions** by bootstrapping the per-question differences. Question difficulty cancels out (a hard question is hard for both), so paired tests detect much smaller differences than comparing two separate CIs. `p` here is the share of bootstrap draws with difference ≤ 0.

**Pitfalls seen here:**
1. **Incomplete labels** (the Apple FY2024/FY2025 comparative column).
2. **Retrieval and end-to-end metrics disagree:** the reranker lowered nDCG while P2's answers beat P1's. Retrieval metrics **diagnose**; end-to-end metrics **decide**.
3. **A metric artifact:** recall with duplicate-filing labels penalizes filters.
4. **Synthetic lexical overlap inflates BM25.**

---

## Step 3: Judging the judges (`03_judges.py`)

```bash
uv run python -m phase4_evaluation.03_judges
```

16 labeled answers built from **real failures in Phases 2–3**: the Microsoft segment reported as the total, Amazon's net capex reported as gross purchases, Tesla's segment margin as the total margin, guidance reported as actuals, plus their correct counterparts and other error types (wrong year, invented number, billion/million, a contradiction). Six checkers:

```
checker                 accuracy  catches wrong  false alarms   cost/check
strict numeric             5/16         2/11         2/5         free
tolerant numeric           6/16         1/11         0/5         free
judge v1 (4o-mini)         9/16         5/11         1/5         ~$0.0001
judge v1 (gpt-4.1)        15/16        10/11         0/5         ~$0.001
judge v2 (4o-mini)        10/16         7/11         2/5         ~$0.0001
judge v2 (gpt-4.1)        15/16        11/11         1/5         ~$0.001
```

### 3a. Four findings

1. **Only a strong LLM judge catches "right number, wrong label".** Both numeric guardrails pass the Microsoft segment and Tesla segment cases (the numbers exist). gpt-4.1 catches them: *"'Revenue increased $19.2 billion or 16%' refers specifically to the Productivity and Business Processes segment."*
2. **The judge found errors in my labels.** It flagged three "faithful" answers because those chunks **don't contain the table's column-header row** (`2025 | 2024 | 2023` sits in the previous chunk), so the passage can't show which year a number belongs to. The answers are *true*, but not *supported by the passage*; the generator guessed the column order correctly by convention. I relabeled them (recorded in the code). **Faithfulness means "supported by the given passages", not "true".** The chunking bug behind it is Phase 5's job: tables split from their headers.
3. **Reasoning–verdict inconsistency:** judge v1 (gpt-4.1) wrote *"the metric is 'purchases of property and equipment, net of proceeds from sales and incentives'"*, then returned **supported** for an answer claiming plain purchases.
4. **The fix: the LLM extracts, code decides** (judge v2, `judge_faithfulness_v2`, `:165`). Per claim the judge returns a quote plus booleans (`value_matches`, `entity_matches`, `metric_matches`, `period_matches`, `unit_matches`, `contradicted`), and **code** computes the verdict: supported only if all hold. v2 (gpt-4.1) catches **11/11**, with readable reasons like `failed: metric_matches | quote: Purchases of property and equipment, net of proceeds…`. Its one false alarm was a precision quibble: it read the computed "12.4%" as contradicting the table's rounded "12".

**Why gpt-4.1 and not gpt-4o-mini:** mini catches 5–7 of 11 as a judge. The same model that writes the answers also misses its own kind of errors (**self-preference bias**).

### 📘 Deep dive: LLM-as-judge

**What it is:** an LLM grades outputs against a rubric (correctness vs a reference, faithfulness vs the context, relevance, style). It's cheap and scalable compared with human grading.

**Known biases and mitigations:**

| Bias | Mitigation |
|---|---|
| **Self-preference:** grading its own model family leniently | A different, stronger judge (gpt-4.1 vs gpt-4o-mini here) |
| **Verbosity:** longer answers look better | Rubrics about facts, not style; claim-level checks |
| **Position** (pairwise) | Swap the order and average |
| **Reasoning–verdict inconsistency** | Reasoning *before* the verdict (schema order), or **computed verdicts** from a structured rubric (v2) |
| **Leniency** (the critic rejected 0) | Test the judge on known-bad cases; calibrate |
| **Non-determinism** | temperature 0 and caching (`chat_json`) |

**Meta-evaluation:** treat the judge as a classifier. Build labeled cases that include **your system's real failure modes**, and measure recall on bad answers and false alarms on good ones before trusting its scores. Re-run it when you change the judge prompt or model.

**Faithfulness vs correctness:**
- **Correctness:** answer vs the reference. It needs references, and it catches wrong answers even when they're "faithful" to wrong context.
- **Faithfulness / groundedness:** answer vs the retrieved context. It needs no reference, so it can run **online** on production traffic, and it catches hallucination. It *can't* catch a faithful answer built from the wrong chunk.
- You want both. Frameworks (Ragas faithfulness and answer correctness, DeepEval, TruLens groundedness, Phoenix evals) implement variants of these. We built ours to see the internals and to target "wrong label".

---

## Step 4: End-to-end evaluation (`04_end_to_end.py`)

```bash
uv run python -m phase4_evaluation.04_end_to_end --split test                     # the reported number
uv run python -m phase4_evaluation.04_end_to_end --split dev --no-faithfulness
uv run python -m phase4_evaluation.04_end_to_end --split test --reuse            # re-grade saved answers
```

For every question and pipeline (`run_pipelines`, `:39`): the answer, whether it refused, the passages used, latency, and **cost from the pipeline's own trace** (tagged `eval:<split>`). The answers are **saved before grading**. Then `grade()` (`:67`) runs the correctness judge (reference-based), `number_matches` (deterministic), and judge v2 faithfulness (test only).

### 4a. Results

**Test (47 answerable + 5 unanswerable), the reported numbers:**

| Pipeline | Correct [95% CI] | Refused unanswerable | Refused answerable | Faithfulness | Latency p50 | $/question |
|---|---|---|---|---|---|---|
| P1 naive | 0.53 [0.38, 0.68] | 5/5 | 17/47 | 0.69 | 1.0 s | $0.00028 |
| P2 hybrid + rerank | 0.68 [0.53, 0.81] | 5/5 | 7/47 | 0.73 | 1.4 s | $0.00028 |
| P3 query intelligence | **0.70** [0.55, 0.83] | 5/5 | 8/47 | **0.85** | 1.4 s | $0.00032 |

- **Paired, test:**
  - **P3 vs P1: +0.17 [+0.04, +0.32], significant.**
  - P2 vs P1: +0.15 [−0.02, +0.32], borderline (p = 0.05).
  - **P3 vs P2: +0.02 [−0.09, +0.13], not significant.**
- **Dev (54 + 8)**, correctness only: P1 0.54, P2 0.76, P3 0.76. P2 and P3 are both significantly better than P1 (+0.22), and P3 = P2 exactly.
- Latency is low because most query plans and embeddings were cached from earlier runs. Uncached P3 is ~3.7 s (Phase 3).

| | dev | test | Change |
|---|---|---|---|
| P1 | 0.54 | 0.53 | −0.01 |
| P2 (not changed in Phase 4) | 0.76 | 0.68 | −0.08 |
| P3 (tuned on dev in Phase 4) | 0.76 | 0.70 | −0.06 |

P2, untouched, drops as much as P3. **The gap is mostly split difficulty and noise, not overfitting.**

### 4b. What the results say

1. **The Phase 1 → 3 journey is real:** +17 points correct, significant on held-out questions.
2. **Most of it is refusing less:** P1 refused 17/47 answerable questions and P2 7/47. Hybrid search and reranking find evidence that dense-only search misses. Wrong answers stay rare (2–4 per pipeline).
3. **P3's planner doesn't make answers measurably more correct than P2's rules,** but it makes them **more faithful (0.73 → 0.85)**. Its answers stick closer to the passages.
4. **P3 answers some questions P2 refused, and some of those answers are mislabeled** (dev: 4 incorrect vs 1):
   - **Tesla restructuring "$583M":** that's the *employee termination expenses* within "Restructuring and other" ($684M total). A sub-component reported as the total.
   - **NVIDIA dividends "$0.016":** another fiscal year's row of a stockholders'-equity statement that stacks several years **without year labels in the chunk**.
   - **"Right number, wrong label"** again, and again caused by chunks that lose their table context. That's Phase 5.
5. **Comparisons are still weak** (test: P3 1/4). Two answers **honestly said one company's figure was missing** (Microsoft total assets, NVIDIA R&D not retrieved), which is correct behavior with a retrieval miss behind it. One ("Tesla net income $3,855M vs reference $3,794M") is really **my reference's ambiguity**: $3,855M is total net income, and $3,794M is the amount attributable to common stockholders. The question didn't say which.
6. **The deterministic figure check (are the reference figures in the answer?) agrees with the LLM judge on nearly every numeric question:** test 25/27 (P1), 27/27 (P2), 24/27 (P3); dev 34/35, 35/35, 35/35. For figure questions a free check almost reproduces the judge. Keep it as a cheap first pass, and use the judge for text, labels and disagreements.

**New P3 failure modes on test** (traced, not fixed; fixing them on test would turn test into dev):
- **Over-decomposition:** "Apple's market share in smartphones, PCs, tablets and wearables" became **four** sub-questions, splitting the 6 result slots so that the one chunk answering all four never ranked.
- **The dev fix has a cost:** the X/X+1 filing rule widened "Tesla's sales model in fiscal 2024" to the FY2025 filing, and FY2025 chunks crowded out the FY2024 passage. **A fix that helped on dev hurt on test**, which is exactly what a held-out split is for.

### 4c. Engineering an evaluation run

- **Rate limits:** the first test run crashed on gpt-4.1's **30,000 tokens per minute**. Each faithfulness check sends ~3k tokens of passages, so it's ~10 checks per minute. The fix: `OpenAI(max_retries=8)` (exponential backoff that honours `retry-after`).
- **Save before grading:** the crash lost all generated answers. Answers are now saved to `results/<split>_answers.json` before any judge call, and `--reuse` re-grades them without regenerating (regenerated answers differ slightly, so reuse also keeps comparisons fixed).
- **Caching:** judge outputs are cached (`chat_json`), so re-running a report costs nothing, and a prompt change invalidates the cache automatically.
- **Tagged traces:** `TRACE_TAG=eval:test` puts a tag on every trace from the run. `06_traces` shows untagged traffic by default (`--tag eval:test` to see the eval runs), so evaluation traffic doesn't pollute real-usage metrics.
- **Cost of this phase:** the golden-set generation and judging on gpt-4.1 cost roughly a dollar or two. Pipeline answers on gpt-4o-mini are a few cents.

### 📘 Deep dive: evaluation-driven development

The loop:
```
change (prompt, chunking, retriever, model)
   └─► run on DEV: retrieval metrics (diagnose) + end-to-end (decide) → inspect failures in traces
         └─► keep or revert
               └─► occasionally, on a FRESH TEST set: the honest number
                     └─► baseline.json + regression gate in CI
```
**Rules that came from this phase:**
1. **Separate dev and test.** Once you've looked at failures on a set, it's dev.
2. **Use an untuned control** to interpret a dev→test gap (P2 here).
3. **Report CIs and paired differences.** "+2 points" with a CI of ±10 isn't a result.
4. **Evaluate the evaluators** (judges, labels, references). Each had errors here.
5. **Decide with end-to-end metrics, diagnose with component metrics.**
6. **Make runs robust and cheap:** retries, saved intermediates, caching, tagged traces.

---

## Step 5: The regression gate (`05_regression.py`)

```bash
uv run python -m phase4_evaluation.05_regression --update-baseline   # accept the current numbers
uv run python -m phase4_evaluation.05_regression                     # exit 1 on regression
```
`baseline.json` (committed) holds P3's test metrics. `RULES` (`:23`):

| Metric | Rule | Why a tolerance |
|---|---|---|
| correct | ≥ baseline − 0.05 | ±1–2 questions flip between runs (sampling, API, judge) |
| refused_ok | ≥ baseline | Refusing unanswerables is a safety property: no tolerance |
| false_refusals | ≤ baseline + 1 | |
| faithfulness | ≥ baseline − 0.05 | |
| latency_p50_s | ≤ baseline × 1.5 | **Only comparable under the same cache state**: this baseline was measured warm |
| cost_per_q | ≤ baseline × 1.5 | |

A simulated regression (correct −0.10, latency ×2) fails with `FAILED: correct, latency_p50_s` and exit code 1. **In CI** (Phase 7), a small smoke subset runs on every change and the full set nightly. The tolerances must be wider than run-to-run noise, or the gate gets ignored.

---

## What to rerun after a change

| You changed... | Rerun |
|---|---|
| Any pipeline code (Phases 1–3) | `04 --split dev` → inspect → when done, `04 --split test` → `05_regression` |
| The judge prompt or model | `03_judges` first (does it still catch the cases?), then `04 --reuse` (re-grade saved answers) |
| Chunks or the index (Phase 5) | Rebuild `01` (labels depend on chunk ids and text), then `02`, `04` |
| Golden-set review decisions | `01 --split <split>`, then `02`, `04` |
| Accepting new numbers | `05_regression --update-baseline` (commit `baseline.json`) |

## What's still broken, and the phase that fixes it

| Failure | Evidence | Phase |
|---|---|---|
| Tables split from their column headers → year unknowable, wrong-row answers | Judge "false alarms" (Tesla margin, Amazon cash flow), NVIDIA dividends $0.016 | **5** (table-aware chunking, contextual headers) |
| Sub-components or segments reported as totals | Tesla restructuring $583M vs $684M; Microsoft segment (Phase 2) | 5 (context) + a faithfulness gate at runtime |
| The reranker is neutral-to-harmful on held-out finance questions | Test nDCG 0.50 → 0.38 after reranking | 5/7 (a different reranker; validate on dev and a fresh test) |
| Over-decomposition splits k across too many sub-questions | Apple market share → 4 sub-questions | 6 (the agent decides when to split, iterates) |
| Comparisons: one company's evidence not retrieved | Test: 1/4 correct | 5 (chunks) and 6 (an agent that notices the missing half and searches again) |
| ~50 questions can't separate P2 from P3 | CI ±0.13 | Grow the test set (real queries in Phase 7) |

## Glossary (quick reference)

| Term | One-line meaning |
|---|---|
| **Golden set** | Questions with reference answers (and relevance labels) used to measure a system |
| **dev / test split** | A set you may tune on / a set you only report on |
| **Synthetic QA generation** | An LLM writes questions and answers from sampled chunks |
| **Evidence quote** | A verbatim span of the chunk that supports the reference answer (validated in code) |
| **Lexical overlap bias** | Synthetic questions reuse their source's words, which favors keyword retrieval |
| **Answer-bearing relevance** | A chunk counts as relevant if it contains the answer, not just the source text |
| **nDCG** | Ranking quality with graded relevance and a log rank discount, normalized to the ideal |
| **Bootstrap CI** | Resample questions to estimate how much a mean would move |
| **Paired bootstrap** | Resample per-question *differences* between two systems on the same questions |
| **LLM-as-judge** | An LLM grades outputs against a rubric |
| **Meta-evaluation** | Measuring the judge itself on labeled cases |
| **Self-preference bias** | A model grades its own family's outputs leniently |
| **Correctness vs faithfulness** | Matches the reference / supported by the retrieved context |
| **Rubric-based judge (v2)** | The judge fills per-attribute booleans; code computes the verdict |
| **Regression gate** | Automated check that fails a change if metrics drop beyond tolerance |
| **Untuned control** | A system you didn't change, used to separate overfitting from split difficulty |

## Self-check before moving on
- [ ] Why did BM25 look so strong on the synthetic sets? What would you change to reduce that bias?
- [ ] Give two errors in the golden set that only human review caught.
- [ ] Why did the answer-bearing label rule change P3's hit@6 from 0.61 to 0.70? What does that say about incomplete labels?
- [ ] P3 vs P2: +0.02 on test with CI [−0.09, +0.13]. What can you conclude? How many questions would you need?
- [ ] The reranker lowered test nDCG but P2 still beat P1 end to end. Which metric do you trust for the decision, and why?
- [ ] Why is it wrong to drop the reranker *because of* the test result? What's the correct procedure?
- [ ] gpt-4.1 flagged three "true" answers as unfaithful. Was it right? What does faithfulness mean exactly?
- [ ] What problem does judge v2 (booleans + computed verdict) solve that v1 had?
- [ ] P2 and P3 both dropped ~6–8 points from dev to test. Why isn't that proof of overfitting?
- [ ] Why does the regression gate need tolerances, and why is the latency rule tied to cache state?
