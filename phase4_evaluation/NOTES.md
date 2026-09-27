# Phase 4: Evaluation

```
01_build_golden     golden_dev.jsonl (62) + golden_test.jsonl (52): synthetic (gpt-4.1, verbatim evidence,
                    critic, human review) + hand-verified comparisons + unanswerables
02_retrieval_eval   6 retrieval configs × hit/recall/precision/MRR/nDCG@6, bootstrap CIs, paired tests
03_judges           meta-evaluation: 6 checkers on 16 labeled answers incl. "right number, wrong label"
04_end_to_end       P1 vs P2 vs P3: correctness, faithfulness, refusals, latency, cost (traces tagged eval:<split>)
05_regression       gate vs baseline.json, exit 1 on regression
```
Start with `WALKTHROUGH.md` for the step-by-step explanation and the deep dives.

```bash
uv run python -m phase4_evaluation.01_build_golden --split dev      # and --split test
uv run python -m phase4_evaluation.02_retrieval_eval --split test
uv run python -m phase4_evaluation.03_judges
uv run python -m phase4_evaluation.04_end_to_end --split test
uv run python -m phase4_evaluation.05_regression
```

## 1. The headline (held-out test, 47 answerable + 5 unanswerable)

| Pipeline | Correct [95% CI] | Refused answerable | Faithfulness |
|---|---|---|---|
| P1 naive | 0.53 [0.38, 0.68] | 17/47 | 0.69 |
| P2 hybrid + rerank | 0.68 [0.53, 0.81] | 7/47 | 0.73 |
| P3 query intelligence | 0.70 [0.55, 0.83] | 8/47 | **0.85** |

- **P3 vs P1: +0.17, significant.** P2 vs P1: +0.15, borderline. **P3 vs P2: +0.02 [−0.09, +0.13], not significant.**
- All pipelines refused all unanswerable questions (5/5 test, 8/8 dev).
- **Most of the gain is fewer false refusals**, not fewer wrong answers (2–4 per pipeline).
- **P3's clear win is faithfulness** (0.73 → 0.85), not correctness.

## 2. What went wrong in the *measurement*, and how it was caught

| Layer | Error | Caught by | Fix |
|---|---|---|---|
| Golden questions | 9 dev / 8 test unusable (open-ended, yes/no, duplicates); **the critic rejected 0 on dev** | Human review | Recorded `REVIEW_DROPS` with reasons |
| Reference answers | Tesla FY2023 net income = $7,091M: **wrong** (that's FY2024; the column order was misread) | Human review | Dropped |
| Reference answers | Tesla "net income" ambiguity ($3,855M total vs $3,794M attributable) | Reading the judge's reasoning | Noted: write references that name the exact line item |
| Relevance labels | Quote-only labels missed the same figure in the next year's comparative column | Diagnosing P3's "misses" | Answer-bearing labels (P3 hit@6 went 0.61 → 0.70) |
| Judge labels | Three "faithful" cases weren't *supported*: the chunks lack the column-header row | The gpt-4.1 judge | Relabeled; faithfulness = supported by the passages, not true |
| Judge v1 | Reasoning said "net of proceeds", verdict said "supported" | Meta-evaluation | Judge v2: booleans per attribute, verdict computed in code |
| Eval run | Crashed on gpt-4.1's 30k TPM limit and lost the generated answers | The first test run | Retries, save-before-grade, `--reuse` |

## 3. What went wrong in the *system* (found in Phase 4)

- **Fixed on dev (Phase 3 code):**
  - The planner sent "fiscal year X" questions to the X+1 filing only → year X → filings {X, X+1}.
  - The router refused an Activision question about Microsoft's 10-K → a covered company named → never refuse.
- **Found on test, not fixed (fixing would contaminate test):**
  - **Over-decomposition:** Apple market share → 4 sub-questions → k split 4 ways.
  - **The dev fix's cost:** the X/X+1 widening crowded out an FY2024-only passage (Tesla's sales model).
  - **The reranker on finance questions:** it pushed the labeled chunk out on 8 test questions (6 got worse end to end) and pulled it in on 4. Test nDCG went 0.50 (RRF) → 0.38 (reranked).
- **"Right number, wrong label" is still the main error class:**
  - Tesla restructuring $583M (a component) vs $684M (the total).
  - NVIDIA dividends $0.016 (another year's row).
  - The root cause is table chunks that lose their headers and context → **Phase 5**.

## 4. Judges (meta-evaluation on 16 labeled answers)

| Checker | Catches wrong | False alarms |
|---|---|---|
| Strict / tolerant numeric guardrail | 2/11 · 1/11 | 2/5 · 0/5 |
| Judge v1 / v2 with gpt-4o-mini | 5/11 · 7/11 | 1/5 · 2/5 |
| Judge v1 with gpt-4.1 | 10/11 | 0/5 |
| **Judge v2 with gpt-4.1** | **11/11** | 1/5 |

A judge ~13× more expensive per check (~$0.001 vs ~$0.0001) is what catches wrong-label answers. **Use it offline and on a sample online; don't block every request on it** (1–3 s per check).

## 5. dev vs test

| | dev | test |
|---|---|---|
| P1 (untuned) | 0.54 | 0.53 |
| P2 (untouched in Phase 4) | 0.76 | 0.68 |
| P3 (tuned on dev in Phase 4) | 0.76 | 0.70 |

P2 drops as much as P3 without any tuning, so **the gap is mostly split difficulty and noise.** Without the untuned control, it would have looked like overfitting.

---

## Interview Q&A

**Q: How do you evaluate a RAG system?**
At two levels. **Retrieval**: recall@k, MRR and nDCG@k on questions with relevance labels, to diagnose. **End to end**: answer correctness against references, faithfulness of the answer to the retrieved context, refusal behavior on unanswerable questions, latency and cost, to decide. Use a golden set with a held-out test split, report confidence intervals, compare systems with paired tests, and validate the judges before trusting them. In my project the full Phase 1→3 pipeline improved correctness by 17 points on held-out questions (significant), while the last stage (an LLM query planner) didn't significantly improve correctness but raised faithfulness from 0.73 to 0.85.

**Q: How do you build a golden set when you have no labeled data?**
Generate questions from sampled chunks with a strong LLM, require a verbatim evidence quote and validate it in code, cross-check that the answer's figures appear in the quote, filter with a critic, and have a human review. Add hand-written hard cases (comparisons, traps) and unanswerable questions. Stratify across sources and sections, and measure how much the questions copy their source (lexical overlap), since that biases toward keyword search. In mine the LLM critic rejected nothing, and human review found 17 bad items, including a reference answer with a misread table column.

**Q: What is LLM-as-judge, and how do you know the judge is right?**
An LLM grades outputs against a rubric. You know it's right by meta-evaluating it: build labeled cases including your system's real failure modes and measure the judge's recall on bad answers and false alarms on good ones. Mitigate its biases: use a different, stronger model (self-preference), put reasoning before the verdict, or better, have the judge fill a structured rubric (does the value, entity, metric, period and unit match?) and compute the verdict in code. My v1 judge once reasoned correctly and then gave the opposite verdict; v2 fixed that and caught 11 of 11 wrong-label answers.

**Q: Faithfulness vs answer correctness?**
Correctness compares the answer with a reference answer. It needs labels and catches any wrong answer. Faithfulness (groundedness) compares the answer with the retrieved context. It needs no labels, so it can run online on production traffic, and it catches hallucination. But a faithful answer built from the wrong chunk still passes it. Also: "faithful" means supported by the passages, not true. My judge correctly flagged true answers whose chunk lacked the table header needed to know the year.

**Q: Your new retrieval stage improved dev by 8 points and test by 2. What happened?**
First check noise: with ~50 questions the CI is about ±0.13, so both could be the same number. Then compare an untuned control: in my case a pipeline I hadn't touched dropped just as much from dev to test, so the gap was split difficulty, not overfitting. If only the tuned system drops, you've overfit dev; build a fresh test set and don't tune on the old one.

**Q: How would you set up evaluation in CI for a RAG app?**
A versioned golden set in the repo, a script that runs the pipeline and saves answers before grading, a cached judge, and a regression gate comparing against a committed baseline with tolerances wider than run-to-run noise (mine: correctness −0.05, latency ×1.5, zero tolerance on refusing unanswerables). A small smoke subset on every PR, the full set nightly, and baselines updated deliberately. Tag the evaluation traffic in traces so it doesn't pollute production metrics.

**Q: Retrieval metrics and answer quality disagree. Which do you trust?**
End-to-end answer quality for decisions, and retrieval metrics to diagnose. They disagree for real reasons: incomplete relevance labels (the answer was in an unlabeled chunk), several chunks containing the same evidence, or a generator that recovers from imperfect ranking. In my test run the reranker lowered nDCG while the pipeline with it still beat the one without. I traced it to 6 genuinely worse questions, 4 better, and 2 label gaps.

## Experiments to try
- [ ] **(On dev only.)** Evaluate "P2 without the reranker" and "P3 without the X+1 filing rule" end to end. If either wins on dev, build a fresh test set before believing it.
- [ ] Merge sub-questions that share a ticker and filings in `plan()` (anti-over-decomposition), then measure on dev.
- [ ] Cut lexical overlap: regenerate questions with a stricter "no shared phrases" instruction, or paraphrase them in a second pass. Does BM25's lead shrink?
- [ ] Add 20 real questions you'd actually ask about these companies (no peeking at the chunks) as a third, "natural" split. How do scores compare with the synthetic splits?
- [ ] Put judge v2 (gpt-4.1) in `06_ask` as an optional runtime check on 10% of requests. What do the latency and cost look like in `06_traces`?
- [ ] Re-grade the test answers with judge v2 on gpt-4o-mini (`--reuse`, change the model). How much does the faithfulness score move? That's the judge's effect on your metric.
