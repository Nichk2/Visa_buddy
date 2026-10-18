# Visa Buddy v4: plan and pre-registered criteria

Written 2026-10-16, before any v4 code. The criteria below are fixed: v4 is judged against
them as written, and any later change is recorded here with its reason, never applied silently.

## Why v4 exists

v1 to v3 were the lab phase. Each was measured on frozen test sets, and each taught one thing:

| Version | What it showed |
|---|---|
| v1 | Its numbers were never measured (8/10 on 0 test queries, a loss curve drawn from a formula). |
| v2 | Fine-tuning fixes behaviour: citations up, over-refusal 0.111 -> 0.037. It also introduced a safety regression (rf08) found only by held-out tests. |
| v3 | More training data fixed rf08 but not stale knowledge: on rules no model was trained on, every model answers r03 (whole program online from outside Canada, PGWP?) with "Yes"; v3 even cites the COVID policy's 2022 end date while doing it. |

The conclusion: for rules that change, facts must come from today's official pages, never
from the model's memory. Fine-tuning stays central, but its job changes from *knowing the
rules* to *reading the sources faithfully*.

## Design

1. **Retrieve**: BM25 keyword search alongside the MiniLM embeddings, plus a small re-ranker.
2. **Extract**: the model quotes the official sentence(s) that answer the question. Code checks
   each quote appears word for word in the retrieved pages; invented quotes are rejected.
3. **Answer**: a short answer written only from those quotes.
4. **Verify**: code compares the answer with its quotes (yes/no verdict, negation, numbers).
   On a mismatch the user sees only the official quotes, never the wrong sentence.
5. **Model**: a current 3-4B instruct model in 4-bit (MLX), served on the developer's MacBook
   (M3, 8 GB). Chosen by a bake-off on the frozen sets, without fine-tuning.
6. **Fine-tuning**: LoRA on Modal, only if it beats the untuned model on these criteria. Data:
   quote-first answers and knowledge-conflict examples ("the source says otherwise, the source wins"),
   made by a gated teacher, with every eval page and question held out.

## Frozen test sets

| Set | Rows | Purpose |
|---|---|---|
| gold_v2 | 54 answerable + 12 must-refuse | general questions (v2 answer key) |
| safety_v1 | 20 must-refuse + 6 honest twins | misrepresentation, evasion, legal strategy, predictions |
| rules_v1 | 23 (21 "no", 2 "yes") | rule questions from never-trained pages; 5 recent changes |
| rules_yes_v1 | 11 (all "yes") | stops a model passing the verdict check by always saying "No" |

## Pre-registered criteria (v4 passes only if all hold)

| # | Criterion | v4 must reach | Best before v4 |
|---|---|---|---|
| 1 | Wrong rules **shown to the user** across gold_v2, safety twins, rules_v1 and rules_yes_v1 (wrong verdict or contradiction; a quote-only fallback counts as safe) | **0** | 2 contradictions (v3, gold + safety); wrong verdicts on rules_v1: 1 / 1 / 2 (baseline / v2 / v3) |
| 2 | Must-refuse refused, gold_v2 + safety_v1 | **32 / 32** | 32 (v3) |
| 3 | Fact coverage, gold_v2 | >= 0.671 | 0.671 (v3) |
| 4 | Fact coverage, rules_v1 | >= 0.630 | 0.630 (v2) |
| 5 | Refused although the answer is in the sources (gold_v2 and rules_v1, each) | <= 0.05 | 0.037 / 0.000 |
| 6 | Retrieval recall@5 on gold_v2: overall / citizenship | >= 0.889 / >= 0.70 | 0.889 / 0.62 |
| 7 | Runs on the MacBook (M3, 8 GB), no Colab | median reply <= 15 s | n/a |
| 8 | rules_yes_v1: wrong verdict / fact coverage | 0 / >= 0.6 | 0 / 0.727 (v2 and v3) |

Comparisons are system to system: if v4 changes the corpus or retrieval, v3 is re-scored on
the same corpus as well, and both results are reported.

## Sealed final exam (frozen 2026-10-17, before fine-tuning)

`eval/final_exam_v1.jsonl` (sha256 `0a3c04b8ea9b...`): 25 hand-written questions from pages added in
corpus_v2 (never in any training input, never used to shape a fix): 19 rule/fact questions (3 on the
2026 Express Entry points changes) and 6 must-refuse. Several fixes so far were made after reading
failures on the other four sets, so those sets are no longer a clean exam. This one is scored
**exactly once**, at the end of v4, with the shipped system on the Mac, and never used for decisions.

| Final-exam criterion | v4 must reach |
|---|---|
| Wrong rules shown | 0 |
| Must-refuse refused | 6 / 6 |
| Fact coverage (19 answerable) | >= 0.60 |
| Refused although the answer is in the sources | <= 0.10 (2 of 19; small set) |

The result is reported as measured, pass or fail.

## Phases

0. Freeze held-out rule sets and score the earlier models on them. **Done** (rules_v1, rules_yes_v1).
1. This document.
2. Refresh the corpus (re-crawl, raise the per-program cap, add missing 2024-25 pages; HTML over PDF)
   and improve retrieval; measure recall@5 (criterion 6).
3. Model bake-off on the Mac, no fine-tuning.
4. Extract -> answer -> verify pipeline; measure criteria 1-5, 7, 8.
5. SFT data (quote-first + knowledge-conflict), gated teacher, data card.
6. LoRA on Modal; compare untuned vs tuned vs v3.
7. Serve on the Mac.

## Finding from phase 0

Every model gets all 11 "yes" rules right, and every wrong verdict on rules_v1 is a "Yes"
where the official rule says "No" (r03, r05). The models lean permissive: they grant
permissions the source doesn't grant. The knowledge-conflict training data (phase 5) should
lean on rules that forbid or restrict.

## Phase 2 results (2026-10-16)

- corpus_v2: 610 pages crawled (was 495); 113 new work-permit pages (the old 130-page cap),
  and pages now cover the 2024-25 visitor-to-work-permit and spouse open-work-permit changes.
  35 existing pages changed text in the 5 days since the first crawl. No eval fact was affected
  (2 eval source pages changed: an expired maintenance notice on the eTA pages).
- Page labels: 6 pages excluded (1 archived, 5 historical rules); 50 employer-facing or pilot
  pages kept as index-only (never sources for training answers).
- Retrieval, gold_v2 recall@5 (corpus_v2): dense 0.889 -> hybrid 0.944; citizenship 0.62 -> 0.88.
  **Criterion 6 met.** Known loss: vi07 (visitor visa fee) drops from rank 4 to 6 under hybrid;
  left as is rather than tuned to one test question.

## Phase 3 results: bake-off (2026-10-17)

Untuned, v4 pipeline, hybrid retrieval on corpus_v2, vLLM on Modal (L4). Run 1 was invalid (quote
verifier bug: multi-line and plain-string quotes rejected); fixed and re-run.

| | Qwen3-4B-Instruct-2507 | Phi-4-mini-instruct | SmolLM3-3B |
|---|---|---|---|
| Fact coverage gold / rules / yes | **0.744 / 0.609 / 0.727** | 0.167 / 0.087 / 0.545 | 0.349 / 0.304 / 0.000 |
| Wrong rules shown | 4 (2 grader false alarms) | 0 | 2 |
| Must-refuse refused /32 | 31 | 32 | 18 |
| Over-refusal gold / rules | 0.130 / 0.174 | 0.778 / 0.783 | 0.519 / 0.522 |

**Chosen: Qwen3-4B-Instruct-2507.** Untuned it already passes gold fact coverage (0.744 vs v3's
0.671). Its real errors are one new kind: a real quote from the wrong part of the right page:
r03 (the expired COVID distance-learning policy), y02 (non-DLI section for a DLI-to-DLI transfer),
rf10 ("Resume citizenship" fee given as the new-application fee). Next: date-aware quote filter
(code), grader v3.1 for two false-alarm patterns (ee06, fa05), then fine-tuning for quote selection
and over-refusal.

Follow-up (2026-10-17): the ee06/fa05 "contradictions" were not a grader bug. The pipeline joined
multi-line quotes with spaces, merging two bullets into one clause; quotes now keep their line
breaks, and both answers grade correct (offline re-render of the saved outputs). Expired quotes:
a quote describing a temporary policy with an end date before today is dropped, and the model gets
one retry without those passages. On the saved outputs it flags exactly one quote (r03). Both
changes need a model re-run to be measured; they will be measured with the next model run.

## Changes to this plan

- 2026-10-18, **gate exception (the project owner's decision)**: teacher gates for phase 5 data. Mistral Small 3.2
  failed (3 wrong rules, fact coverage 0.747, over-refusal 0.111). Qwen3-30B-A3B failed run 1 (2 wrong rules,
  29/32 refusals); after the shared-trap pipeline fixes, run 2 failed by one refusal (30/32 vs >= 31) while
  beating the untuned student on the other three measures (fact coverage 0.796 vs 0.762, over-refusal 0.074 vs
  0.130, wrong rules 1 vs 2). Decision: train on the Qwen3-30B rows anyway, because (1) refusal training rows are
  kept only where the teacher actually refused, (2) its misses (rf10, the resume-citizenship fee; s07, a
  deterrent answer to a misrepresentation question) are shared with or arguably better than the student, and
  (3) the safeguard is the comparison below, not the gate. **Rule added: the fine-tuned model ships only if it
  meets the pre-registered criteria at least as well as the untuned Qwen3-4B on the same sets; otherwise the
  untuned model ships and that is reported.** No further pipeline changes are made from failures on gold_v2,
  safety_v1, rules_v1 or rules_yes_v1; the sealed final exam decides.
- 2026-10-17: phase 4's pipeline was built before the bake-off (phase 3), because a model has to be
  judged inside the pipeline it will run in.

(none yet)
