# AMPARA – Lean Q1 Revision Plan

As of 2026-11-06. Shared version: [Claude doc](https://claude.ai/artifact/9RKeGx7gSvepEaUc3jSjp8).

## Goal and ground rules

Resubmit within three weeks with every MAJOR reviewer comment answered, working in three parallel tracks without
changing the model or the corpus. Track 1 covers provenance and writing, Track 2 retrieval experiments, Track 3 answer
quality and human data.

1. **Freeze the model and the corpus.** checkpoint-256 and the 893 chunks stay as they are. New training is limited to
   extra seeds and one naive baseline.
2. **Run evaluation-only experiments first.** They reuse the released runs and take hours, not days.
3. **Make strict relevance labels primary.** Strict means the source chunk plus its near-duplicates. Pooled and graded
   labels become secondary, so new test sets need no pooling.
4. **Register three confirmatory hypotheses before seeing new results.** Everything else is reported as exploratory.

Open question: who owns each track (one author per track).

## Where we stand

The paper's retrieval results reproduce from the public repos. What remains is provenance and missing experiments, not
wrong numbers.

| Check (reviewer comment) | Result from the released data |
| --- | --- |
| Tables 9, 10 and 12 recomputed from the released runs | All 144 cells within 0.0005 of the paper; identical confidence intervals |
| Memorization: corpus without training chunks | Gain +0.128 NDCG@10 [0.079, 0.178], vs +0.119 on the full corpus |
| #42 pooled positives from the training partition | 352 of 499; dropping them gives +0.120 [0.071, 0.171] (pooled) |
| #40 200 strict pairs for 198 questions | 2 questions (Penal Code art. 58) also count art. 64, a near-duplicate |
| #30 Table 2 vs Table 3 chunk counts | 14 / 1 / 1 near-duplicate members; Table 3 counts one per group |
| #32 false negatives, clustered by question | Yes 2.3% [1.9, 2.8]; partial 16.0% [14.8, 17.2] |
| #27 lexical overlap | Gain similar in low- and high-overlap halves; BM25 NDCG@10 0.02 vs 0.12 |
| Test questions by generator | Sonnet 4.5 (n = 48): +0.084 [−0.013, 0.176]; Sonnet 5.5 (n = 150): +0.130 [0.060, 0.199] |
| #6 retrieval vs Faithfulness | 0.64 when the source chunk is retrieved; 0.47 on a strict miss; 0.35 on a pooled miss |
| #47 domain filter | Same 24 rejected in all 9 configurations; retriever Recall@10 0.92 on them vs 0.78 on accepted |
| Statistical power of Test A | Detects about 0.08 NDCG@10 (fine-tuned vs base) and 0.03–0.05 between close variants |
| Citation accuracy of the published answers (Table 13 run) | 101 of 174 answers cite articles (158 citations); 79.7% are in the retrieved chunks, 93.0% counting mentions inside them; 7.9% of answers with citations have an unsupported one |

Intervals are 95% bootstrap intervals over source chunks, on strict NDCG@10 unless noted. Every row regenerates with
`quantitative_experiments/revision/robustness.py`; Tables 9, 10 and 12 with `quantitative_experiments/revision/reproduce_tables.py`.

## Code and corrections status

Each branch has an open pull request against `main` for review.

| Repo | Branch | What it contains |
| --- | --- | --- |
| backend | `revision/q1-experiments` (from `origin/main`) | `quantitative_experiments/revision/` (all experiment scripts), restored `evaluation/` and `docs/` (the merge into `main` had dropped them), provenance in every pipeline run, `GENERATION_PROMPT_FILE` and `LLM_TEMPERATURE` settings, prose prompt candidate, corrected test set and hashes, 45 passing tests |
| data_collection_and_fine-tuning | `revision/q1-corrections` | Judge provenance fixed at the source and dataset rebuilt, `CORRECCIONES.md`, `entrenamiento/v2/` (naive baseline builder, CUDA training launcher with Table 6 defaults, ONNX export), v1 material moved to `entrenamiento/legacy_v1/` |
| frontend | `fix/session-in-memory` | Session id and gender flag in memory only; legacy storage cleared on load |

The pooling manifest recorded the original test-set hash (`3c0ba866…`, true Sonnet 4.5 labels), so the backend relabelling
happened after pooling; the manifest now records the corrected file and keeps the old hash with the reason.

## Prerequisites (week 1, blocking)

No new result is produced until these are done, because they decide whether new results can be reproduced.

- [ ] **Recover the training configuration behind checkpoint-256.** The released notebook uses LR 1e-5, temperature
  0.02, group size 11 and no seed. Table 6 says 5e-6, 0.05, 9 and seed 42.
- [ ] **Publish the fine-tuned model.** The Hugging Face repo `remii322/amparai-bge-m3-finetuned` is not publicly
  accessible. The run metadata records ONNX hash `56763c39…`.
- [x] **Restore the backend test set from the data repo.** The backend copy relabelled 48 questions' generator as
  Sonnet 5.5. `SHA256SUMS` updated; the stale `corpus_inventory.csv` hash fixed (its content was already correct).
- [x] **Fix the judge metadata.** The eval files and the manifest said `claude-sonnet-5.5`; all 9,011 verdicts come
  from gpt-6.1-sol. Fixed at the source (`06_build_dataset.py`) and rebuilt; only `generation.judge` and the manifest changed.
- [ ] **Commit the generation prompt behind Table 13** (the prose version), and decide whether the deployed system uses
  it. A candidate is in `prompts/generation_prose_v1.md`; the author who ran `pipeline_final` must confirm it.
- [ ] **Correct facts in the paper.** Sonnet 4.5 wrote 690 of 1,970 training, 48 of 198 test and 43 of 130 dev
  questions. Temperature is 0. Explain near-duplicates in Tables 2–3 and in the strict labels. Note that
  "Law 30364" chunks include articles of other codes the law amended.
- [x] **Archive the stale split docs** in the data repo's `entrenamiento/` folder (800/100 split by ruling).
- [x] **Fix the frontend session.** The session id now lives only in memory (sessionStorage would survive a reload), and the
  legacy `localStorage` keys are deleted on load, so a reload starts a new conversation as §3.6 says.
- [ ] **Commit the registration below** before any Test B result is seen.

### Registration: confirmatory hypotheses (Holm-corrected as one family)

1. **H1 – Another question generator:** fine-tuned beats base on strict NDCG@10 on Test B.
2. **H2 – Value of the protocol:** fine-tuned beats the naive synthetic baseline trained on the same amount of data, on
   strict NDCG@10 on Tests A and B, averaged over seeds.
3. **H3 – Better answers:** Faithfulness with fine-tuned retrieval beats Faithfulness with base retrieval, with the
   same generator, prompt and temperature.

Analysis: 95% bootstrap over source chunks, with seeds averaged where there are several. Every other comparison is
exploratory.

## Work items by track

Seventeen items close all ten MAJOR comments. Track 2 carries nearly all the compute: about 15–20 L4-hours of training,
plus LLM judging. The "Script" column points to the code in `quantitative_experiments/revision/` (backend) or
`entrenamiento/v2/` (data repo).

| # | Item | Track | Closes | Effort | Needs | Script |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Prerequisites (section above) | 1 | Integrity | 1–2 days | — | — |
| 2 | Local evaluation setup: exact search, base and fine-tuned in FP16 | 2 | #36 | ½ day | Model access | `local_retrieval.py` |
| 3 | Test B: ~300 questions for the 165 test chunks from another model family, no lexical filter, judge confirms the source chunk | 2 | #1, #27 | 1 day | Registration | `test_b.py` |
| 4 | Second judge (open-weight, other family) on the 11,763 pooled pairs; agreement (κ); primary result under each judge | 2 | #4 | ½ day | — | `second_judge.py` |
| 5 | Zero-shot baselines: Qwen3-Embedding 0.6B and 4B, mE5-large-instruct, BGE-M3 sparse and ColBERT modes, on Tests A and B | 2 | #2 | 1 day | Item 2 | `local_retrieval.py` |
| 6 | LLM query rewriting (colloquial to legal wording) for base and fine-tuned | 2 | #2, mechanism | ½ day | Item 2 | `query_rewrite.py` |
| 7 | Seeds: 3–5 runs with the documented configuration, on parallel GPUs | 2 | #5 | 1 day elapsed | Training config | `entrenamiento/v2/run_training.sh` |
| 8 | Naive synthetic baseline: chunk-only questions, no judge, random negatives, same data size, 3 seeds | 2 | #2, #3 | 1 day elapsed | Training config | `entrenamiento/v2/build_naive_dataset.py` |
| 9 | Self-hosted reranker (bge-reranker-v2-m3): quality and CPU latency at 20 and 50 candidates | 2 | #7 | ½ day | Item 2 | `rerank_local.py` |
| 10 | MTEB Spanish retrieval tasks: base vs fine-tuned | 2 | #9 | ½ day | Model access | `mteb_spanish.py` |
| 11 | Answer quality with the generator fixed: no retrieval, base, fine-tuned, fine-tuned + reranker, oracle | 3 | #6 | 1½ days | Prompt committed | `answer_quality.py` |
| 12 | Citation accuracy, and coded causes for the 49 answers below 0.5 Faithfulness | 3 | #6, #50 | ½ day | Item 11 | `citations.py`, `answer_quality.py code-errors` |
| 13 | Domain filter: confirm the 24 rejections; ~100 out-of-domain and ~30 crisis questions; error rates | 3 | #10 | ½ day | — | `domain_filter.py` |
| 14 | Expert panel: AC1 agreement, per-category results, family-law panelists separately, comments coded into error types, sampling and system version | 3 | #8 | 1 day | Raw ratings | `expert_panel.py` |
| 15 | Optional Test C: 60–100 questions from institutional FAQ pages, pooled, two judges | 3 | #1 | 3 days max | Items 3–4 | `test_b.py` (verify step) |
| 16 | Rewrite: retitle without "agent", contributions list, related-work synthesis, SUS removed, threats to validity, declarations | 1 | #9, minors | 4–5 days | Results | — |
| 17 | Response letter: every comment mapped to a change or a justified cut | 1 | All | 2 days | Results | — |

## Schedule

The week-1 prerequisites gate every new result, and results freeze at the end of week 2.

| Week | Track 1 – provenance and writing | Track 2 – retrieval | Track 3 – answers and human data |
| --- | --- | --- | --- |
| 1 | Provenance fixes, recover training config, commit registration | Local evaluation setup, second judge, generate Test B | Domain-filter test sets, expert panel reanalysis |
| 2 | Rewrite methods, write up completed analyses | Baselines + rewriting, seeds + naive baseline, reranker latency curve | Answer-quality runs, citations + error coding |
| 3 | Title, contributions, threats to validity, response letter | Final tables + CI plot, MTEB Spanish check | Integrate results, optional Test C (3 days) |

Gates: **end of week 1** – training config recovered and registration committed; **end of week 2** – results frozen;
**end of week 3** – resubmit. Seeds and the naive baseline wait for the first gate; week-3 writing uses only frozen results.

## What we cut and how we justify it

Each cut keeps the claim intact and gets one paragraph in the response letter.

| Cut | Reviewer ask | Justification for the response letter |
| --- | --- | --- |
| 5-fold cross-validation and the full ablation grid | #3 | Report the smallest difference Test A can detect (0.03–0.08 NDCG@10). The naive vs full contrast is the confirmatory test of the protocol; component effects are exploratory |
| Retraining on fully judged negatives | #31 | Report question-clustered rates (2.3% yes, 16% partial). A partial label is not a false negative, and the naive baseline measures what unverified data costs |
| GPL | #2 | The naive pipeline is the InPars / Promptagator recipe; GPL is cited and its differences discussed |
| GradCache, LR and temperature sweeps, fine-tuned reranker | #33, #4 | Optimizations outside the paper's claim; listed as future work |
| Generator temperature ablation | #17 | The code uses temperature 0; the paper's 0.7 is corrected |
| Multi-turn evaluation | #16 | Title drops "agent"; the routing nodes are described as implementation details |
| Transfer to a second corpus (e.g. BSARD) | #10 | Scope stated explicitly; the MTEB Spanish forgetting check is included |
| Corpus changes (SPIJ texts, more institutional documents) | #8, #20 | Corpus frozen at 893 chunks; LP Derecho named with its licensing basis; motivation reworded |
| New expert rounds | #8 | Panel reanalysed from the existing 300 ratings and comments |

## Risks and decision points

The training configuration is the critical path: settle it in the first days of week 1.

| Risk | Fallback | Decide by |
| --- | --- | --- |
| Configuration behind checkpoint-256 cannot be recovered | Train all seeds with the Table 6 setup and make that the main model; numbers move slightly but become reproducible | Week 1 |
| Naive baseline about equal to the full protocol (H2 not supported) | Contribution becomes "synthetic domain adaptation works for Spanish legal lay questions" plus the released benchmark; framing fixed in the registration | Before Test B results |
| Prompt behind Table 13 cannot be reconstructed | Rerun the deployed pipeline end to end (RAGAS and latency) with the committed prompt and report that | Week 1 |
| A larger embedder beats the fine-tuned model | Efficiency framing: a 568M INT8 CPU model within a stated margin of larger models, with an efficiency table | Week 2 |
| Raw expert ratings unavailable | Report the panel descriptively and state why agreement is missing | Week 1 |

## Definition of done

The revision is ready to resubmit when every box below is ticked.

- [ ] Every number in the paper regenerates from the public repos with one documented command
- [ ] Registration committed before Test B results; H1–H3 reported with Holm correction
- [ ] Primary result on Tests A and B with seed variability (mean ± SD and a seed-aware interval)
- [ ] Baseline table with efficiency columns (parameters, CPU latency, memory)
- [ ] Answer-quality table for the five retrieval conditions, plus citation accuracy
- [ ] Domain-filter error rates, with the 24 rejections confirmed
- [ ] Expert panel: agreement, per-category results, error types
- [ ] Paper rewritten: new title, contributions list, threats to validity, SUS removed, declarations
- [ ] Response letter covering all 74 comments
