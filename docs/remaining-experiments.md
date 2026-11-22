# Remaining experiments without legal assessors

This runbook supersedes the scheduling and expert-panel assumptions in the November 6 revision plan. It prepares
execution; it does not establish that the experiments have been completed. H1's earlier local attempt was interrupted
after base retrieval, with no fine-tuned comparison. Do not pool or silently reuse those partial outputs.

The operational handoff, with staged commands, environments, logs and incomplete-scoring recovery, is available in
[Spanish GPU execution instructions](instrucciones-experimentos-gpu.md). Real CUDA training and provider execution
still require validation on the execution host.

## What is necessary

| Priority | Experiment | Work | Claim it can support |
| --- | --- | --- | --- |
| Core | H1: fixed published checkpoint vs base on Test B | Two retrieval runs; strict NDCG@10, paired source-cluster CI | Robustness to a changed synthetic question-generation protocol |
| Core if claiming protocol value | H2: matched full protocol vs naive | Six CUDA training runs, dev selection for each, evaluation on B and A | Whole-protocol improvement at equal training-row/update budget |
| Core if claiming better answers | H3: controlled grounding | Four retrieval contexts, 200 questions each, fixed generator/prompt/settings, automatic scoring | Automated source grounding, not legal correctness |
| Core analysis | Joint H1–H3 correction | Collect exactly three primary raw p-values, apply Holm once | Multiplicity-controlled primary comparisons |
| Strongly recommended | Same reranker (Voyage rerank-2.5) on base and FT | Two reranking runs, same model and 100 candidates; included in H3 wrapper | Whether reranking changes the practical retrieval tradeoff |
| Strongly recommended | One competitive zero-shot embedding baseline | Evaluate on A and B, choose model before seeing results | Whether tuning beats a credible alternative |
| Needed for a training-stability claim | Repeated full-data training | Three full-data seeds, additional to the six H2 runs | Stability of the principal training recipe |
| Conditional on retaining safety claims | Independent end-to-end crisis evaluation | New held-out cases, real graph/API, both crisis settings | Routing behavior on that held-out challenge set |
| Optional | More embedders, query rewriting, MTEB, five seeds, many rerank depths | After the core comparison | Broader exploratory evidence |

No new expert-panel task is assumed. Remove legal-validity and victim-outcome claims that require unavailable assessors.
Automatic judges, matching cited article numbers, and source recovery do not replace legal assessment.

The three primary contrasts are H1 **FT − base strict NDCG@10 on B**, H2 **matched-full − naive strict NDCG@10 on B**,
and H3 **FT − base Faithfulness on B**. Test A and the reranker contrasts are supporting/exploratory, not extra primary
hypotheses. Retain negative or null results. Save a dated specification before the next run; a local file is not an
external preregistration. The existing source-anchoring sensitivity analysis is retrospective.

## 1. H1: CUDA, MPS or CPU

Follow the isolated environment instructions in [the experiment README](../quantitative_experiments/revision/README.md).
From the backend root:

```bash
python -m quantitative_experiments.revision.h1 --dry-run --device cuda
# Later, on the execution host; auto chooses CUDA, then MPS, then CPU:
python -m quantitative_experiments.revision.h1 --run --device auto
```

Only `--run` loads models. The public checkpoint is pinned and hash-checked; its identity as checkpoint-256 is based on
the publisher's model card. Do not claim a byte-for-byte match to the historical ONNX export. The CI covers query/source
variation for one checkpoint, not training-seed variability. Keep FP16/FP32 results labeled separately.

## 2. H2: six matched training runs

See [the H2 runbook](h2-runs.md) for the sibling data repository, pinned base snapshot, dependencies and exact settings.

```bash
bash quantitative_experiments/revision/run_h2.sh --dry-run
# Training requires NVIDIA CUDA; evaluation can be separated and use MPS.
bash quantitative_experiments/revision/run_h2.sh --run --stage train
bash quantitative_experiments/revision/run_h2.sh --run --stage evaluate --device auto
```

Both arms contain **1,376 rows**, not equal unique-question counts: the released builder yields 1,286 distinct questions
for matched-full and 1,376 for naive. The comparison changes several protocol components together. It is not an exact
InPars/Promptagator reproduction or a component ablation. Choose each run's checkpoint using development data only.
Use the same explicit query prefix for H2 training, development selection and Test A/B retrieval; the wrapper validates
the frozen training-row prompt and forwards it to both evaluation commands. This does not change the training-data hashes.

Three additional full-data seeds are useful if the paper retains a seed-stability claim. They are separate from H2:
with those included, the minimal training package is nine runs. Freeze a single tested CUDA training environment and
archive its versions, logs, model/data hashes and checkpoint-selection records.

## 3. H3: four matched answer-grounding conditions

The wrapper consumes **a complete portable H1 output directory**. It verifies H1 artifact hashes, reranks the top 100
candidates of both base and FT with Voyage rerank-2.5 through the OpenRouter rerank API (the paper's reranker; the paper
cited it as `voyageai/rerank-2.5` and OpenRouter now exposes the same model as `mongodb/rerank-2.5`, the default
`RERANKER_SPEC=openrouter:mongodb/rerank-2.5`), then generates 200 answers for each of `base`, `finetuned`,
`base_rerank`, `finetuned_rerank`. A local alternative is `RERANKER_SPEC=st:/path/to/pinned-bge-reranker-v2-m3`.

```bash
bash quantitative_experiments/revision/run_h3.sh --dry-run
# Configure on the execution host before using --run:
export H1_RUN_DIR=/path/to/completed/h1
# Optional; default is the paper's Voyage rerank-2.5 via OpenRouter:
# export RERANKER_SPEC=openrouter:mongodb/rerank-2.5
export H3_DEVICE=auto
export H3_OUT_DIR=/path/to/new/h3-output
# Configure API keys locally; never commit them.
bash quantitative_experiments/revision/run_h3.sh --run
# If a run is interrupted, continue it (same config and commit; refuses if the frozen protocol changed):
# bash quantitative_experiments/revision/run_h3.sh --run --resume
```

Actual execution makes paid generation/scoring calls. The environment needs the backend's generation dependencies and
RAGAS 0.4.3; the H1-only requirements file is insufficient. `OPENROUTER_API_KEY` is required; `OPENAI_API_KEY` is also
required when `JUDGE_API=openai`. Freeze `LLM_MODEL`, `LLM_PROVIDER_ONLY`, `LLM_TEMPERATURE`, `LLM_MAX_OUTPUT_TOKENS`,
`GENERATION_PROMPT_FILE`, `JUDGE_MODEL`, and `JUDGE_API` before execution. The wrapper records these nonsecret settings.

The default prompt is `generation_prose_v1.md` (the prose prompt from git commit `c805a9b`), not the sectioned
production prompt `generation.md` (commit `589eb6d`). Evidence: none of the 198 answers behind Table 13 contains the
sections or help-line referrals that `generation.md` requires (0/198 mention Línea 100, 105 or CEM; median length 403
characters), so they were not produced with it. This is consistent with the prose prompt but does not prove it is
byte-identical to the one used; the file and its SHA-256 are recorded in every run. Set `GENERATION_PROMPT_FILE=generation.md`
to evaluate the current production prompt instead. All four main conditions receive the same prompt
and top-10 context policy. The optional `none` condition uses a different prompt and must remain exploratory;
`source_context` supplies the source and aliases, not a proven oracle/upper bound.

Use the isolated answer directory for all four conditions. Scoring cache entries are tied to the answers, contexts,
questions and judge settings; incomplete scores must be retried. Report metric coverage and paired sample size. Do not
declare H3 supported from a smaller silently selected subset. Before claiming better answers, inspect/report refusals,
empty answers, generation errors and unsupported citations; high faithfulness alone can reward answers that say little.
This runner evaluates controlled generation, not the complete deployed graph or its crisis safety behavior.

## 4. Final primary family

After all three primary comparisons exist, run the offline collector. It rejects missing primary contrasts or incomplete
base/FT Faithfulness coverage:

```bash
python -m quantitative_experiments.revision.confirmatory \
  --h1 quantitative_experiments/results/revision/h1/comparison.json \
  --h2 quantitative_experiments/results/revision/h2/evaluation/test_b_primary.json \
  --h3 /path/to/h3-output/answers/summary.json \
  --out quantitative_experiments/results/revision/confirmatory.json
```

Do not interpret a one-comparison Holm correction as correction of H1–H3. The collector records input hashes and keeps
the raw and adjusted values. A bootstrap-tail estimate reported as zero reflects finite Monte Carlo resolution.

## 5. Supporting and conditional work

Choose one zero-shot embedding competitor before running it. Use `local_retrieval.py` on both fixed datasets, with
model-recommended query instructions; report its compute/memory budget. That existing CLI starts inference immediately,
so it is not included in a default preparation command. Additional embedders and rewriting are optional after the core.

For crisis evaluation, the existing `domain_filter.py` executes individual nodes and combines their majority labels;
it is not an end-to-end validation. A separate runner and new frozen challenge set remain to be implemented. Include
overt current danger, historical/hypothetical/quoted danger, ordinary in-domain/OOD controls and ambiguous cases; use
isolated sessions, record final responses/routes/errors/latency, and compare crisis enabled versus disabled. The original
30 development cases are regression cases, not independent confirmation. Avoid claims of legal correctness, clinical
safety, or zero added latency. If this study is deferred, narrow the manuscript's safety claims accordingly.

## Status and publication threshold

This PR prepares the core runs and retrospective relevance-label check; it does not make the paper submission-ready.
Readiness depends on the observed effects, model/training provenance, transparent automated-evaluation limitations,
licensing and author disclosures, and a manuscript whose claims match the evidence. Journal selection remains Q1,
non-MDPI; that constraint does not remove the need for these experiments.
