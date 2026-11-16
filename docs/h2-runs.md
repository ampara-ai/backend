# H2: matched protocol comparison

H2 asks whether the complete synthetic-data protocol improves strict NDCG@10 over a naive synthetic-data baseline at the same training-row and optimizer-update budget. **Test B is the primary endpoint; Test A is supporting.** Freeze this endpoint and the configuration before inspecting either comparison.

The preparation script is [`run_h2.sh`](../quantitative_experiments/revision/run_h2.sh). It defaults to a dry run, which only prints the commands. It does not require CUDA, a downloaded model, generated training files, or prior experiment outputs, and it creates no outputs. Actual training or inference requires `--run`.

## Design and prerequisites

| Condition | Training rows | Distinct questions | Training seeds |
| --- | ---: | ---: | --- |
| Full protocol, size matched | 1,376 | 1,286 | 42, 1, 2 |
| Naive chunk questions | 1,376 | 1,376 | 42, 1, 2 |

The full condition is a deterministic sample of the 2,057 expanded training rows; the builder uses seed 520. The naive condition uses pre-judge chunk questions, their source positive, and random training-partition negatives. Its candidate negative pool has ten entries per row; the full condition has four to ten. Training samples a group of one positive and eight negatives, following the existing FlagEmbedding pipeline. This is a whole-protocol comparison: it changes question composition, filtering, negative construction, and balancing together. It does not isolate individual components, match the number of distinct questions, or reproduce InPars/Promptagator exactly.

Before actual execution:

1. Clone the data repository beside backend, or set `AMPARA_DATA_REPO` to its absolute path. The launcher requires `entrenamiento/v2/build_naive_dataset.py` and `run_training.sh`, the frozen input data, and the 130-question development set.
2. Review the updated `entrenamiento/v2/notebooks/finetuning_bge_m3_ampara_dataset_v2_en.ipynb`: its training output records learning rate `5e-6`, temperature `0.05`, group size `9`, five epochs, seed `42`, and maximum length `2048`. The earlier claim that only a mismatched notebook existed is stale. Preserve the remaining limitations of historical checkpoint provenance; these new runs do not establish byte identity with the original model.
3. Prepare a validated CUDA environment with Python available as `python`, PyTorch, FlagEmbedding, DeepSpeed, and `torchrun`. Install and freeze compatible training versions before launching any condition. The H1 inference requirements are not a training environment. Reuse exactly the same environment for all six jobs; the upstream launcher saves `pip_freeze.txt`, `gpu.txt`, `train.log`, and `provenance.json` in each completed training directory.
4. Obtain the **complete, unmodified** local `BAAI/bge-m3` snapshot at revision `5617a9f61b028005a4858fdac845db406aefb181`, including its tokenizer/configuration and sparse/ColBERT head files. Set `AMPARA_BASE_SNAPSHOT` to that directory. The wrapper verifies the public `pytorch_model.bin` SHA-256 `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38` before training. A dense-only H1 download is not a complete training snapshot. No model-download command is run by this wrapper.
5. Choose a new `H2_OUT_DIR`. Archive the base snapshot manifest, both repository commits/diffs, exact command/environment, and hashes of development/Test A/Test B inputs and the corpus before starting. Preserve generated `training_data/build_report.json` and all training/selection/ranking metadata. The existing training scripts record the base path, not its immutable Hub revision, so retain the snapshot revision/hash declaration with the run.

All six jobs use one GPU, FP16, per-device batch one, gradient accumulation 16, five epochs, and maximum sequence length 2048. For 1,376 rows this gives 86 updates per epoch, 430 total updates, 43 warmup updates, and checkpoints every 85 updates. These settings are fixed in the wrapper; changing them requires a separately documented protocol amendment, not selecting a configuration after seeing test scores.

## Preview and execute

Run these commands from backend. The first command is safe on a machine without the model or GPU:

```bash
bash quantitative_experiments/revision/run_h2.sh --dry-run
```

For execution, activate the prepared Python environment, then set paths appropriate to that machine:

```bash
export AMPARA_DATA_REPO=/absolute/path/to/data_collection_and_fine-tuning
export AMPARA_BASE_SNAPSHOT=/absolute/path/to/pinned/bge-m3/snapshot
export H2_OUT_DIR=/absolute/path/to/new/h2-experiment

# Six training jobs, then dev selection and A/B evaluation on CUDA:
bash quantitative_experiments/revision/run_h2.sh --run --stage all --device cuda
```

To train on NVIDIA hardware and evaluate later on another machine:

```bash
# CUDA machine; the experiment root must not already exist.
bash quantitative_experiments/revision/run_h2.sh --run --stage train

# Copy the entire experiment root, including training_data and all six
# training directories. Set H2_OUT_DIR and AMPARA_DATA_REPO on the target.
# An inference environment with the backend's embedding dependencies is needed.
bash quantitative_experiments/revision/run_h2.sh --dry-run --stage evaluate --device mps
bash quantitative_experiments/revision/run_h2.sh --run --stage evaluate --device mps
```

`--device` affects checkpoint selection and retrieval only. `auto` selects CUDA, then MPS, then CPU through the existing encoder; use an explicit device to keep the evaluation environment fixed. CUDA/MPS evaluation uses FP16; `--device cpu` uses FP32 throughout. Training always requires CUDA. Both conditions must use the same evaluation settings. The wrapper itself performs no API judging or additional question generation.

Training requires an absent experiment root; evaluation requires an absent `evaluation/` directory. There is no implicit resume. A failed/interrupted execution leaves its evidence on disk and the wrapper refuses to overwrite it. Diagnose the failure and use a separately documented fresh run. Do not delete evidence to make a command resume. The underlying scripts remain independently callable for a deliberately audited recovery.

## Selection, results, and interpretation

For each of the six jobs, `select_checkpoint.py` evaluates checkpoints **only on development data**. It chooses the greatest strict NDCG@10, then greatest Recall@10; an exact tie retains the earliest candidate in its sorted order. A final model saved at the run root is also a candidate. Never force checkpoint-256 for these new training runs or select checkpoints using Test A or B.

The wrapper reads each run's `selected_checkpoint.txt`, retrieves over the unchanged 893-chunk corpus at depth 100, and compares full-matched against naive using only strict NDCG@10. Expected outputs are:

```text
H2_OUT_DIR/
  training_data/{training_full_matched.json,training_naive.json,build_report.json}
  training/{full_matched,naive}_seed{42,1,2}/
  evaluation/runs/test_b/{full_matched,naive}_seed{42,1,2}.json
  evaluation/runs/test_a/{full_matched,naive}_seed{42,1,2}.json
  evaluation/test_b_primary.{md,json}
  evaluation/test_a_supporting.{md,json}
```

Comparison uses 10,000 bootstrap replicates and the existing default seed zero. Source clusters are paired between conditions; training seeds are resampled independently between conditions by `compare.py`. Identical seed numbers do not turn this into a paired-seed bootstrap. Report individual training-seed results as well as the mean, SD, effect, and interval; three seeds provide limited information about training variability.

Use the Test B NDCG@10 raw bootstrap result as H2's sole contribution to the later H1–H3 Holm family adjustment. The Test A result is supporting. The wrapper does not claim that a one-test adjustment completes the family. H2 supports conclusions about automated source/alias retrieval, not legal correctness or safety, and a negative H2 result must be reported.

The wrapper reuses the existing selector/retrieval/comparison implementations. They do not have all of the portable H1 launcher's provenance and numeric-validation checks: in particular, the shared embedding cache uses a weaker cache key, and comparison checks missing queries rather than validating every ranking field. Evaluate against the unchanged corpus/checkpoints in a clean cache environment, preserve metadata, and audit ranking coverage/uniqueness and effective precision before interpreting results. Avoid silently mixing historical ONNX rankings with these FP16/FP32 runs.

H2 is six new training jobs. Three additional full-data training seeds can assess the main model's training variability, but they are a separate experiment. The original-checkpoint H1 comparison does not require retraining, and the old plan's eleven training jobs are not a prerequisite for this H2 comparison.
