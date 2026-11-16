#!/usr/bin/env bash
# H2 preparation/orchestration only. No work is executed without --run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "$HERE/../.." && pwd)"
DATA_REPO="${AMPARA_DATA_REPO:-$BACKEND_ROOT/../data_collection_and_fine-tuning}"
OUTPUT_ROOT="${H2_OUT_DIR:-$BACKEND_ROOT/quantitative_experiments/results/revision/h2}"
BASE_SNAPSHOT="${AMPARA_BASE_SNAPSHOT:-<pinned-local-BAAI-bge-m3-snapshot>}"
MODE=dry-run
MODE_SET=false
STAGE=all
DEVICE=auto

usage() {
  cat <<'EOF'
Usage: bash quantitative_experiments/revision/run_h2.sh [--dry-run|--run]
       [--stage train|evaluate|all] [--device auto|cuda|mps|cpu]

Default: print commands only; no downloads, training, inference, or output writes.
Training always requires CUDA. --device controls checkpoint selection/retrieval.
Environment:
  AMPARA_DATA_REPO       Sibling data repository by default.
  AMPARA_BASE_SNAPSHOT   Pinned local BAAI/bge-m3 snapshot; required to train.
  H2_OUT_DIR            Fresh experiment root (default results/revision/h2).
Training and evaluation can run separately, using the same H2_OUT_DIR.
There is no implicit resume. See docs/h2-runs.md before using --run.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run|--dry-run)
      if [ "$MODE_SET" = true ]; then echo "Choose only one execution mode" >&2; exit 2; fi
      MODE="${1#--}"; MODE_SET=true; shift ;;
    --stage|--device)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 2; }
      if [ "$1" = --stage ]; then STAGE="$2"; else DEVICE="$2"; fi
      shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$STAGE" in train|evaluate|all) ;; *) echo "Invalid stage: $STAGE" >&2; exit 2 ;; esac
case "$DEVICE" in auto|cuda|mps|cpu) ;; *) echo "Invalid device: $DEVICE" >&2; exit 2 ;; esac

# Interpret caller-supplied relative paths before changing working directory.
case "$DATA_REPO" in /*) ;; *) DATA_REPO="$PWD/$DATA_REPO" ;; esac
case "$OUTPUT_ROOT" in /*) ;; *) OUTPUT_ROOT="$PWD/$OUTPUT_ROOT" ;; esac
if [ -n "${AMPARA_BASE_SNAPSHOT:-}" ]; then
  case "$BASE_SNAPSHOT" in /*) ;; *) BASE_SNAPSHOT="$PWD/$BASE_SNAPSHOT" ;; esac
fi
TRAIN_DATA="$OUTPUT_ROOT/training_data"
TRAIN_RUNS="$OUTPUT_ROOT/training"
EVALUATION="$OUTPUT_ROOT/evaluation"
DV2="$DATA_REPO/generacion_dataset/data/output/dataset_v2"
TEST_B="$BACKEND_ROOT/quantitative_experiments/results/revision/test_b"
TEST_A="$BACKEND_ROOT/quantitative_experiments/test_set"
cd "$BACKEND_ROOT"

perform() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$MODE" = run ]; then "$@"; fi
}

perform_eval() {
  # Avoid empty-array expansion under macOS Bash 3.2 with set -u.
  if [ "$DEVICE" = cpu ]; then perform "$@" --fp32; else perform "$@"; fi
}

if [ "$MODE" = run ]; then
  command -v python >/dev/null || { echo "Activate the Python environment first" >&2; exit 1; }
  # Check every destination before the builder or training launcher can write.
  if [ "$STAGE" != evaluate ]; then
    [ ! -e "$OUTPUT_ROOT" ] || { echo "H2_OUT_DIR already exists; choose a fresh experiment root" >&2; exit 1; }
    [ -n "${AMPARA_BASE_SNAPSHOT:-}" ] && [ -d "$BASE_SNAPSHOT" ] || {
      echo "Training requires AMPARA_BASE_SNAPSHOT pointing to the pinned local base model" >&2; exit 1;
    }
    [ -f "$DATA_REPO/entrenamiento/v2/build_naive_dataset.py" ] &&
      [ -f "$DATA_REPO/entrenamiento/v2/run_training.sh" ] || {
        echo "Missing H2 builder/training launcher in AMPARA_DATA_REPO" >&2; exit 1;
      }
    command -v nvidia-smi >/dev/null && command -v torchrun >/dev/null || {
      echo "Training requires the CUDA training environment (nvidia-smi and torchrun)" >&2; exit 1;
    }
    python - "$BASE_SNAPSHOT" <<'PY'
import hashlib
import importlib.util
import sys
from pathlib import Path
import torch

if not torch.cuda.is_available():
    raise SystemExit("Training requires an available CUDA device")
for package in ("FlagEmbedding", "deepspeed"):
    if importlib.util.find_spec(package) is None:
        raise SystemExit(f"Missing training dependency: {package}")
folder = Path(sys.argv[1])
weights = folder / "pytorch_model.bin"
if not weights.is_file() or not (folder / "config.json").is_file():
    raise SystemExit("Use the complete pinned BAAI/bge-m3 snapshot, including pytorch_model.bin and config.json")
digest = hashlib.sha256()
with weights.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38":
    raise SystemExit("Base weights do not match pinned BAAI/bge-m3 revision 5617a9f61b028005a4858fdac845db406aefb181")
if list(folder.glob("model*.safetensors")):
    raise SystemExit("Use the unmodified pinned base snapshot; alternate safetensors could override the verified bin weights")
PY
  fi
  if [ "$STAGE" != train ]; then
    [ ! -e "$EVALUATION" ] || { echo "Evaluation output already exists; no implicit resume" >&2; exit 1; }
    for input in "$DV2/eval_dev_dataset.jsonl" "$DV2/eval_dev_qrels.jsonl" \
      "$TEST_B/eval_test_b_dataset.jsonl" "$TEST_B/eval_test_b_qrels.jsonl" \
      "$TEST_A/eval_test_dataset.jsonl" "$TEST_A/eval_test_qrels.jsonl"; do
      [ -f "$input" ] || { echo "Missing evaluation input: $input" >&2; exit 1; }
    done
    # Resolve once, before training/output writes; all evaluations share it.
    DEVICE="$(python - "$DEVICE" <<'PY'
import sys
import torch

requested = sys.argv[1]
cuda, mps = torch.cuda.is_available(), torch.backends.mps.is_available()
if requested == "auto":
    requested = "cuda" if cuda else "mps" if mps else "cpu"
if (requested == "cuda" and not cuda) or (requested == "mps" and not mps):
    raise SystemExit(f"Requested evaluation device is unavailable: {requested}")
print(requested)
PY
)"
  fi
fi

printf 'H2 %s; stage=%s; evaluation device=%s\n' "$MODE" "$STAGE" "$DEVICE"
printf '1376 training rows per arm; seeds 42,1,2; dev selection; Test B primary, Test A supporting.\n'

if [ "$STAGE" != evaluate ]; then
  if [ "$MODE" = run ]; then
    mkdir -p "$(dirname "$OUTPUT_ROOT")"
    mkdir "$OUTPUT_ROOT"  # Atomic claim; a concurrent launcher must fail.
  fi
  perform python "$DATA_REPO/entrenamiento/v2/build_naive_dataset.py" --n-neg 10 --out-dir "$TRAIN_DATA"
  if [ "$MODE" = run ]; then
    python - "$TRAIN_DATA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "build_report.json").read_text())
if report["rows"] != 1376 or report["seed"] != 520 or report["n_neg"] != 10:
    raise SystemExit("Training dataset differs from the frozen H2 design")
expected = {
    "training_naive.json": "0f8afe1e0fba94a81f3cd870a8d9dfda711c5162f773101985b18f387b68ebdb",
    "training_full_matched.json": "233e553ebb1352a4e11d2b5e4e558067d7066a0c984888a39d4eb9c54ac24ff4",
}
for name, digest in expected.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"Frozen H2 training data hash differs: {name}")
PY
  fi
  for arm in full_matched naive; do
    for seed in 42 1 2; do
      perform env BASE_MODEL="$BASE_SNAPSHOT" SEED="$seed" \
        TRAIN_FILE="$TRAIN_DATA/training_${arm}.json" OUTPUT_DIR="$TRAIN_RUNS/${arm}_seed${seed}" \
        LR=5e-6 TEMPERATURE=0.05 GROUP_SIZE=9 EPOCHS=5 BATCH=1 GRAD_ACCUM=16 \
        MAX_LEN=2048 WARMUP_RATIO=0.1 SAVE_STEPS=85 N_GPUS=1 \
        bash "$DATA_REPO/entrenamiento/v2/run_training.sh"
    done
  done
fi

if [ "$STAGE" != train ]; then
  if [ "$MODE" = run ]; then
    # The upstream launcher publishes provenance only after successful training.
    python - "$TRAIN_DATA" "$TRAIN_RUNS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data, runs = map(Path, sys.argv[1:])
bases, environments = set(), set()
expected_data = {
    "naive": "0f8afe1e0fba94a81f3cd870a8d9dfda711c5162f773101985b18f387b68ebdb",
    "full_matched": "233e553ebb1352a4e11d2b5e4e558067d7066a0c984888a39d4eb9c54ac24ff4",
}
for arm in ("full_matched", "naive"):
    digest = hashlib.sha256((data / f"training_{arm}.json").read_bytes()).hexdigest()
    if digest != expected_data[arm]:
        raise SystemExit(f"Frozen H2 training data hash differs: {arm}")
    for seed in (42, 1, 2):
        folder = runs / f"{arm}_seed{seed}"
        p = json.loads((folder / "provenance.json").read_text())
        required = {"records": 1376, "seed": seed, "train_file_sha256": digest,
                    "learning_rate": 5e-6, "temperature": 0.05, "train_group_size": 9,
                    "epochs": 5, "per_device_batch": 1, "grad_accum": 16, "n_gpus": 1,
                    "max_len": 2048, "total_steps": 430, "warmup_steps": 43, "save_steps": 85}
        if any(p.get(key) != value for key, value in required.items()):
            raise SystemExit(f"Training provenance mismatch: {folder}")
        if not (folder / "train.log").is_file() or not (folder / "config.json").is_file():
            raise SystemExit(f"Training did not finish completely: {folder}")
        bases.add(p["base_model"])
        environments.add(hashlib.sha256((folder / "pip_freeze.txt").read_bytes()).hexdigest())
if len(bases) != 1 or len(environments) != 1:
    raise SystemExit("All six runs must use the same base snapshot and training environment")
PY
    mkdir "$EVALUATION"  # Atomic claim; do not share an in-progress evaluation.
  fi

  for arm in full_matched naive; do
    for seed in 42 1 2; do
      run_dir="$TRAIN_RUNS/${arm}_seed${seed}"
      perform_eval python -m quantitative_experiments.revision.select_checkpoint \
        --run-dir "$run_dir" --dataset "$DV2/eval_dev_dataset.jsonl" --qrels "$DV2/eval_dev_qrels.jsonl" \
        --device "$DEVICE"
      selected="<path from $run_dir/selected_checkpoint.txt>"
      if [ "$MODE" = run ]; then IFS= read -r selected < "$run_dir/selected_checkpoint.txt"; fi
      perform_eval python -m quantitative_experiments.revision.local_retrieval \
        --model "cls:$selected" --name "${arm}_seed${seed}" \
        --dataset "$TEST_B/eval_test_b_dataset.jsonl" --qrels "$TEST_B/eval_test_b_qrels.jsonl" \
        --out-dir "$EVALUATION/runs/test_b" --device "$DEVICE" --max-length 2048 --depth 100
      perform_eval python -m quantitative_experiments.revision.local_retrieval \
        --model "cls:$selected" --name "${arm}_seed${seed}" \
        --dataset "$TEST_A/eval_test_dataset.jsonl" --qrels "$TEST_A/eval_test_qrels.jsonl" \
        --out-dir "$EVALUATION/runs/test_a" --device "$DEVICE" --max-length 2048 --depth 100
    done
  done
  for test in b a; do
    if [ "$test" = b ]; then
      dataset="$TEST_B/eval_test_b_dataset.jsonl"; qrels="$TEST_B/eval_test_b_qrels.jsonl"; role=primary
    else
      dataset="$TEST_A/eval_test_dataset.jsonl"; qrels="$TEST_A/eval_test_qrels.jsonl"; role=supporting
    fi
    ranks="$EVALUATION/runs/test_$test"
    perform python -m quantitative_experiments.revision.compare \
      --dataset "$dataset" --qrels "$qrels" \
      --system "naive=$ranks/naive_seed42.json,$ranks/naive_seed1.json,$ranks/naive_seed2.json" \
      --system "full_matched=$ranks/full_matched_seed42.json,$ranks/full_matched_seed1.json,$ranks/full_matched_seed2.json" \
      --reference naive --metric ndcg@10 --resamples 10000 --out "$EVALUATION/test_${test}_${role}.md"
  done
fi

printf 'H2 %s finished. Final H1-H3 Holm adjustment is a separate step.\n' "$MODE"
