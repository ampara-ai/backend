#!/usr/bin/env bash
# Prepare four matched answer-grounding conditions. Only --run performs inference/API calls.
set -euo pipefail

MODE="${1:---dry-run}"
if [[ "$#" -gt 1 ]]; then echo "Choose only --dry-run or --run" >&2; exit 2; fi
case "$MODE" in --dry-run|--run) ;; *) echo "Usage: bash run_h3.sh [--dry-run|--run]" >&2; exit 2 ;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
H1_RUN_DIR="${H1_RUN_DIR:-$ROOT/quantitative_experiments/results/revision/h1}"
H3_OUT_DIR="${H3_OUT_DIR:-$ROOT/quantitative_experiments/results/revision/h3}"
# Reranker used for the two reranked conditions. Default: Voyage rerank-2.5 through OpenRouter (the paper called it
# voyageai/rerank-2.5; OpenRouter now exposes the same model as mongodb/rerank-2.5). A local alternative is
# RERANKER_SPEC=st:/path/to/pinned-bge-reranker-v2-m3.
RERANKER_SPEC="${RERANKER_SPEC:-openrouter:mongodb/rerank-2.5}"
H3_DEVICE="${H3_DEVICE:-auto}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.6-luna}"
JUDGE_API="${JUDGE_API:-openrouter}"
# Default: the prose prompt (git c805a9b, `generation_prose_v1.md`), not the sectioned production prompt `generation.md`.
# The 198 answers behind the paper's Table 13 are plain prose (no sections, no help-line referrals), which matches this prompt.
export GENERATION_PROMPT_FILE="${GENERATION_PROMPT_FILE:-generation_prose_v1.md}"
export LLM_MODEL="${LLM_MODEL:-openai/gpt-oss-120b}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0}"
export LLM_MAX_OUTPUT_TOKENS="${LLM_MAX_OUTPUT_TOKENS:-8192}"
export LLM_PROVIDER_ONLY="${LLM_PROVIDER_ONLY-groq}"
export LLM_ALLOW_FALLBACKS="${LLM_ALLOW_FALLBACKS:-false}"
export LLM_DATA_COLLECTION="${LLM_DATA_COLLECTION:-deny}"
# These offline generation functions never query MongoDB, but app settings require a URI.
export MONGODB_URI="${MONGODB_URI:-mongodb://unused}"
DATASET="$ROOT/quantitative_experiments/results/revision/test_b/eval_test_b_dataset.jsonl"
QRELS="$ROOT/quantitative_experiments/results/revision/test_b/eval_test_b_qrels.jsonl"

command_step() {
  if [[ "$MODE" == --run ]]; then
    "$@"
  else
    printf '%q ' "$@"
    printf '\n'
  fi
}

if [[ "$MODE" == --run ]]; then
  : "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY locally for generation and evaluation}"
  if [[ "$JUDGE_API" == openai ]]; then
    : "${OPENAI_API_KEY:?Set OPENAI_API_KEY locally for the selected judge API}"
  fi
  "$PYTHON" - "$H1_RUN_DIR" "$H3_OUT_DIR" "$RERANKER_SPEC" "$DATASET" "$QRELS" "$H3_DEVICE" "$JUDGE_MODEL" "$JUDGE_API" <<'PY'
import hashlib, importlib.metadata, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from quantitative_experiments.revision.h1 import detect_hardware, select_device
from quantitative_experiments.provenance import llm_provenance

first, out, dataset, qrels = map(Path, [sys.argv[1], sys.argv[2], sys.argv[4], sys.argv[5]])
reranker_spec = sys.argv[3]
device, judge, api = sys.argv[6:9]
if api not in ('openrouter', 'openai'):
    raise SystemExit('JUDGE_API must be openrouter or openai')
hw = detect_hardware()
if not hw['torch_available']:
    raise SystemExit('PyTorch is required for reranking')
resolved_device = select_device(device, cuda=hw['cuda'], mps=hw['mps'])
if out.exists():
    raise SystemExit('H3_OUT_DIR must not exist; no implicit reuse of previous answers or scores')
reranker = None
if reranker_spec.startswith('st:'):
    reranker = Path(reranker_spec[3:])
    if not reranker.is_dir() or not (reranker / 'config.json').is_file():
        raise SystemExit('A local RERANKER_SPEC must point to a complete pinned local reranker snapshot')
elif not reranker_spec.startswith('openrouter:'):
    raise SystemExit('RERANKER_SPEC must start with openrouter: or st:')
status = json.loads((first / 'status.json').read_text())
if status.get('state') != 'complete':
    raise SystemExit('H1_RUN_DIR must be a completed portable H1 run')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()
for name in ['base.json', 'finetuned.json', 'protocol.json']:
    if status['artifacts'].get(name) != sha(first / name):
        raise SystemExit(f'H1 artifact hash mismatch: {name}')
protocol = json.loads((first / 'protocol.json').read_text())
if protocol['settings']['depth'] < 100:
    raise SystemExit('H3 requires first-stage depth >=100 for the matched reranker comparison')
for key, path in [('dataset', dataset), ('qrels', qrels)]:
    if protocol['inputs'][key]['sha256'] != sha(path):
        raise SystemExit(f'H1 input differs from frozen Test B {key}')
corpus = Path('quantitative_experiments/data/inputs/document_chunks_all.jsonl')
if protocol['inputs']['corpus']['sha256'] != sha(corpus):
    raise SystemExit('Corpus changed after H1')
expected_qids = {json.loads(line)['dataset_id'] for line in dataset.read_text().splitlines() if line.strip()}
for name in ['base', 'finetuned']:
    run = json.loads((first / f'{name}.json').read_text())
    if set(run) != expected_qids or any(len(ids) < 100 or len(ids) != len(set(ids)) for ids in run.values()):
        raise SystemExit(f'{name} must contain >=100 unique candidates for every Test B question')
prompt = Path('prompts') / os.environ['GENERATION_PROMPT_FILE']
if not prompt.is_file():
    raise SystemExit(f'Generation prompt missing: {prompt}')
record = {
    'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'dataset_sha256': sha(dataset),
    'qrels_sha256': sha(qrels), 'corpus_sha256': sha(corpus), 'h1_protocol_sha256': sha(first/'protocol.json'),
    'reranker_spec': reranker_spec,
    'reranker_files_sha256': ({str(p.relative_to(reranker)): sha(p) for p in reranker.rglob('*')
                              if p.is_file() and '.cache' not in p.relative_to(reranker).parts}
                              if reranker is not None else None),
    'conditions': ['base','finetuned','base_rerank','finetuned_rerank'],
    'primary_contrast': 'finetuned minus base automated Faithfulness',
    'context_chunks': 10, 'rerank_depth': 100, 'reranker_max_length': 1024 if reranker is not None else None,
    'device': resolved_device, 'judge_model': judge, 'judge_api': api,
    'generation': {k: os.environ[k] for k in ['LLM_MODEL','LLM_TEMPERATURE','LLM_MAX_OUTPUT_TOKENS',
                                            'LLM_PROVIDER_ONLY','GENERATION_PROMPT_FILE']},
    'generation_prompt_sha256': sha(prompt),
    'generation_llm': llm_provenance(),
    'question_ids': [json.loads(line)['dataset_id'] for line in dataset.read_text().splitlines() if line.strip()],
    'frozen_inputs': {k: {'path':str(p.resolve()), 'sha256':sha(p)} for k,p in
                      {'dataset':dataset,'qrels':qrels,'corpus':corpus,'prompt':prompt}.items()},
    'packages': {p: importlib.metadata.version(p) for p in ['ragas','openai','torch','sentence-transformers']},
    'interpretation': 'Automated source grounding; no legal validation. Reranker contrasts are exploratory.'
}
out.mkdir(parents=True, exist_ok=False)  # Atomic claim after validation; concurrent execution must fail.
(out/'protocol.json').write_text(json.dumps(record, indent=2)+'\n')
(out/'status.json').write_text(json.dumps({'state':'running'})+'\n')
PY
else
  echo "Preparation only. --run performs two reranking runs, 800 answer generations, and automated scoring."
  echo "Set H1_RUN_DIR to a completed H1 run. Reranker: $RERANKER_SPEC (paid API unless RERANKER_SPEC=st:<local snapshot>)."
  printf 'Generator=%s temperature=%s prompt=%s; judge=%s via %s\n' "$LLM_MODEL" "$LLM_TEMPERATURE" "$GENERATION_PROMPT_FILE" "$JUDGE_MODEL" "$JUDGE_API"
fi

for condition in base finetuned; do
  command_step "$PYTHON" -m quantitative_experiments.revision.rerank_local \
    --first-stage "$H1_RUN_DIR/$condition.json" --model "$RERANKER_SPEC" \
    --dataset "$DATASET" --qrels "$QRELS" --depths 100 --name "${condition}_rerank" \
    --out-dir "$H3_OUT_DIR/rerank/$condition" --device "$H3_DEVICE" --max-length 1024
done
for condition in base finetuned base_rerank finetuned_rerank; do
  case "$condition" in
    base|finetuned) ranking="$H1_RUN_DIR/$condition.json" ;;
    base_rerank) ranking="$H3_OUT_DIR/rerank/base/base_rerank_k100.json" ;;
    finetuned_rerank) ranking="$H3_OUT_DIR/rerank/finetuned/finetuned_rerank_k100.json" ;;
  esac
  command_step "$PYTHON" -m quantitative_experiments.revision.answer_quality generate \
    --condition "$condition" --run "$ranking" --dataset "$DATASET" --out-dir "$H3_OUT_DIR/answers" \
    --all --k 10 --concurrency 4
done
command_step "$PYTHON" -m quantitative_experiments.revision.h3_checks "$H3_OUT_DIR"
command_step "$PYTHON" -m quantitative_experiments.revision.answer_quality score \
  --conditions base finetuned base_rerank finetuned_rerank --reference base \
  --dataset "$DATASET" --out-dir "$H3_OUT_DIR/answers" \
  --judge-model "$JUDGE_MODEL" --judge-api "$JUDGE_API" --max-tokens 20000 --concurrency 4
command_step "$PYTHON" -m quantitative_experiments.revision.answer_quality citations \
  --conditions base finetuned base_rerank finetuned_rerank --dataset "$DATASET" --out-dir "$H3_OUT_DIR/answers"
command_step "$PYTHON" -m quantitative_experiments.revision.h3_checks "$H3_OUT_DIR"
if [[ "$MODE" == --run ]]; then
  "$PYTHON" - "$H3_OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); summary=json.loads((p/'answers/summary.json').read_text())
complete=all(metric['scored']==summary['n_questions'] and metric['missing']==0
             for c in ['base','finetuned','base_rerank','finetuned_rerank']
             for metric in summary['coverage'][c].values())
(p/'status.json').write_text(json.dumps({'state':'complete' if complete else 'incomplete_scoring'})+'\n')
PY
fi
