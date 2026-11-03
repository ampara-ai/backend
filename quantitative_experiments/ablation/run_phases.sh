#!/usr/bin/env bash
# Genera los rankings (profundidad 100) de todas las configuraciones de la ablación sobre el set de TEST y deja listo el
# pooling. No ajusta hiperparámetros: híbrido con los parámetros fijos HYBRID_FUSION (eval_ablation.py) y reranker sobre los 100 candidatos.
#
# Mongo guarda un solo campo `embedding`, así que se reindexa antes de cada variante (base, finetuned); build_runs verifica
# que los vectores guardados coinciden con los que sirve hoy el model service (aborta si no).
#
# Variables:
#   DS              dataset de preguntas (JSONL, formato eval_test_dataset.jsonl)
#   OUT             carpeta de rankings
#   RERANK_BACKEND  openrouter (Voyage rerank-2.5, ≈0,0018 USD por consulta con 100 candidatos; por defecto) | local (ONNX, ~1,5 pares/s)
#   CORPUS          corpus de 893 chunks para reindexar
#
# Uso (desde la raíz del repo):  DS=... OUT=... bash quantitative_experiments/ablation/run_phases.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${DS:?define DS}"; : "${OUT:?define OUT}"
RERANK_BACKEND="${RERANK_BACKEND:-openrouter}"
CORPUS="${CORPUS:-quantitative_experiments/data/inputs/document_chunks_all.jsonl}"
mkdir -p "$OUT"

runs() {  # <reranker-backend> <configs...>  (reintenta si el índice de Atlas aún no sincroniza)
  local backend="$1"; shift
  for try in 1 2 3 4 5 6 7 8; do
    uv run python -m quantitative_experiments.ablation.build_runs --dataset-path "$DS" --out-dir "$OUT" \
        --configs "$@" --reranker-backend "$backend" --depth 100 && return 0
    echo "   reintento $try (índice sin sincronizar)"; sleep 45
  done; return 1
}

echo "=== BM25"; runs local bm25_only || exit 1
echo "=== BASE: reindexando"
uv run python scripts/ingest_chunks.py --with-base-embeddings --jsonl "$CORPUS" 2>&1 | grep -E "completados|Error|✗" || exit 1
runs local dense_base hybrid_base || exit 1
runs "$RERANK_BACKEND" dense_base_rerank100 hybrid_base_rerank100 || exit 1
echo "=== FINETUNED: reindexando"
uv run python scripts/ingest_chunks.py --with-embeddings --jsonl "$CORPUS" 2>&1 | grep -E "completados|Error|✗" || exit 1
runs local dense_finetuned hybrid_finetuned || exit 1
runs "$RERANK_BACKEND" dense_finetuned_rerank100 hybrid_finetuned_rerank100 || exit 1
echo "=== LISTO: rankings en $OUT. Siguiente: pooling con pooling/pool_eval.py sobre estos rankings."
