#!/usr/bin/env bash
# Pasa las preguntas de test por el pipeline COMPLETO (filtro de dominio, decisión de recuperación, recuperación con la configuración
# de la ablación, y opcionalmente generación + source grounding) y guarda rankings de nivel pipeline y de nivel recuperador.
# Complementa a run_phases.sh (recuperador aislado, que alimenta el pooling): las salidas van a OTRA carpeta para no mezclarlas.
#
# Variables:  DS (dataset de test), OUT (carpeta de salida, p. ej. quantitative_experiments/results/pipeline),
#             GENERATE (configuraciones con generación; por defecto solo la arquitectura final, dense_finetuned: las demás llegan hasta la recuperación),
#             RERANK_BACKEND (openrouter por defecto), CORPUS.
# Uso:  DS=... OUT=... bash quantitative_experiments/ablation/run_full_pipeline.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${DS:?define DS}"; : "${OUT:?define OUT}"
GENERATE="${GENERATE:-dense_finetuned}"; RERANK_BACKEND="${RERANK_BACKEND:-openrouter}"
CORPUS="${CORPUS:-quantitative_experiments/data/inputs/document_chunks_all.jsonl}"
mkdir -p "$OUT"

run() {  # <configs...>: las que están en GENERATE llevan generación (se separan en dos llamadas)
  local gen=() nogen=()
  for c in "$@"; do [[ " $GENERATE " == *" $c "* ]] && gen+=("$c") || nogen+=("$c"); done
  for try in 1 2 3 4 5 6 7 8; do
    { [ ${#nogen[@]} = 0 ] || uv run python -m quantitative_experiments.ablation.full_pipeline --dataset-path "$DS" --out-dir "$OUT" \
          --configs "${nogen[@]}" --reranker-backend "$RERANK_BACKEND"; } && \
    { [ ${#gen[@]} = 0 ] || uv run python -m quantitative_experiments.ablation.full_pipeline --dataset-path "$DS" --out-dir "$OUT" \
          --configs "${gen[@]}" --reranker-backend "$RERANK_BACKEND" --with-generation; } && return 0
    echo "   reintento $try (índice sin sincronizar)"; sleep 45
  done; return 1
}

run bm25_only || exit 1
for v in base finetuned; do
  flag=$([ "$v" = base ] && echo --with-base-embeddings || echo --with-embeddings)
  echo "=== $v: reindexando"
  uv run python scripts/ingest_chunks.py "$flag" --jsonl "$CORPUS" 2>&1 | grep -E "completados|Error|✗" || exit 1
  run "dense_$v" "hybrid_$v" "dense_${v}_rerank100" "hybrid_${v}_rerank100" || exit 1
done
echo "=== LISTO: $OUT (<config>.json = nivel pipeline, <config>.retriever.json = nivel recuperador, <config>.trace.jsonl = traza)"
