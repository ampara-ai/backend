#!/usr/bin/env bash
# RAGAS (Faithfulness y Answer Relevancy) y latencia del pipeline completo con la arquitectura final, con el juez GPT-5.6 Luna
# ( `openai/gpt-5.6-luna` en OpenRouter) y los embeddings de Answer Relevancy qwen/qwen3-embedding-8b.
#
# Requisito: la traza de la generación: quantitative_experiments/results/pipeline_final/dense_finetuned.trace.jsonl
# (se genera con: DS=... OUT=... GENERATE=dense_finetuned bash quantitative_experiments/ablation/run_full_pipeline.sh, o con
#  full_pipeline.py --with-generation --concurrency 1 --no-filter-cache, para medir la latencia sin interferencias).
#
# El script es reanudable: guarda cada pregunta evaluada en ragas_parcial.jsonl; si se agota el crédito basta con volver a ejecutarlo.
# Costo orientativo: GPT-5.6 Luna cuesta 0,20 USD/M tokens de entrada y 1,20 USD/M de salida; 174 respuestas con dos métricas pueden costar
# entre ~1,7 y ~5 USD. Haz primero el piloto (MODE=pilot) para medir el costo real por respuesta.
#
# Uso (desde la raíz del repo):
#   MODE=pilot bash quantitative_experiments/ablation/run_ragas.sh    # 10 respuestas al azar -> ragas_luna_piloto/
#   MODE=full  bash quantitative_experiments/ablation/run_ragas.sh    # las 174 respondidas  -> ragas_luna/
#   MODE=retry bash quantitative_experiments/ablation/run_ragas.sh    # reintenta las que quedaron sin Faithfulness
#   SAMPLE=60 MODE=full ...                                           # solo una muestra de 60 (si el crédito no alcanza)
set -uo pipefail
cd "$(dirname "$0")/../.."
MODE="${MODE:-pilot}"; JUDGE="${JUDGE:-openai/gpt-5.6-luna}"
TRACE=quantitative_experiments/results/pipeline_final/dense_finetuned.trace.jsonl
case "$MODE" in
  pilot) OUT=quantitative_experiments/results/pipeline_final/ragas_luna_piloto; EXTRA="--sample 10";;
  full)  OUT=quantitative_experiments/results/pipeline_final/ragas_luna; EXTRA="${SAMPLE:+--sample $SAMPLE}";;
  retry) OUT=quantitative_experiments/results/pipeline_final/ragas_luna; EXTRA="--retry-missing";;
  *) echo "MODE debe ser pilot, full o retry"; exit 1;;
esac
set -a; . ./.env; set +a
before=$(curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $OPENROUTER_API_KEY" | python3 -c "import sys,json;d=json.load(sys.stdin)['data'];print(d['total_credits']-d['total_usage'])")
uv run python -m quantitative_experiments.ablation.ragas_final --trace "$TRACE" --out-dir "$OUT" --judge-model "$JUDGE" --concurrency 4 $EXTRA
after=$(curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $OPENROUTER_API_KEY" | python3 -c "import sys,json;d=json.load(sys.stdin)['data'];print(d['total_credits']-d['total_usage'])")
python3 -c "print(f'crédito OpenRouter: antes {float(\"$before\"):.2f} -> después {float(\"$after\"):.2f} (gastado {float(\"$before\")-float(\"$after\"):.3f} USD)')"
