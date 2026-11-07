# Experimentos de la revisión Q1

Código de los experimentos y análisis de [`docs/revision-plan.md`](../../docs/revision-plan.md). Las salidas van a
`quantitative_experiments/results/revision/`. Cada script tiene su uso completo en el docstring (`--help`).

**Dónde corre cada cosa**

- *Sin servicios*: lee archivos versionados; corre en cualquier máquina en segundos.
- *Mac o CUDA*: carga modelos con PyTorch; elige CUDA, luego Apple MPS, luego CPU (`--device` lo fuerza; `--fp32` si MPS da
  problemas con FP16). El entrenamiento es solo CUDA y está en el repo de datos (`entrenamiento/v2/`).
- *API*: llama a OpenRouter con `OPENROUTER_API_KEY` en `.env`. Todos aceptan `--dry-run` (sin clave ni costo: cuenta las
  peticiones y muestra la primera), guardan cada respuesta en una caché y se reanudan donde quedaron.

**Preparación**

```bash
uv sync --group experiments          # agrega onnxruntime y mteb al entorno del backend
export AMPARA_DATA_REPO=../data_collection_and_fine-tuning   # si el repo de datos no está clonado al lado
uv run pytest                        # incluye la reproducción de las Tablas 9, 10 y 12
```

## Scripts por ítem del plan

| Ítem | Script | Corre | Qué produce |
| --- | --- | --- | --- |
| — | `reproduce_tables.py` | sin servicios | Tablas 9, 10 y 12 desde las corridas publicadas; falla si una celda difiere del artículo |
| — | `robustness.py` | sin servicios | Todo "Where we stand": memorización, #27, #30, #32, #40, #42, #47, #6, potencia |
| 2, 5 | `local_retrieval.py` | Mac o CUDA | Ranking exacto con cualquier modelo (`cls:`, `st:`, `onnx:`, `bgem3-sparse:`, `bgem3-colbert:`) |
| 3, 15 | `test_b.py` | API | Test B: preguntas de otra familia de modelos y otro prompt, verificadas por el juez del artículo |
| 4 | `second_judge.py` | API | Segundo juez sobre el pool, kappa y la comparación primaria con cada juez |
| 6 | `query_rewrite.py` | API | Consultas reescritas en lenguaje jurídico o HyDE, para `local_retrieval --query-field query_rewritten` |
| 7, 8 | `select_checkpoint.py` | Mac o CUDA | Checkpoint de cada corrida elegido en dev |
| 9 | `rerank_local.py` | Mac o CUDA | bge-reranker-v2-m3 sobre 20/50/100 candidatos, calidad y latencia (también en CPU) |
| 10 | `mteb_spanish.py` | Mac o CUDA | Base vs fine-tuned en tareas de recuperación de MTEB en español |
| 11, 12 | `answer_quality.py` | API | Respuestas con el generador fijo (sin recuperación / base / fine-tuned / + reranker / oráculo), RAGAS, citas y causas de error |
| 12 | `citations.py` | sin servicios | Exactitud de las citas normativas (lo usa `answer_quality.py citations`) |
| 13 | `domain_filter.py` | API | Auditoría de los 24 rechazos, conjuntos fuera de dominio y de crisis, sensibilidad, especificidad y estabilidad |
| 14 | `expert_panel.py` | sin servicios | Acuerdo (AC1, AC2, alfa), resultados por categoría y plantilla de codificación de comentarios |
| H1–H3 | `compare.py` | sin servicios | Diferencias pareadas con bootstrap por chunk (y por semilla) y Holm sobre la familia confirmatoria |

## Orden sugerido

```bash
# Semana 1 — sin modelos ni API
uv run python -m quantitative_experiments.revision.reproduce_tables
uv run python -m quantitative_experiments.revision.robustness
uv run python -m quantitative_experiments.revision.domain_filter audit-template      # completar in_scope a mano
uv run python -m quantitative_experiments.revision.answer_quality citations --conditions released_final

# Semana 1 — API (primero en seco)
uv run python -m quantitative_experiments.revision.test_b generate --model <otra familia> --dry-run
uv run python -m quantitative_experiments.revision.test_b generate --model <otra familia>
uv run python -m quantitative_experiments.revision.test_b verify
uv run python -m quantitative_experiments.revision.second_judge judge --model <open-weight, otra familia>
uv run python -m quantitative_experiments.revision.second_judge analyze --model <el mismo>

# Semana 2 — Mac o CUDA: FP16 base y fine-tuned, líneas de base, Test B, reranker
uv run python -m quantitative_experiments.revision.local_retrieval --model cls:BAAI/bge-m3 --name dense_base_fp16
uv run python -m quantitative_experiments.revision.local_retrieval --model cls:<checkpoint-256> --name dense_finetuned_fp16
uv run python -m quantitative_experiments.revision.local_retrieval --model st:Qwen/Qwen3-Embedding-0.6B --name qwen3_0.6b --latency-sample 30
uv run python -m quantitative_experiments.revision.local_retrieval --model bgem3-sparse:BAAI/bge-m3 --name bgem3_sparse
uv run python -m quantitative_experiments.revision.local_retrieval --model cls:BAAI/bge-m3 --name dense_base_fp16 \
    --dataset quantitative_experiments/results/revision/test_b/eval_test_b_dataset.jsonl \
    --qrels quantitative_experiments/results/revision/test_b/eval_test_b_qrels.jsonl
uv run python -m quantitative_experiments.revision.rerank_local --first-stage released:dense_finetuned --depths 20 50 100 \
    --name bge_reranker_on_ft --cpu-latency-sample 30
uv run --group experiments python -m quantitative_experiments.revision.mteb_spanish \
    --model base=cls:BAAI/bge-m3 --model finetuned=cls:<checkpoint-256>

# Semana 2 — respuestas con el generador fijo (confirmar antes el prompt: prompts/README.md)
export GENERATION_PROMPT_FILE=generation_prose_v1.md
for c in none base finetuned finetuned_rerank oracle; do
  uv run python -m quantitative_experiments.revision.answer_quality generate --condition $c
done
uv run python -m quantitative_experiments.revision.answer_quality score --conditions none base finetuned finetuned_rerank oracle --reference base
uv run python -m quantitative_experiments.revision.answer_quality citations --conditions base finetuned finetuned_rerank oracle
uv run python -m quantitative_experiments.revision.answer_quality code-errors --condition finetuned --model <codificador>

# Familia confirmatoria (H1 en Test B, H2 con semillas; H3 sale de answer_quality score)
uv run python -m quantitative_experiments.revision.compare --dataset <Test B> --qrels <qrels Test B> \
    --system base=<run base> --system finetuned=<run seed42>,<run seed1>,<run seed2> --reference base --confirmatory finetuned
```

Holm sobre H1–H3 con los tres valores p bootstrap:
`uv run python -c "from quantitative_experiments.revision.compare import holm; print(holm({'H1': p1, 'H2': p2, 'H3': p3}))"`.

## Procedencia

Las corridas del pipeline (`ablation/full_pipeline.py`, `answer_quality.py`, `domain_filter.py`) guardan
`*.provenance.json` con el commit, el modelo, la temperatura, el proveedor y el SHA-256 de cada prompt. Las corridas de
`local_retrieval.py` guardan `*.meta.json` con el modelo, el dispositivo, la precisión, los parámetros y los tiempos.
