# Experimentos de la revisión Q1

Código de los experimentos y análisis de [`docs/revision-plan.md`](../../docs/revision-plan.md). Las salidas van a
`quantitative_experiments/results/revision/`. Cada script tiene su uso completo en el docstring (`--help`).

**Plan de ejecución vigente, sin asesores legales:** [experimentos pendientes](../../docs/remaining-experiments.md).
**Entrega al equipo con GPU:** [instrucciones en español, comandos y archivos que deben devolver](../../docs/instrucciones-experimentos-gpu.md).
H1, H2 y H3 tienen preparación sin ejecución por defecto; los wrappers H2/H3 requieren `--run` para iniciar trabajo.
No vuelva a generar Test B para ejecutar estos experimentos: se reutilizan las preguntas congeladas.

H2 conserva el prefijo de consulta del entrenamiento también en desarrollo y Test A/B; H1 mantiene su protocolo
publicado. El reranker ordena por logits sin sigmoide. La evaluación de respuestas conserva los valores negativos
válidos de Answer Relevancy y `code-errors` usa exactamente los puntajes seleccionados en el último `summary.json`;
para cachés antiguas, vuelva a ejecutar `score` con el juez/configuración elegidos antes del análisis de errores.

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
| H1 | `h1.py` | CUDA / MPS / CPU | Comparación base/FT sobre Test B; por defecto solo prepara, `--run` ejecuta |
| H2 | `run_h2.sh` | CUDA; evaluación MPS/CPU posible | Seis entrenamientos, selección en dev y comparación Test B/A; dry-run por defecto |
| H3 | `run_h3.sh` | CUDA / MPS / CPU + API | Cuatro condiciones comparables; insumos/proveedor/prompt congelados; dry-run por defecto |
| H1–H3 | `confirmatory.py` | sin servicios | Reúne exactamente los tres contrastes primarios y aplica Holm; rechaza resultados incompletos o de Test A |
| 4 | `second_judge.py` | API | Segundo juez sobre el pool, kappa y la comparación primaria con cada juez |
| 4 | `second_judge_independent.py` | sin servicios | Sensibilidad a forzar los chunks de origen como relevantes; usa los veredictos ya guardados |
| 6 | `query_rewrite.py` | API | Consultas reescritas en lenguaje jurídico o HyDE, para `local_retrieval --query-field query_rewritten` |
| 7, 8 | `select_checkpoint.py` | Mac o CUDA | Checkpoint de cada corrida elegido en dev |
| 9 | `rerank_local.py` | Mac o CUDA | bge-reranker-v2-m3 sobre 20/50/100 candidatos, calidad y latencia (también en CPU) |
| 10 | `mteb_spanish.py` | Mac o CUDA | Base vs fine-tuned en tareas de recuperación de MTEB en español |
| 11, 12 | `answer_quality.py` | API | Respuestas con el generador fijo (sin recuperación / base / fine-tuned / + reranker / oráculo), RAGAS, citas y causas de error |
| 12 | `citations.py` | sin servicios | Exactitud de las citas normativas (lo usa `answer_quality.py citations`) |
| 13 | `domain_filter.py` | API | Auditoría de los 24 rechazos, conjuntos fuera de dominio y de crisis, sensibilidad, especificidad y estabilidad |
| 14 | `expert_panel.py` | sin servicios | Acuerdo (AC1, AC2, alfa), resultados por categoría y plantilla de codificación de comentarios |
| H1–H3 | `compare.py` | sin servicios | Diferencias pareadas con bootstrap por chunk (y por semilla) y Holm sobre la familia confirmatoria |

## Comandos históricos de la revisión (referencia)

Este bloque documenta los scripts originales; no es una secuencia que deba ejecutarse de corrido. Para nuevas corridas,
use el [runbook vigente](../../docs/remaining-experiments.md), que añade aislamiento de salidas y las condiciones H3
faltantes. Los comandos de entrenamiento, inferencia y API originales pueden ejecutar trabajo inmediatamente.

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

## H1 portable: CUDA, Apple MPS o CPU

`h1.py` reutiliza las 200 preguntas de Test B y el corpus publicado. No genera preguntas, no entrena modelos ni llama a
un LLM. **Por defecto solo valida los insumos y muestra la configuración.** Para descargar pesos y ejecutar inferencia
se requiere `--run` explícito. Las instrucciones se ejecutan desde la raíz de `backend`.

En el equipo de ejecución, use Python 3.12 y un entorno separado:

```bash
python3 -m venv .venv-h1
source .venv-h1/bin/activate
python -m pip install --upgrade pip
```

Para NVIDIA, instale primero la versión CUDA de PyTorch apropiada para el controlador del equipo siguiendo el
[selector oficial de PyTorch](https://pytorch.org/get-started/locally/). En macOS Apple Silicon o CPU, la instalación
siguiente incluye PyTorch. No hace falta instalar el backend web ni sus servicios:

```bash
python -m pip install -r quantitative_experiments/revision/requirements-h1.txt
python -m quantitative_experiments.revision.h1 --dry-run
```

El modo automático elige **CUDA → MPS → CPU**. Usa FP16 en aceleradores y FP32 en CPU, con la misma configuración para
ambos modelos. Si PyTorch aún no está instalado, el modo de preparación funciona con NumPy y python-dotenv; deja el
dispositivo sin resolver. `--dry-run --device cuda` permite preparar un equipo remoto desde un Mac y muestra
`device_available_locally: false`. Con `--run`, un dispositivo solicitado explícitamente que no esté disponible produce
un error antes de descargar modelos o crear salidas. La detección local corresponde al equipo donde se ejecuta.

Ejecute **una** de estas opciones en el equipo elegido cuando quiera iniciar el experimento:

```bash
# Selección automática
python -m quantitative_experiments.revision.h1 --run --device auto

# NVIDIA / CUDA
python -m quantitative_experiments.revision.h1 --run --device cuda \
  --out-dir quantitative_experiments/results/revision/h1_cuda

# Mac / Apple MPS
python -m quantitative_experiments.revision.h1 --run --device mps \
  --out-dir quantitative_experiments/results/revision/h1_mps
```

Los valores por defecto fijan las revisiones públicas `5617a9f…` de `BAAI/bge-m3` y `f2fab1a2…` de
`ampara-ai/bge-m3-ampara`; este último identifica sus pesos como checkpoint-256 en la ficha del modelo. Se verifican sus
SHA-256. Una ejecución descarga aproximadamente 3.5 GB si los pesos no están en la caché de Hugging Face. No necesita un
token para estas dos revisiones públicas.

Se cargan los modelos secuencialmente y se libera la memoria del acelerador entre ellos. `--doc-batch` (4) y
`--query-batch` (32) permiten reducir el consumo de memoria; `--precision fp32` permite elegir esa precisión para ambos
modelos. No se cambia de dispositivo ni de precisión silenciosamente si hay un error. Un checkpoint local completo,
con configuración y tokenizer, se puede pasar con `--finetuned-model /ruta/checkpoint-256`. Para otro repositorio remoto,
es obligatorio indicar también `--finetuned-revision <commit de 40 caracteres>`.

Cada ejecución crea un directorio nuevo; rechaza uno que ya exista, incluso vacío. **No reutiliza automáticamente la corrida
local interrumpida ni la caché de embeddings de otros experimentos.** Las salidas incluyen:

- `protocol.json`: insumos, revisiones y hashes, configuración, código, entorno y fecha antes de la inferencia.
- `base.json` / `finetuned.json` y sus `.meta.json`: rankings completos y procedencia.
- `per_query.csv`: NDCG@10 de ambos modelos, diferencia y clúster por consulta.
- `comparison.json` / `comparison.md`: efecto e IC del 95% con 10,000 remuestreos por fuente, semilla 0.
- `status.json`: avance y estado `complete`, `failed` o `interrupted`. Solo `complete` indica un experimento terminado;
  una terminación forzada del proceso puede dejar el último estado `running`.

H1 usa **strict NDCG@10** como única métrica primaria. La corrección conjunta Holm de H1–H3 queda pendiente. El intervalo
corresponde a un checkpoint fijo y no incluye variabilidad de entrenamiento. Este experimento mide recuperación de
fuentes con preguntas sintéticas, no corrección jurídica. El protocolo local no equivale a un prerregistro externo.

Las pruebas de `tests/test_h1.py` usan datos y codificadores simulados: no descargan modelos ni ejecutan inferencia real.

## Segundo juez sin positivos de origen forzados

El análisis histórico `second_judge.py analyze` conserva todos los positivos estrictos, incluso cuando el segundo juez
los rechaza. El experimento nuevo compara dos políticas sobre **el mismo pool congelado**, las mismas consultas y los
mismos nueve rankings: `independent` usa exclusivamente los veredictos `si`; `source_anchored` además incorpora todos los
chunks de origen. `parcial` y `no` no son relevantes en la política independiente. No se modifica el análisis histórico.

Desde la raíz de `backend`, con Python, NumPy y python-dotenv disponibles:

```bash
python -m quantitative_experiments.revision.second_judge_independent --model google/gemma-4-31b-it
```

No requiere GPU, claves, servicios, llamadas a un LLM ni el repositorio de entrenamiento. Reutiliza los veredictos
existentes. Las salidas se guardan en
`results/revision/second_judge/google__gemma-4-31b-it/independent/` dentro de `quantitative_experiments`:

- `analysis.md` y `analysis.json`: comparación, IC del 95%, cobertura, desacuerdos sobre fuentes y hashes de insumos/código.
- `qrels_independent.jsonl` y `qrels_source_anchored.jsonl`: positivos bajo cada política.
- `per_query.jsonl`: métricas base/FT para cada consulta y política, incluido su clúster de origen.

El programa deriva los pares esperados de `pool_pairs_test.jsonl` y los qrels estrictos, y exige igualdad exacta con los
pares juzgados. No confía en el conteo de `coverage.json`. Colapsa duplicados con etiquetas, modelos y roles consistentes;
rechaza conflictos, etiquetas desconocidas, modelos mezclados, pares faltantes o extra y rankings con consultas faltantes
o documentos duplicados. Los documentos recuperados fuera del pool cuentan como no relevantes y su número en el top 10
se informa por configuración.

Se conserva la totalidad de las consultas; una consulta sin positivos aporta cero, y su ID se publica. El bootstrap
pareado usa 10,000 remuestreos de chunks de origen y semilla 0. `--bootstrap-resamples`, `--seed` y `--output-dir` permiten
cambiar estos parámetros explícitamente. Se reporta Recall@10 ordinario y, por separado, la variante del artículo con
denominador `min(relevantes, 10)`.

En los archivos publicados, hay 11,963 pares únicos, 138 filas duplicadas consistentes y 198 consultas. El segundo juez
acepta 197 de los 200 pares de origen, marca 2 como parciales y rechaza 1. Se obtienen 811 positivos independientes frente
a 814 con origen forzado. FT − base en NDCG@10 independiente es **+0.117189 [0.073348, 0.162625]**.

Este es un **análisis retrospectivo de sensibilidad**, no una hipótesis nueva prerregistrada ni un benchmark independiente.
La cobertura completa se refiere al pool congelado, no a todos los documentos potencialmente relevantes del corpus.
Sin asesores legales, estos resultados sustentan robustez de recuperación bajo etiquetas automáticas; no validan la
corrección jurídica de respuestas ni la seguridad del sistema. Tampoco reemplazan los experimentos pendientes H1–H3.

Pruebas locales, sin servicios:

```bash
python -m pytest tests/test_second_judge_independent.py tests/test_revision.py -q
```

## Procedencia

Las corridas del pipeline (`ablation/full_pipeline.py`, `answer_quality.py`, `domain_filter.py`) guardan
`*.provenance.json` con el commit, el modelo, la temperatura, el proveedor y el SHA-256 de cada prompt. Las corridas de
`local_retrieval.py` guardan `*.meta.json` con el modelo, el dispositivo, la precisión, los parámetros y los tiempos.
