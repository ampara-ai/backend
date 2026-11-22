# Instrucciones para ejecutar H1–H3 en el equipo con GPU

Esta guía prepara la entrega de las corridas al equipo. **Las pruebas del código son locales y simuladas; todavía falta validar la ejecución real en CUDA y las llamadas del proveedor.** No ejecutar todos los bloques de este documento como un único script.

| Etapa | Equipo | Trabajo |
| --- | --- | --- |
| H1 | CUDA, MPS o CPU | Base y checkpoint publicado sobre las mismas 200 preguntas de Test B |
| H2 | Entrenamiento CUDA; evaluación CUDA/MPS/CPU | 6 entrenamientos: `full_matched` y `naive`, semillas 42, 1, 2; selección en dev y evaluación B/A |
| H3 | CUDA/MPS/CPU y API | 2 rerankings y 4 condiciones × 200 respuestas; generación y evaluación con costo |
| Holm | Cualquier equipo | Análisis local de los tres contrastes primarios completos |

No regenerar Test B, modificar el corpus ni elegir configuraciones mirando sus resultados. Sin asesores legales, H3 mide fundamentación automática en las fuentes; no demuestra corrección jurídica ni seguridad para personas en riesgo. El [plan vigente](remaining-experiments.md) distingue las corridas adicionales y condicionales.

## 1. Fijar código, rutas y entornos

Usar los commits revisados de `backend` y `data_collection_and_fine-tuning`; clonar ambos repositorios en carpetas hermanas. Ejecutar los comandos siguientes desde `backend`, en **Bash**, con rutas absolutas adaptadas al equipo:

```bash
set -euo pipefail
cd /ruta/ampara/backend
# Obtener exactamente la rama del PR #3, no ejecutar estos pasos desde main.
git fetch origin feature/portable-revision-experiments-and-remaining-runs
git switch --detach FETCH_HEAD
export AMPARA_DATA_REPO=/ruta/ampara/data_collection_and_fine-tuning
# Commit del repositorio de datos usado al revisar el builder y el launcher.
git -C "$AMPARA_DATA_REPO" fetch origin 23ef4204def3b1f6f352f717c74db5c1c5e8a813
git -C "$AMPARA_DATA_REPO" switch --detach FETCH_HEAD
export EXP_ROOT=/ruta/resultados/ampara_revision_01
mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/evidence"
export H1_RUN_DIR="$EXP_ROOT/h1"
export H2_OUT_DIR="$EXP_ROOT/h2"
export H3_OUT_DIR="$EXP_ROOT/h3"

git rev-parse HEAD > "$EXP_ROOT/evidence/backend_commit.txt"
git -C "$AMPARA_DATA_REPO" rev-parse HEAD > "$EXP_ROOT/evidence/data_commit.txt"
git status --short > "$EXP_ROOT/evidence/backend_status.txt"
git -C "$AMPARA_DATA_REPO" status --short > "$EXP_ROOT/evidence/data_status.txt"
```

`h1`, `h2` y `h3` **no deben existir todavía**, ni siquiera vacías: los programas las crean y rechazan sobrescribirlas. Los logs quedan fuera de esas carpetas. Conservar cualquier corrida fallida; usar otro `EXP_ROOT` para un nuevo intento. No cambiar código ni insumos durante una corrida.

Preparar tres entornos, o entornos equivalentes ya comprobados:

- **H1/evaluación H2:** Python 3.12 y [requirements-h1.txt](../quantitative_experiments/revision/requirements-h1.txt), siguiendo [la instalación de H1](../quantitative_experiments/revision/README.md#h1-portable-cuda-apple-mps-o-cpu). En NVIDIA, instalar primero la distribución de PyTorch compatible con el controlador usando el selector oficial; no copiar una instalación CPU o MPS al servidor.
- **Entrenamiento H2:** entorno CUDA separado con PyTorch, FlagEmbedding, DeepSpeed y `torchrun`. El archivo de H1 no instala este entorno. Antes de las seis corridas, comprobar en una carpeta de prueba independiente que el stack carga el modelo, completa un paso de entrenamiento y guarda/reabre un checkpoint. Esa prueba no es una semilla del experimento. Congelar las versiones que pasen esta comprobación; no actualizar paquetes entre semillas. Este smoke test real y el archivo de versiones CUDA siguen pendientes de ejecución en el equipo con GPU.
- **H3:** dependencias del backend y RAGAS `0.4.3`, además de PyTorch y sentence-transformers para reranking. El entorno mínimo H1 no alcanza. Revisar la compatibilidad de este entorno por separado del entrenamiento; no instalar todas las dependencias sobre el entorno CUDA ya validado.

Al activar cada entorno, guardar `python -m pip freeze` en `evidence/` con su nombre (`h1_pip_freeze.txt`, `h2_train_pip_freeze.txt`, `h2_eval_pip_freeze.txt`, `h3_pip_freeze.txt`). Guardar también la salida de `nvidia-smi` para CUDA y las rutas/comandos usados. No exportar todo el entorno de variables, porque puede contener claves.

## 2. H1: completar primero la comparación base/FT

Activar el entorno H1. `cuda` exige NVIDIA disponible; cambiarlo a `mps` en Apple Silicon o a `cpu` para CPU. Mantener un único dispositivo y precisión para los dos modelos:

```bash
python -m quantitative_experiments.revision.h1 --dry-run --device cuda \
  --out-dir "$H1_RUN_DIR" | tee "$EXP_ROOT/logs/h1_preview.log"

# Este comando sí descarga pesos si faltan y ejecuta inferencia.
python -m quantitative_experiments.revision.h1 --run --device cuda \
  --out-dir "$H1_RUN_DIR" 2>&1 | tee "$EXP_ROOT/logs/h1.log"

python - "$H1_RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
assert json.loads((p / "status.json").read_text())["state"] == "complete"
print((p / "comparison.md").read_text())
PY
```

Los modelos públicos y sus revisiones están fijados por defecto. No se necesita un token para esas revisiones. H1 carga los modelos secuencialmente; CUDA/MPS usan FP16 y CPU usa FP32. Ante falta de memoria, conservar el intento fallido y documentar un nuevo intento con `--doc-batch`/`--query-batch` menores o `--precision fp32`, iguales para ambos modelos. No mezclar resultados FP16 y FP32 ni reutilizar la corrida local interrumpida. H3 requiere H1 `complete`, rankings con al menos 100 candidatos y sus hashes intactos.

## 3. H2: entrenar seis modelos y elegir checkpoints en dev

Activar el entorno de entrenamiento CUDA validado. Preparar el snapshot **completo** de `BAAI/bge-m3` en revisión `5617a9f61b028005a4858fdac845db406aefb181`, incluyendo tokenizer/configuración y las cabezas sparse/ColBERT. La descarga densa de H1 no sustituye ese snapshot. Guardar su revisión y un manifiesto SHA-256 de los archivos junto a la evidencia. El wrapper verifica los pesos principales antes de entrenar.

```bash
export AMPARA_BASE_SNAPSHOT=/ruta/modelos/bge-m3-snapshot-completo
nvidia-smi > "$EXP_ROOT/evidence/h2_gpu.txt"
```

Primero comprobar un paso real en una carpeta independiente. Este bloque construye los mismos datos H2 pero **no crea `H2_OUT_DIR`**. Mantiene los parámetros de entrenamiento; solo limita la ejecución a un paso y guarda inmediatamente. Requiere GPU y sí ejecuta entrenamiento de prueba:

```bash
export PRECHECK_ROOT="$EXP_ROOT/cuda_smoke_01"
test ! -e "$PRECHECK_ROOT"
python "$AMPARA_DATA_REPO/entrenamiento/v2/build_naive_dataset.py" \
  --n-neg 10 --out-dir "$PRECHECK_ROOT/training_data" \
  > "$EXP_ROOT/logs/h2_smoke_data.log"

BASE_MODEL="$AMPARA_BASE_SNAPSHOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node 1 "$AMPARA_DATA_REPO/entrenamiento/v2/train_m3.py" \
  --model_name_or_path "$AMPARA_BASE_SNAPSHOT" \
  --cache_dir "$PRECHECK_ROOT/cache_model" \
  --train_data "$PRECHECK_ROOT/training_data/training_full_matched.json" \
  --cache_path "$PRECHECK_ROOT/cache_data" \
  --train_group_size 9 --query_max_len 2048 --passage_max_len 2048 --pad_to_multiple_of 8 \
  --same_dataset_within_batch True --small_threshold 0 --drop_threshold 0 \
  --output_dir "$PRECHECK_ROOT/model" --learning_rate 5e-6 --fp16 \
  --num_train_epochs 5 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --dataloader_drop_last True --warmup_steps 43 --gradient_checkpointing \
  --deepspeed "$AMPARA_DATA_REPO/entrenamiento/v2/ds_config_zero0.json" \
  --logging_steps 1 --save_steps 1 --max_steps 1 --negatives_cross_device \
  --temperature 0.05 --sentence_pooling_method cls --normalize_embeddings True \
  --unified_finetuning False --use_self_distill False --fix_encoder False \
  --self_distill_start_step 43 --seed 42 \
  2>&1 | tee "$EXP_ROOT/logs/h2_cuda_smoke.log"

python - "$PRECHECK_ROOT/model/checkpoint-1" <<'PY'
import sys
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
p = Path(sys.argv[1])
assert p.is_dir(), f"No se guardó el checkpoint: {p}"
tokenizer = AutoTokenizer.from_pretrained(p, local_files_only=True)
model = AutoModel.from_pretrained(p, local_files_only=True)
print("Checkpoint reabierto en CPU:", p, "parámetros:", sum(x.numel() for x in model.parameters()))
PY
python -m pip freeze > "$EXP_ROOT/evidence/h2_train_pip_freeze.txt"
```

Guardar el log, la configuración del smoke y las versiones. Si falla, corregir el entorno antes de iniciar las seis semillas; no convertir el smoke en una corrida H2 ni cambiar los datos para lograr que pase. Pasar esta prueba comprueba carga, un paso y guardado; no garantiza estabilidad durante las cinco épocas. Con el entorno ya fijado, lanzar las seis corridas:

```bash
bash quantitative_experiments/revision/run_h2.sh --dry-run --stage train \
  | tee "$EXP_ROOT/logs/h2_train_preview.log"
bash quantitative_experiments/revision/run_h2.sh --run --stage train \
  2>&1 | tee "$EXP_ROOT/logs/h2_train.log"
```

Se mantienen los 1,376 registros de cada brazo y sus hashes congelados (regenerados al fijar el campo `prompt` vacío). Ninguna consulta lleva prefijo, ni en entrenamiento ni en selección en dev ni en evaluación A/B, igual que en el sistema desplegado. El wrapper valida ese campo; no editar los datos. Las seis semillas comparten configuración y entorno; no seleccionar checkpoint mirando Test A/B.

Activar después el entorno de evaluación. Se puede copiar toda la carpeta H2 a otro equipo y redefinir allí `H2_OUT_DIR`/`AMPARA_DATA_REPO`; conservar `training_data`, todos los checkpoints y los archivos de procedencia. Para evaluar en Mac, sustituir `cuda` por `mps`; para CPU, por `cpu`:

```bash
python -m pip freeze > "$EXP_ROOT/evidence/h2_eval_pip_freeze.txt"
bash quantitative_experiments/revision/run_h2.sh --dry-run --stage evaluate --device cuda \
  | tee "$EXP_ROOT/logs/h2_eval_preview.log"
bash quantitative_experiments/revision/run_h2.sh --run --stage evaluate --device cuda \
  2>&1 | tee "$EXP_ROOT/logs/h2_eval.log"
```

Verificar seis `checkpoint_selection.json`/`selected_checkpoint.txt`, doce rankings y las comparaciones `evaluation/test_b_primary.{json,md}` y `evaluation/test_a_supporting.{json,md}`. Test B aporta H2 a Holm; Test A es apoyo. Las tres semillas adicionales con todos los datos son otro experimento, no parte de estas seis. Detalles y limitaciones de caché: [runbook H2](h2-runs.md).

## 4. H3: cuatro condiciones, mismo generador y juez

Activar el entorno H3. El reranker por defecto es Voyage rerank-2.5 por la API de OpenRouter (el del artículo; antes `voyageai/rerank-2.5`, hoy `mongodb/rerank-2.5`, el mismo modelo con otro identificador); no necesita snapshot local. Para usar `BAAI/bge-reranker-v2-m3` local, fijar `RERANKER_SPEC=st:<snapshot completo y fijado>`. Configurar las claves **solo en el entorno local**: `OPENROUTER_API_KEY`; agregar `OPENAI_API_KEY` si el juez usa la API de OpenAI. No pegarlas en comandos compartidos ni logs. La generación usa OpenRouter aunque el juez use otra API.

```bash
export PYTHON=python
# opcional; por defecto openrouter:mongodb/rerank-2.5 (Voyage)
# export RERANKER_SPEC=openrouter:mongodb/rerank-2.5
export H3_DEVICE=cuda
export GENERATION_PROMPT_FILE=generation_prose_v1.md   # prompt en prosa (commit c805a9b); generation.md es el de producción actual
export LLM_MODEL=openai/gpt-oss-120b
export LLM_PROVIDER_ONLY=groq
export LLM_TEMPERATURE=0
export LLM_MAX_OUTPUT_TOKENS=8192
export LLM_ALLOW_FALLBACKS=false
export LLM_DATA_COLLECTION=deny
export JUDGE_API=openrouter
export JUDGE_MODEL=openai/gpt-5.6-luna
python -m pip freeze > "$EXP_ROOT/evidence/h3_pip_freeze.txt"

bash quantitative_experiments/revision/run_h3.sh --dry-run \
  | tee "$EXP_ROOT/logs/h3_preview.log"

# Este comando hace reranking y llamadas con costo: 800 respuestas más evaluación.
bash quantitative_experiments/revision/run_h3.sh --run \
  2>&1 | tee "$EXP_ROOT/logs/h3.log"
```

Los identificadores anteriores son los valores del código, no una comprobación de disponibilidad del proveedor. Confirmar acceso al modelo/juez elegidos antes de la corrida y fijar cualquier cambio en el protocolo antes de generar respuestas. Para `JUDGE_API=openai`, usar el identificador correspondiente de esa API y su clave; no cambiar de juez durante la corrida.

Las condiciones son `base`, `finetuned`, `base_rerank` y `finetuned_rerank`, con las mismas 200 preguntas y contexto top-10. El prompt actual genera un experimento nuevo; no reconstruye la Tabla 13 histórica. Revisar `status.json`, `answers/summary.json` y la cobertura por condición/métrica. Reportar respuestas vacías, rechazos, errores y citas sin respaldo además de Faithfulness.

### Si quedan scores incompletos

Para continuar una corrida interrumpida o con scores incompletos, relanzar con la misma configuración y el mismo commit, añadiendo `--resume`:

```bash
bash quantitative_experiments/revision/run_h3.sh --run --resume
```

`--resume` reutiliza `H3_OUT_DIR`, **se niega a continuar si el protocolo congelado cambió** (insumos, prompt, modelos, temperatura, juez o commit de git), salta los pasos ya completos (rerank y respuestas de cada condición) y reintenta los incompletos apoyándose en las cachés, de modo que no se vuelve a pagar lo ya hecho. Sin `--resume`, el wrapper sigue negándose a usar una carpeta existente.

Alternativa manual: si las cuatro generaciones están completas y el único problema son scores faltantes, conservar los archivos y reintentar **solo scoring**, con el mismo entorno, juez, API y configuración originales. Volver a definir las variables de esa corrida si se abrió otra terminal. Este reintento también puede hacer llamadas con costo:

```bash
python -m quantitative_experiments.revision.h3_checks "$H3_OUT_DIR"
python -m quantitative_experiments.revision.answer_quality score \
  --conditions base finetuned base_rerank finetuned_rerank --reference base \
  --dataset quantitative_experiments/results/revision/test_b/eval_test_b_dataset.jsonl \
  --out-dir "$H3_OUT_DIR/answers" \
  --judge-model "$JUDGE_MODEL" --judge-api "$JUDGE_API" --max-tokens 20000 --concurrency 4 \
  2>&1 | tee "$EXP_ROOT/logs/h3_score_retry.log"
python -m quantitative_experiments.revision.answer_quality citations \
  --conditions base finetuned base_rerank finetuned_rerank \
  --dataset quantitative_experiments/results/revision/test_b/eval_test_b_dataset.jsonl \
  --out-dir "$H3_OUT_DIR/answers"
python -m quantitative_experiments.revision.h3_checks "$H3_OUT_DIR"
```

Los scores completos compatibles se reutilizan y los incompletos se reintentan. Revisar la nueva cobertura en `summary.json`; el reintento manual no actualiza el estado global del wrapper. Conservar el `status.json` original y entregar el log de recuperación junto al resumen completo; no marcarlo como `complete` a mano. Si hubo cambios de insumos, de respuestas o de juez, conservar ese intento como separado y documentar cómo se repetirá la corrida. Si falló la generación, estos comandos de scoring no la reparan.

Answer Relevancy admite valores negativos válidos (similitud coseno); no eliminarlos ni cambiarlos a cero. Para el análisis opcional `code-errors`, conservar `summary.json` y los archivos `*.ragas_partial.jsonl`: el resumen identifica el juez y los puntajes exactos que debe usar. Los resúmenes antiguos requieren volver a ejecutar `score` antes de ese análisis.

## 5. Análisis conjunto y archivos que entregar

Con los tres resultados primarios completos, ejecutar sin GPU ni API:

```bash
python -m quantitative_experiments.revision.confirmatory \
  --h1 "$H1_RUN_DIR/comparison.json" \
  --h2 "$H2_OUT_DIR/evaluation/test_b_primary.json" \
  --h3 "$H3_OUT_DIR/answers/summary.json" \
  --out "$EXP_ROOT/confirmatory.json" \
  2>&1 | tee "$EXP_ROOT/logs/confirmatory.log"
```

Entregar al equipo las comparaciones, resultados por pregunta, rankings, trazas/respuestas, scores y cobertura, análisis de citas, `protocol.json`/`status.json`, selección de checkpoints, logs, versiones, commits y manifiestos de hashes. En H2 incluir `training_data/build_report.json`, `provenance.json`, `train.log`, `pip_freeze.txt` y `gpu.txt` de cada semilla. Mantener los checkpoints/pesos en el almacenamiento de modelos acordado, fuera del PR; compartir su ubicación y hashes. **No subir claves, `.env`, cachés ni pesos a GitHub.** Los resultados se revisan antes de decidir qué artefactos publicar.

Conservar y reportar también resultados negativos o nulos. H1–H3 completos no sustituyen la línea de base competitiva recomendada, las semillas adicionales si se afirma estabilidad, ni la evaluación independiente de crisis si se retienen afirmaciones de seguridad.
