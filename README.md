# Ampara — backend RAG legal

Chatbot RAG sobre violencia contra las mujeres y el grupo familiar en el Perú (tesis de pregrado). Pipeline de
producción: embedding BGE-M3 fine-tuned (ONNX, `model-service`) → búsqueda vectorial en MongoDB Atlas → generación
con LLM vía OpenRouter (LangGraph + FastAPI, Postgres para usuarios y sesiones).

Arranque y despliegue: `docker compose up -d --build` (usa `.env.production`); local, `.env` con `localhost`.
Ingesta del corpus e índices: `scripts/ingest_chunks.py` (ver el docstring del script).

---

# Experimento de ablación de recuperación (OE4)

Todo el material vive en [quantitative_experiments/](quantitative_experiments/). Compara configuraciones de recuperación
sobre preguntas cuyos chunks positivos **nunca se vieron en el entrenamiento** del embedding.

Nombre de configuración = `<etapa>_<embedding>[_rerank<K>]`:

| Config | Descripción |
|---|---|
| `bm25_only` | BM25 (índice Lucene español) |
| `dense_<v>` | solo denso con el embedding de la variante `<v>` |
| `hybrid_<v>` | BM25 + denso con RRF ponderado, parámetros **fijos**: peso del denso 2,0 (BM25 1,0) y k=5 |
| `dense_<v>_rerank100`, `hybrid_<v>_rerank100` | lo anterior + reranker (Voyage rerank-2.5) sobre los 100 candidatos que se evalúan |

Variantes de embedding: `base` (BGE-M3 sin entrenar, `/embed_base`) y `finetuned` (BGE-M3 entrenado con el dataset v2, `/embed`).

**Reranker.** Puntúa los 100 candidatos (no solo los 10 que se entregan), así que puede subir al top-10 chunks de las
posiciones 11..100 y Recall@10 ya no queda fijo por construcción. Recall@100 sí coincide con el de la etapa previa (el
reranker solo reordena dentro de esos 100); MRR@100 y NDCG@100 reflejan el reordenamiento. Es una configuración de
experimento: el pipeline productivo entrega como máximo 10 chunks. Los puntajes se guardan en
`quantitative_experiments/data/rerank_cache.jsonl`. Costo con OpenRouter ≈ 0,0018 USD por consulta con 100 candidatos
(un reranker autoalojado en CPU resultó demasiado lento para evaluar 100 candidatos por consulta).

**Parámetros fijos.** Los de la fusión híbrida se eligieron una sola vez en dev con el embedding base
(`ablation/tune_hybrid.py`, rejilla en `ablation/hybrid_tuning_dev_base.csv`: peso del denso 2,0 y k=5, NDCG@10 estricto 0,378) y
se aplican sin reajustar a las dos variantes; el reranker usa K=100 sin ajuste. Ni el test ni más corridas de dev intervienen en
elegirlos. El RRF estándar (1:1, k=60) daba 0,271 en dev con el base, frente a 0,375 del denso solo: BM25 aporta poco y la
fusión sin pesos lo perjudica. Las conclusiones valen para estas configuraciones.

Métricas: Recall, MRR, NDCG y MAP @10 y @100. Recall@k, MAP@k y NDCG@k se normalizan por min(|relevantes|, k).

## Estructura

```
quantitative_experiments/
├── data/inputs/        corpus de 893 chunks (document_chunks_all.jsonl)
├── test_set/           preguntas de test y qrels estrictos (dataset v2)
├── pooling/            pooling de qrels de test: pool_eval.py, run_judge.py, prompt del juez
├── ablation/           eval_ablation, build_runs, run_phases.sh, score_runs, build_final_table, paired_bootstrap
├── results/            salidas (rankings, puntajes)
├── PROVENANCE.md       origen de cada insumo
└── SHA256SUMS          hashes de los insumos
```

## Reproducción

Requisitos: `uv sync`; un `.env` (ver `.env.example`) con `OPENROUTER_API_KEY` (generador del pipeline, reranker del experimento, juez del pooling, juez de RAGAS), la conexión a MongoDB Atlas y `MODEL_SERVICE_URL`; el repositorio del model service junto a este (o `MODEL_SERVICE_DIR`) para registrar el hash del modelo servido. Los resultados ya calculados están en `quantitative_experiments/results/` y se pueden re-puntuar sin ningún servicio (paso C).

Desde la raíz del repo. Verificar insumos: `cd quantitative_experiments && shasum -a 256 -c SHA256SUMS`.

**A. Rankings** — requiere el model service y MongoDB. Un solo campo `embedding` en Mongo sirve a una variante a la vez,
así que se reindexa entre variantes. `ablation/run_phases.sh` encadena: rankings de BM25; reindexado
(`scripts/ingest_chunks.py` con `--with-base-embeddings` o `--with-embeddings`) y rankings del denso, del híbrido y de las
configuraciones con reranker, para cada variante, sobre el set de test
(`DS=quantitative_experiments/test_set/eval_test_dataset.jsonl OUT=quantitative_experiments/results/runs bash quantitative_experiments/ablation/run_phases.sh`;
las variables están documentadas en el script).

Cada ranking guardado tiene profundidad 100 (`results/runs/<config>.json`). Dos salvaguardas, ambas en `eval_ablation.py`:

- La búsqueda vectorial del experimento es exacta (`exact=True`, ENN), no ANN: el ranking no depende de cómo Atlas
  reconstruya el índice.
- Mongo puede contener chunks ajenos al corpus (p. ej. documentos de prueba subidos por la API de admin). Se descartan de
  cada lista de candidatos antes de fusionar o reordenar (`CORPUS_PATH`).

**A2. Pipeline completo.** `ablation/full_pipeline.py` evalúa las mismas configuraciones **dentro del grafo de AMPARA**, reutilizando los nodos de
producción (filtro de dominio, decisión de recuperación, generación y source grounding; el grafo desplegado no se modifica): solo cambia el paso de recuperación,
que ejecuta la configuración de la ablación y guarda 100 candidatos para las métricas @10 y @100. Escribe, por configuración, el ranking a nivel pipeline
(`<config>.json`, las preguntas rechazadas por el filtro cuentan con ranking vacío), a nivel recuperador (`<config>.retriever.json`) y una traza por pregunta
(`<config>.trace.jsonl`, con la respuesta y las fuentes si se usa `--with-generation`). `ablation/run_full_pipeline.sh` encadena base y finetuned; solo la arquitectura final (`dense_finetuned`) genera respuesta, las demás llegan hasta la recuperación
(`DS=... OUT=quantitative_experiments/results/pipeline bash quantitative_experiments/ablation/run_full_pipeline.sh`).
El grafo desplegado aplica un umbral de similitud: solo llegan al generador los chunks cuyo puntaje de Atlas, (1 + coseno) / 2, supera 0,5 (coseno > 0); si ninguno lo supera, responde con el mensaje fijo de "sin fuentes suficientes" (`settings.retrieval_score_threshold`).
Las salidas van a una carpeta distinta de `results/runs` (que alimenta el pooling).

**A3. RAGAS y latencia.** `ablation/run_ragas.sh` evalúa Faithfulness y Answer Relevancy de las respuestas del pipeline final con el juez `openai/gpt-5.6-luna` y los embeddings `qwen/qwen3-embedding-8b` (OpenRouter), y resume la latencia por respuesta y por etapa. Es reanudable (guarda cada pregunta evaluada) y tiene modos `pilot` (10 respuestas, para medir el costo), `full` y `retry` (`MODE=full bash quantitative_experiments/ablation/run_ragas.sh`).

**B. Qrels completos por pooling** (los qrels estrictos solo marcan el chunk de origen: otro chunk que también responde
cuenta como error y subestima el recall, de forma desigual entre métodos). Detalle en `pooling/README.md`:

1. `python pooling/pool_eval.py export --runs quantitative_experiments/results/runs/*.json --depth 20` arma el pool con los
   top-k de TODOS los sistemas que se comparan.
2. `uv run python quantitative_experiments/pooling/run_judge.py` — juicio de relevancia (si/parcial/no) por API (OpenRouter), ciego al sistema y al rol.
3. `python pooling/pool_eval.py ingest` — qrels pooled (origen + "si"; sensibilidad: + "parcial").

**C. Métricas y comparación**

```bash
uv run python -m quantitative_experiments.ablation.score_runs --dataset-path quantitative_experiments/test_set/eval_test_dataset.jsonl --runs-dir quantitative_experiments/results/runs --out-dir quantitative_experiments/results/scores/retriever_strict
uv run python -m quantitative_experiments.ablation.score_runs ... --qrels quantitative_experiments/pooling/output/eval_test_qrels_pooled.jsonl --out-dir quantitative_experiments/results/scores/retriever_pooled
uv run python -m quantitative_experiments.ablation.paired_bootstrap --results-dir quantitative_experiments/results/scores/retriever_pooled --dataset-path quantitative_experiments/test_set/eval_test_dataset.jsonl --cluster-by positive_chunk
uv run --with matplotlib python -m quantitative_experiments.ablation.make_figures   # figura de la ablación (PNG y PDF)
```

Se reportan ambas variantes de qrels: estricta (cota inferior) y ampliada. `score_runs` re-puntúa las mismas corridas sin
tocar Mongo. `eval_ablation` (misma recuperación, más Faithfulness y Answer Relevancy con RAGAS) y `build_final_table` se
usan en la pasada final.

## Notas para el auditor

- Dataset v2 (repo de datos): partición por chunk antes de generar; generador Claude Sonnet 5.5 y juez independiente
  `gpt-6.1-sol`. Pooling de test: mismo juez, ciego al sistema y al rol del chunk.
- Selección de hiperparámetros: la época del entrenamiento se eligió en dev con los qrels estrictos; los parámetros del híbrido
  salieron una sola vez de dev con el embedding base y no se reajustaron (`ablation/hybrid_tuning_dev_base.csv`). El test se
  evalúa una sola vez.
- `document_chunks_all.jsonl` de la raíz es el corpus limpio de 893 chunks (idéntico a `quantitative_experiments/data/inputs/`) y es el valor por defecto de `scripts/ingest_chunks.py`.
