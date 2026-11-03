# Pooling de qrels del set de test

Los qrels estrictos de `test_set/` solo marcan el chunk de origen de cada pregunta; otro chunk que también la responde cuenta
como error y el recall queda como cota inferior, de forma desigual entre sistemas. El pooling junta los top-k de **todos** los
sistemas que se comparan y un juez independiente (LLM `gpt-6.1-sol` vía API, ciego al sistema y al rol del chunk) decide la relevancia
de cada par (pregunta, chunk). Un sistema que no aporta al pool queda subestimado: por eso entran todos los de la tabla final.

Autosuficiente: usa `../test_set/`, `../data/inputs/document_chunks_all.jsonl` y `prompts/judge_system.txt`. No depende del repo de datos.

## Pasos (desde esta carpeta)

```bash
# 0. Rankings de todas las configuraciones sobre el test (profundidad 100):  ../ablation/run_phases.sh  ->  ../results/runs/*.json
# 1. Armar el pool (top-20 de cada sistema; el nombre del archivo es el del sistema)
python pool_eval.py export --runs ../results/runs/*.json --depth 20
# 2. Juzgar por API (OpenRouter; solo toma las peticiones pendientes, se puede repetir; desde la raíz del repo)
uv run python quantitative_experiments/pooling/run_judge.py
# 3. Validar respuestas y construir los qrels
python pool_eval.py ingest
```

Si `ingest` informa tareas pendientes o mal formadas (sale con código 3), borra esas respuestas de
`output/work/pool_test/responses/`, ejecuta `export` de nuevo (regenera las peticiones pendientes) y repite el paso 2. Los juicios ya hechos se reutilizan: si se añade un sistema o cambia un
ranking, `export` solo pide los pares nuevos. Si cambian el corpus, el dataset o el prompt del juez, borra `output/` y repite (el
programa lo detecta con hashes en `output/pool_manifest.json`).

Variables y opciones de `run_judge.py`: `POOL_JUDGE_MODEL` o `--model` (id de OpenRouter del juez, por defecto `openai/gpt-6.1-sol`), `--effort` (medium), `--concurrency` (6) y `--limit`. Los qrels pooled solo son válidos con todas las peticiones resueltas: un par sin juzgar cuenta como no relevante. Antes de `ingest` comprueba que `python pool_eval.py pending | wc -l` da 0.

## Salidas (`output/`)

| Archivo | Contenido |
|---|---|
| `pool_pairs_test.jsonl` | pares (pregunta, chunk) y qué sistemas los recuperaron (el juez **no** lo ve; queda como registro del pool) |
| `pool_judgments_test.jsonl` | veredicto por par: `si` / `parcial` / `no`, justificación y modelo juez |
| `eval_test_qrels_pooled.jsonl` | qrels principales: chunk de origen + pares `si` |
| `eval_test_qrels_pooled_parcial.jsonl` | sensibilidad: origen + `si` + `parcial` |
| `pool_test_report.json` | conteos y procedencia de los jueces |
