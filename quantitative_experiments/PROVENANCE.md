# Insumos congelados del experimento

Copias exactas (verificables con `SHA256SUMS`: `cd quantitative_experiments && shasum -a 256 -c SHA256SUMS`) de los archivos que usa
el experimento. No se editan a mano. Se generaron en el repositorio de datos (`peru-legal-data-rag`), que no se modifica desde aquí.

| Archivo | Origen (repo de datos) | Qué es |
|---|---|---|
| `data/inputs/document_chunks_all.jsonl` | `recoleccion/data/processed/document_chunks_all.jsonl` | corpus limpio de 893 chunks (el que se indexa en Mongo) |
| `data/inputs/corpus_inventory.csv` | derivado de los metadatos del corpus | inventario por chunk: tipo, familia de fuente, identificador normativo, título, longitud y hash del texto |
| `test_set/eval_test_dataset.jsonl` | `generacion_dataset/data/output/dataset_v2/eval_test_dataset.jsonl` | 198 preguntas de test con su chunk de origen (`positive_chunk_ids`) |
| `test_set/eval_test_queries.jsonl` | `.../dataset_v2/eval_test_queries.jsonl` | solo las consultas (`id`, `text`) |
| `test_set/eval_test_qrels.jsonl` | `.../dataset_v2/eval_test_qrels.jsonl` | qrels **estrictos** (solo el chunk de origen y sus duplicados) |
| `pooling/prompts/judge_system.txt` | `generacion_dataset/pipeline_v2/prompts/judge_system.txt` | prompt del juez de relevancia del pooling |

El split es **por chunk** (hecho antes de generar nada): ningún chunk de test aparece en el entrenamiento ni como positivo ni como
negativo, y las preguntas de test se generaron a partir de su chunk con un LLM generador (Claude Sonnet 5.5) y se verificaron con un
juez independiente (`gpt-6.1-sol`). Snapshot de origen: `dataset_v2/snapshots/ola_1.0`.
