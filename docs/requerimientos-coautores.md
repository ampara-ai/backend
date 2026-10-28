# Requerimientos de los coautores para la revisión Q1

Para continuar con la revisión del artículo necesitamos insumos que solo tienen quienes corrieron los experimentos finales. Los grupos 1 y 3 bloquean el trabajo: sin ellos no se pueden repetir ni ampliar los resultados reportados.

## 1. Datos y código finales (bloqueante)

- [ ] Conjuntos de preguntas: prueba (198), desarrollo (130) y entrenamiento (2,057 registros, 1,970 preguntas), con su partición y su chunk fuente.
- [ ] Veredictos del juez sobre los pares positivos y negativos (9,011 pares), con su etiqueta yes / partial / no.
- [ ] Juicios de relevancia estrictos y agrupados (11,763 pares juzgados).
- [ ] Rankings por consulta de las 9 configuraciones (top-100), y resultados por consulta de RAGAS y de latencia.
- [ ] Código que produjo los números del artículo: 9 configuraciones, RRF ponderado (1:2, k = 5), reranker sobre 100 candidatos, pooling y bootstrap. Este repositorio solo tiene una versión anterior (82 preguntas, 5 configuraciones).
- [ ] Notebook de entrenamiento y constructor del dataset: prompts de generación de preguntas, filtros y minado de negativos con BM25.
- [ ] Publicar el repositorio de datos que cita el artículo. Hoy hay tres repositorios públicos, no cuatro.

## 2. Datos a confirmar

| Pregunta | Artículo | Código |
| --- | --- | --- |
| Temperatura del generador en los resultados reportados | 0.7 | 0 |
| Modelos generador y juez de la corrida de RAGAS reportada | gpt-oss-120b y GPT-5.6 Luna | gpt-5.6-luna-pro en ambos (`comandos.txt`) |
| Workers en Azure | 4 | 1 |
| Prompt de generación | Prosa, sin listas de pasos | Secciones fijas con pasos numerados |

- [ ] Por qué hay 200 pares estrictos para 198 preguntas.
- [ ] Por qué la Tabla 2 (644 / 77 / 165 chunks) y la Tabla 3 (630 / 76 / 164) no coinciden.
- [ ] Si se guardaron los resultados de los experimentos anteriores: reformulación de la consulta, reranker Qwen e interpolación de scores.
- [ ] Nombre de la fuente legal secundaria y del servicio de resúmenes, y si su autor permite nombrarlo.

## 3. Material del panel de expertos (bloqueante para la validación)

- [ ] Calificaciones originales: 5 expertos × 20 respuestas × 3 criterios, con los comentarios.
- [ ] Especialidad de cada experto, por identificador y sin nombres.
- [ ] Cómo se eligieron las 20 preguntas y respuestas, cuándo se calificaron y qué versión del sistema las generó.
- [ ] Constancia del consentimiento de los expertos.
- [ ] Si el panel aceptaría una ronda más, a ciegas, comparando AMPARA con el mismo LLM sin recuperación.

## 4. Decisiones entre todos los autores

- [ ] Revisar y aprobar el PR #1, en especial el proveedor fijado en Groq sin fallback (si Groq cae, el request falla) y la expiración de sesión a los 15 minutos.
- [ ] Interfaz: mostrar el texto real recuperado (cambio en el frontend) o corregir el artículo.
- [ ] Quitar del artículo el umbral de similitud 0.5, que el código no aplica (recomendado).
- [ ] Memoria de conversación: una sola ventana en todos los nodos, o describir en el artículo lo que hace el código.
- [ ] Congelar el corpus en 893 chunks.
- [ ] Corregir en el frontend la sesión que sobrevive a una recarga (ver `revision-q1.md`).

## 5. Declaraciones para el artículo

- [ ] Fuentes de financiamiento.
- [ ] Conflictos de interés.
- [ ] Contribución de cada autor.

## Formato de entrega

Con estos formatos los datos se cargan sin conversiones. Si ya existen en otro formato, compártanlos tal cual y los adaptamos.

| Insumo | Formato | Campos |
| --- | --- | --- |
| Preguntas | JSONL | `query_id`, `query`, `partition`, `source_chunk_id`, `origin` (ruling / chunk) |
| Juicios de relevancia | JSONL o qrels TREC | `query_id`, `chunk_id`, `label`, `judge` |
| Rankings | Run TREC | `query_id Q0 chunk_id rank score config` |
| Resultados por consulta | CSV | `query_id`, `config` y una columna por métrica |
| Calificaciones del panel | CSV | `expert_id`, `item_id`, `criterion`, `rating`, `comment` |
