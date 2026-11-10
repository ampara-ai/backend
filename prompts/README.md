# Prompts

`app/core/prompt_provider.py` carga estos archivos al importarse. Ningún otro archivo de esta carpeta se envía al LLM.

| Archivo | Uso |
| --- | --- |
| `relevance_filter.md` | Filtro de dominio |
| `context_check.md` | Decisión de recuperación |
| `generation.md` | Generador desplegado (secciones fijas y canales de ayuda; desde el commit 589eb6d, 29-09-2026) |
| `generation_prose_v1.md` | Generador en prosa: versión de `generation.md` anterior a 589eb6d, recuperada de git |

El generador se elige con `GENERATION_PROMPT_FILE` (por defecto `generation.md`).

`generation_prose_v1.md` es el **candidato** del prompt con el que se evaluó la Tabla 13 del artículo: las 174 respuestas de
`quantitative_experiments/results/pipeline_final/` están en prosa y ninguna tiene las secciones ni los canales de ayuda que exige
`generation.md`. Quien corrió esa evaluación debe confirmar que es exactamente el que se usó; los experimentos de la revisión
(`quantitative_experiments/revision/answer_quality.py`) registran el archivo y su SHA-256 en cada corrida.
