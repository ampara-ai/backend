# Revisión para el envío a revista Q1: alineación del backend

Esta rama empieza a alinear el código con lo que describe el artículo de AMPARA y agrega las herramientas de evaluación que necesita la revisión. No cambia la recuperación ni el prompt de generación.

## Cambios en esta rama

| Cambio | Archivos | Verificación |
| --- | --- | --- |
| Núcleo de evaluación: métricas de recuperación (Recall, MRR, MAP, NDCG @k), bootstrap pareado por clústeres y acuerdo entre evaluadores (AC1/AC2 de Gwet, alfa de Krippendorff) | `evaluation/` | Métricas idénticas a ranx (6 decimales). Alfa reproduce el ejemplo publicado de Krippendorff (0.743 / 0.815 / 0.849). AC1/AC2 y alfa coinciden con `irrCAC` y `krippendorff` |
| Proveedor del LLM fijado en cada request: solo Groq, sin fallback, sin proveedores que guardan datos | `app/core/config.py`, `app/core/dependencies.py`, `.env.example` | Revisado en el payload real, también en las llamadas con salida estructurada |
| Source grounding con salida estructurada en lugar de JSON parseado a mano (7 de 174 respuestas de la evaluación caían al fallback por JSON inválido) | `app/graph/nodes/source_grounding.py` | Tests con un LLM simulado |
| Expiración de sesiones inactivas: el historial se borra de memoria tras `SESSION_IDLE_TTL_MINUTES` (15 por defecto) | `app/graph/rag_graph.py`, `app/api/v1/chat.py` | Tests de expiración |
| 23 tests nuevos y `conftest.py` con variables de entorno de prueba | `tests/` | `uv run pytest`: 28 pasan |

En datos concentrados en 4–5, como las calificaciones del panel de expertos, el alfa de Krippendorff cae cerca de 0 mientras el AC2 de Gwet se mantiene cerca de 0.90. Por eso conviene reportar AC1/AC2.

## Diferencias entre el código y el artículo

| # | Tema | Artículo | Código | Estado |
| --- | --- | --- | --- | --- |
| 1 | Temperatura del generador | 0.7 | 0 en todas las llamadas (`app/core/dependencies.py`) | Confirmar qué valor produjo los resultados reportados |
| 2 | Umbral de similitud | Coseno > 0.5 filtra chunks y activa el mensaje de respaldo | No hay umbral; `no_results_node` no está conectado al grafo | Propuesta: quitarlo del artículo |
| 3 | Prompt de generación | Prosa cálida, sin listas de pasos | Secciones fijas (qué puedes hacer, la ley te respalda, dónde pedir ayuda) y aviso de seguridad (105, Línea 100) | Describir en el artículo el prompt real |
| 4 | Memoria de conversación | Últimos cinco pares pregunta–respuesta | Filtro de relevancia: últimos 5 mensajes; context check y generador: historial completo | Decidir: una sola ventana o corregir el artículo |
| 5 | Workers | Cuatro en Azure | Un solo proceso (`Dockerfile`); el código asume uno | Confirmar el despliegue |
| 6 | Fuentes mostradas | El fragmento recuperado con su identificador | Título y resumen escritos por el LLM | Salida estructurada hecha; falta decidir qué se muestra |
| 7 | Proveedor del LLM | Solo Groq | No estaba fijado en el código | Hecho en esta rama |
| 8 | Reranker | Servicio externo que no se puede alojar | model-service ya sirve bge-reranker-v2-m3 en `/rerank` | Evaluarlo con 20, 50 y 100 candidatos y medir latencia |
| 9 | Configuración de RAGAS | Juez GPT-5.6 Luna, generador gpt-oss-120b | `comandos.txt` usa gpt-5.6-luna-pro como generador y juez | Confirmar; el generador no debe juzgarse a sí mismo |
| 10 | Evaluación de recuperación | 198 preguntas, 9 configuraciones, juicios agrupados, bootstrap | 82 preguntas, 5 configuraciones; sin pooling ni bootstrap (`scripts/eval_ablation.py`) | Publicar el código y los datos finales |
| 11 | Repositorios | Cuatro, incluido el de datos | Tres públicos; `README.md` vacío | Publicar el repositorio de datos y un README con los comandos de reproducción |
| 12 | Experimentos no reportados | No se mencionan | Reformulación de la consulta, interpolación de scores, reranker Qwen, RRF ponderado | Reportarlos o repetirlos con los nuevos conjuntos de prueba |

## Hallazgo en el frontend

El frontend público guarda el `session_id` en `localStorage` (`src/lib/api.ts`), así que sobrevive a una recarga de la página. El backend conserva el historial de esa sesión y lo vuelve a usar como contexto, aunque la pantalla aparezca vacía.

Esto contradice la sección 3.6 del artículo, según la cual una recarga borra la conversación, y es un riesgo si quien agrede revisa el dispositivo. La expiración de esta rama limita la exposición a 15 minutos. La corrección completa está en el frontend: limpiar la sesión al cargar la página o usar `sessionStorage`.

## Pendientes

El detalle de lo que necesitamos de los coautores, con el formato de entrega, está en [requerimientos-coautores.md](requerimientos-coautores.md).

- [ ] Datos y código finales de la evaluación: 198 preguntas de prueba, juicios agrupados, archivo de entrenamiento de 2,057 registros y resultados por consulta.
- [ ] Calificaciones originales del panel de expertos, para el análisis de acuerdo.
- [ ] Confirmar la temperatura, la configuración de RAGAS y el número de workers.
- [ ] Decidir qué muestran las fuentes en la interfaz y si se quita el umbral del artículo.
- [ ] Corregir la sesión en el frontend.
- [ ] Restringir CORS al dominio del frontend y fallar al arrancar si `JWT_SECRET_KEY` sigue con el valor por defecto.

## Cómo verificar

```bash
uv sync
uv run pytest
```
