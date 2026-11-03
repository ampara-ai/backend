# Resultados de la ablación (test, 198 preguntas) — para completar los marcadores [ABLATION] del paper

Fuentes: `results/runs` (recuperador aislado, las 198 preguntas), `results/pipeline` (pipeline completo: las preguntas que rechaza el filtro de dominio cuentan con ranking vacío), qrels estrictos (chunk de origen) y pooled (origen + "sí" del juez, 11.763 pares). Recall@k, NDCG@k y MRR@k con la definición del proyecto (hits / min(|G|,k)). Un solo entrenamiento; modelos INT8 servidos.


## Recuperador aislado — qrels estrictos

| Configuración | Recall@10 | MRR@10 | NDCG@10 | MAP@10 | Recall@100 | MRR@100 | NDCG@100 | MAP@100 |
|---|---|---|---|---|---|---|---|---|
| bm25_only | 0.111 | 0.049 | 0.063 | 0.049 | 0.480 | 0.061 | 0.136 | 0.061 |
| dense_base | 0.626 | 0.398 | 0.453 | 0.398 | 0.886 | 0.408 | 0.507 | 0.408 |
| dense_finetuned | 0.801 | 0.499 | 0.571 | 0.499 | 0.970 | 0.508 | 0.610 | 0.508 |
| hybrid_base | 0.616 | 0.365 | 0.426 | 0.365 | 0.874 | 0.375 | 0.478 | 0.375 |
| hybrid_finetuned | 0.785 | 0.482 | 0.555 | 0.482 | 0.970 | 0.490 | 0.595 | 0.490 |
| dense_base_rerank100 | 0.881 | 0.715 | 0.755 | 0.713 | 0.886 | 0.715 | 0.756 | 0.713 |
| dense_finetuned_rerank100 | 0.960 | 0.773 | 0.819 | 0.772 | 0.970 | 0.774 | 0.821 | 0.773 |
| hybrid_base_rerank100 | 0.869 | 0.708 | 0.746 | 0.704 | 0.874 | 0.708 | 0.747 | 0.705 |
| hybrid_finetuned_rerank100 | 0.960 | 0.774 | 0.819 | 0.772 | 0.970 | 0.774 | 0.822 | 0.773 |

## Recuperador aislado — qrels pooled

| Configuración | Recall@10 | MRR@10 | NDCG@10 | MAP@10 | Recall@100 | MRR@100 | NDCG@100 | MAP@100 |
|---|---|---|---|---|---|---|---|---|
| bm25_only | 0.114 | 0.126 | 0.084 | 0.051 | 0.463 | 0.140 | 0.176 | 0.068 |
| dense_base | 0.503 | 0.520 | 0.427 | 0.337 | 0.855 | 0.527 | 0.527 | 0.363 |
| dense_finetuned | 0.651 | 0.635 | 0.545 | 0.439 | 0.970 | 0.642 | 0.650 | 0.480 |
| hybrid_base | 0.494 | 0.498 | 0.407 | 0.314 | 0.838 | 0.506 | 0.505 | 0.339 |
| hybrid_finetuned | 0.637 | 0.618 | 0.532 | 0.426 | 0.953 | 0.624 | 0.631 | 0.462 |
| dense_base_rerank100 | 0.813 | 0.870 | 0.774 | 0.696 | 0.855 | 0.870 | 0.792 | 0.707 |
| dense_finetuned_rerank100 | 0.894 | 0.917 | 0.843 | 0.767 | 0.970 | 0.917 | 0.876 | 0.791 |
| hybrid_base_rerank100 | 0.798 | 0.868 | 0.767 | 0.689 | 0.838 | 0.868 | 0.783 | 0.697 |
| hybrid_finetuned_rerank100 | 0.885 | 0.916 | 0.837 | 0.758 | 0.953 | 0.916 | 0.866 | 0.778 |

## Pipeline completo — qrels estrictos

| Configuración | Recall@10 | MRR@10 | NDCG@10 | MAP@10 | Recall@100 | MRR@100 | NDCG@100 | MAP@100 |
|---|---|---|---|---|---|---|---|---|
| bm25_only | 0.101 | 0.046 | 0.059 | 0.046 | 0.429 | 0.057 | 0.124 | 0.057 |
| dense_base | 0.520 | 0.316 | 0.365 | 0.316 | 0.770 | 0.326 | 0.417 | 0.326 |
| dense_finetuned | 0.689 | 0.419 | 0.484 | 0.419 | 0.848 | 0.427 | 0.520 | 0.428 |
| hybrid_base | 0.520 | 0.291 | 0.346 | 0.291 | 0.758 | 0.300 | 0.394 | 0.300 |
| hybrid_finetuned | 0.674 | 0.412 | 0.475 | 0.411 | 0.848 | 0.419 | 0.513 | 0.419 |
| dense_base_rerank100 | 0.765 | 0.609 | 0.647 | 0.607 | 0.770 | 0.610 | 0.648 | 0.607 |
| dense_finetuned_rerank100 | 0.838 | 0.665 | 0.707 | 0.664 | 0.848 | 0.665 | 0.710 | 0.664 |
| hybrid_base_rerank100 | 0.753 | 0.602 | 0.638 | 0.599 | 0.758 | 0.603 | 0.639 | 0.599 |
| hybrid_finetuned_rerank100 | 0.838 | 0.665 | 0.707 | 0.664 | 0.848 | 0.666 | 0.710 | 0.664 |

## Pipeline completo — qrels pooled

| Configuración | Recall@10 | MRR@10 | NDCG@10 | MAP@10 | Recall@100 | MRR@100 | NDCG@100 | MAP@100 |
|---|---|---|---|---|---|---|---|---|
| bm25_only | 0.096 | 0.114 | 0.073 | 0.044 | 0.407 | 0.126 | 0.157 | 0.059 |
| dense_base | 0.419 | 0.436 | 0.355 | 0.277 | 0.743 | 0.443 | 0.446 | 0.301 |
| dense_finetuned | 0.563 | 0.550 | 0.471 | 0.377 | 0.849 | 0.556 | 0.565 | 0.414 |
| hybrid_base | 0.418 | 0.417 | 0.342 | 0.260 | 0.727 | 0.425 | 0.429 | 0.283 |
| hybrid_finetuned | 0.551 | 0.537 | 0.462 | 0.370 | 0.838 | 0.543 | 0.552 | 0.402 |
| dense_base_rerank100 | 0.705 | 0.759 | 0.672 | 0.602 | 0.743 | 0.759 | 0.688 | 0.612 |
| dense_finetuned_rerank100 | 0.780 | 0.806 | 0.737 | 0.667 | 0.849 | 0.806 | 0.767 | 0.689 |
| hybrid_base_rerank100 | 0.692 | 0.757 | 0.666 | 0.596 | 0.727 | 0.757 | 0.680 | 0.604 |
| hybrid_finetuned_rerank100 | 0.776 | 0.805 | 0.734 | 0.663 | 0.838 | 0.805 | 0.760 | 0.681 |

## Filtro de dominio y umbral (dense_finetuned, pipeline completo)

- Rechazadas por el filtro de dominio: **24 de 198** (12.1%); casi todas sobre derecho penal/procesal general (herencia, difamación, plazos de condena), por tipo de chunk de origen: Ley 30364 2, CP 5, CPP 12, jurisprudencia 5.
- Coseno del mejor chunk (preguntas que pasaron el filtro, n=174): mín 0.406, mediana 0.520, máx 0.693.
- Respuestas "sin fuentes suficientes" por el umbral (puntaje normalizado > 0,5): **0**.


## Bootstrap pareado por chunk de origen (114 clusters, 10.000 remuestreos, IC95 %), qrels estrictos, recuperador aislado

| Comparación | Métrica | Diferencia | IC95 % |
|---|---|---|---|
| denso entrenado − denso base (**primaria**) | NDCG@10 | +0.119 | [0.063, 0.176] |
| denso entrenado − denso base (**primaria**) | Recall@10 | +0.174 | [0.103, 0.247] |
| denso entrenado − BM25 | NDCG@10 | +0.508 | [0.451, 0.566] |
| denso entrenado − híbrido entrenado | NDCG@10 | +0.016 | [−0.006, 0.038] |
| denso entrenado + reranker − denso entrenado | NDCG@10 | +0.248 | [0.193, 0.302] |
| denso entrenado + reranker − denso entrenado | Recall@10 | +0.159 | [0.105, 0.215] |
| denso entrenado + reranker − denso base + reranker | NDCG@10 | +0.064 | [0.032, 0.100] |

Solo la primera fila es la comparación primaria; el resto son secundarias, sin corrección por multiplicidad. Con qrels pooled la diferencia primaria es NDCG@10 +0.119 [0.073, 0.166] y Recall@10 +0.148 [0.096, 0.199].
