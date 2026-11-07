# Second judge: google/gemma-4-31b-it

Agreement on 11961 pairs: 85.5%; Cohen's kappa 0.535 (3 classes), 0.731 (si vs rest).

| Qrels | Relevant pairs | FT − base NDCG@10 | FT − base Recall@10 |
| --- | --- | --- | --- |
| judge1 | 699 | +0.119 [0.073, 0.166] | +0.148 [0.096, 0.199] |
| judge2 | 814 | +0.117 [0.073, 0.162] | +0.138 [0.086, 0.193] |
| intersection | 568 | +0.131 [0.081, 0.180] | +0.167 [0.108, 0.226] |
| union | 945 | +0.106 [0.064, 0.149] | +0.118 [0.071, 0.167] |

| Config | judge1 | judge2 | intersection | union |
| --- | --- | --- | --- | --- |
| bm25_only | 0.084 | 0.086 | 0.081 | 0.089 |
| dense_base | 0.427 | 0.431 | 0.433 | 0.428 |
| dense_finetuned | 0.545 | 0.548 | 0.563 | 0.534 |
| hybrid_base | 0.407 | 0.407 | 0.413 | 0.405 |
| hybrid_finetuned | 0.532 | 0.531 | 0.549 | 0.519 |
| dense_base_rerank100 | 0.774 | 0.778 | 0.787 | 0.773 |
| dense_finetuned_rerank100 | 0.843 | 0.845 | 0.858 | 0.838 |
| hybrid_base_rerank100 | 0.767 | 0.770 | 0.777 | 0.767 |
| hybrid_finetuned_rerank100 | 0.837 | 0.844 | 0.856 | 0.832 |
