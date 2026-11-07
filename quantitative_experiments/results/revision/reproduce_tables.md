# Reproduction of Tables 9, 10 and 12

Largest absolute difference from the manuscript: 0.0005. Verdict: OK.

## table9_pooled (pooled qrels)

| Config | recall@10 | mrr@10 | ndcg@10 | map@10 | recall@100 | mrr@100 | ndcg@100 | map@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_only | 0.114 | 0.126 | 0.084 | 0.051 | 0.463 | 0.140 | 0.176 | 0.068 |
| dense_base | 0.503 | 0.520 | 0.427 | 0.337 | 0.855 | 0.527 | 0.527 | 0.363 |
| dense_finetuned | 0.651 | 0.635 | 0.545 | 0.439 | 0.970 | 0.642 | 0.650 | 0.480 |
| hybrid_base | 0.494 | 0.498 | 0.407 | 0.314 | 0.838 | 0.506 | 0.505 | 0.339 |
| hybrid_finetuned | 0.637 | 0.618 | 0.532 | 0.426 | 0.953 | 0.624 | 0.631 | 0.462 |
| dense_base_rerank100 | 0.813 | 0.870 | 0.774 | 0.696 | 0.855 | 0.870 | 0.792 | 0.707 |
| dense_finetuned_rerank100 | 0.894 | 0.917 | 0.843 | 0.767 | 0.970 | 0.917 | 0.876 | 0.791 |
| hybrid_base_rerank100 | 0.798 | 0.868 | 0.767 | 0.689 | 0.838 | 0.868 | 0.783 | 0.697 |
| hybrid_finetuned_rerank100 | 0.885 | 0.916 | 0.837 | 0.758 | 0.953 | 0.916 | 0.866 | 0.778 |

## table10_strict (strict qrels)

| Config | recall@10 | mrr@10 | ndcg@10 | map@10 | recall@100 | mrr@100 | ndcg@100 | map@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_only | 0.111 | 0.049 | 0.063 | 0.049 | 0.480 | 0.061 | 0.136 | 0.061 |
| dense_base | 0.626 | 0.398 | 0.453 | 0.398 | 0.886 | 0.408 | 0.507 | 0.408 |
| dense_finetuned | 0.801 | 0.499 | 0.571 | 0.499 | 0.970 | 0.508 | 0.610 | 0.508 |
| hybrid_base | 0.616 | 0.365 | 0.426 | 0.365 | 0.874 | 0.375 | 0.478 | 0.375 |
| hybrid_finetuned | 0.785 | 0.482 | 0.555 | 0.482 | 0.970 | 0.490 | 0.595 | 0.490 |
| dense_base_rerank100 | 0.881 | 0.715 | 0.755 | 0.713 | 0.886 | 0.715 | 0.756 | 0.713 |
| dense_finetuned_rerank100 | 0.960 | 0.773 | 0.819 | 0.772 | 0.970 | 0.774 | 0.821 | 0.773 |
| hybrid_base_rerank100 | 0.869 | 0.708 | 0.746 | 0.704 | 0.874 | 0.708 | 0.747 | 0.705 |
| hybrid_finetuned_rerank100 | 0.960 | 0.774 | 0.819 | 0.772 | 0.970 | 0.774 | 0.822 | 0.773 |

## table12 (paired differences, bootstrap over source chunks)

| Comparison | Metric | Strict (recomputed) | Pooled (recomputed) |
| --- | --- | --- | --- |
| dense_finetuned − dense_base | ndcg@10 | +0.119 [0.063, 0.176] | +0.119 [0.073, 0.166] |
| dense_finetuned − dense_base | recall@10 | +0.174 [0.103, 0.247] | +0.148 [0.096, 0.199] |
| dense_finetuned − bm25_only | ndcg@10 | +0.508 [0.451, 0.566] | +0.461 [0.412, 0.512] |
| dense_finetuned − hybrid_finetuned | ndcg@10 | +0.016 [-0.006, 0.038] | +0.014 [-0.001, 0.029] |
| dense_finetuned_rerank100 − dense_finetuned | ndcg@10 | +0.248 [0.193, 0.302] | +0.298 [0.255, 0.341] |
| dense_finetuned_rerank100 − dense_finetuned | recall@10 | +0.159 [0.105, 0.215] | +0.243 [0.201, 0.287] |
| dense_finetuned_rerank100 − dense_base_rerank100 | ndcg@10 | +0.064 [0.032, 0.100] | +0.069 [0.041, 0.100] |
