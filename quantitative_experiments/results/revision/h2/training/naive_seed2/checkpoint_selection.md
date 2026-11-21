# Checkpoint selection on the development set (130 questions, strict qrels)

Query prefix (JSON string): `""`

| Model | NDCG@10 | Recall@10 | Recall@1 |
| --- | --- | --- | --- |
| cls:/Volumes/external/h2_run/training/naive_seed2/checkpoint-85 | 0.463 | 0.638 | 0.315 |
| cls:/Volumes/external/h2_run/training/naive_seed2/checkpoint-170 | 0.501 | 0.692 | 0.346 |
| cls:/Volumes/external/h2_run/training/naive_seed2/checkpoint-255 | 0.495 | 0.685 | 0.331 |
| cls:/Volumes/external/h2_run/training/naive_seed2/checkpoint-340 | 0.499 | 0.669 | 0.354 |
| cls:/Volumes/external/h2_run/training/naive_seed2/checkpoint-425 (selected) | 0.501 | 0.669 | 0.362 |
| cls:/Volumes/external/h2_run/training/naive_seed2/checkpoint-430 | 0.501 | 0.677 | 0.354 |
| cls:/Volumes/external/h2_run/training/naive_seed2 | 0.501 | 0.677 | 0.354 |
