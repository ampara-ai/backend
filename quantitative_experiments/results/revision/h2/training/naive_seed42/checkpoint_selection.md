# Checkpoint selection on the development set (130 questions, strict qrels)

Query prefix (JSON string): `""`

| Model | NDCG@10 | Recall@10 | Recall@1 |
| --- | --- | --- | --- |
| cls:/Volumes/external/h2_run/training/naive_seed42/checkpoint-85 | 0.467 | 0.638 | 0.323 |
| cls:/Volumes/external/h2_run/training/naive_seed42/checkpoint-170 | 0.505 | 0.654 | 0.369 |
| cls:/Volumes/external/h2_run/training/naive_seed42/checkpoint-255 | 0.512 | 0.700 | 0.346 |
| cls:/Volumes/external/h2_run/training/naive_seed42/checkpoint-340 | 0.508 | 0.700 | 0.338 |
| cls:/Volumes/external/h2_run/training/naive_seed42/checkpoint-425 (selected) | 0.516 | 0.685 | 0.362 |
| cls:/Volumes/external/h2_run/training/naive_seed42/checkpoint-430 | 0.516 | 0.685 | 0.362 |
| cls:/Volumes/external/h2_run/training/naive_seed42 | 0.516 | 0.685 | 0.362 |
