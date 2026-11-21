# Checkpoint selection on the development set (130 questions, strict qrels)

Query prefix (JSON string): `""`

| Model | NDCG@10 | Recall@10 | Recall@1 |
| --- | --- | --- | --- |
| cls:/Volumes/external/h2_run/training/naive_seed1/checkpoint-85 | 0.448 | 0.608 | 0.300 |
| cls:/Volumes/external/h2_run/training/naive_seed1/checkpoint-170 (selected) | 0.505 | 0.685 | 0.346 |
| cls:/Volumes/external/h2_run/training/naive_seed1/checkpoint-255 | 0.496 | 0.677 | 0.331 |
| cls:/Volumes/external/h2_run/training/naive_seed1/checkpoint-340 | 0.499 | 0.692 | 0.331 |
| cls:/Volumes/external/h2_run/training/naive_seed1/checkpoint-425 | 0.491 | 0.669 | 0.323 |
| cls:/Volumes/external/h2_run/training/naive_seed1/checkpoint-430 | 0.491 | 0.669 | 0.323 |
| cls:/Volumes/external/h2_run/training/naive_seed1 | 0.491 | 0.669 | 0.323 |
