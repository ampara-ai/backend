# Checkpoint selection on the development set (130 questions, strict qrels)

Query prefix (JSON string): `""`

| Model | NDCG@10 | Recall@10 | Recall@1 |
| --- | --- | --- | --- |
| cls:/Volumes/external/h2_run/training/full_matched_seed42/checkpoint-85 | 0.425 | 0.592 | 0.300 |
| cls:/Volumes/external/h2_run/training/full_matched_seed42/checkpoint-170 | 0.411 | 0.546 | 0.292 |
| cls:/Volumes/external/h2_run/training/full_matched_seed42/checkpoint-255 | 0.456 | 0.623 | 0.308 |
| cls:/Volumes/external/h2_run/training/full_matched_seed42/checkpoint-340 (selected) | 0.477 | 0.677 | 0.308 |
| cls:/Volumes/external/h2_run/training/full_matched_seed42/checkpoint-425 | 0.476 | 0.669 | 0.308 |
| cls:/Volumes/external/h2_run/training/full_matched_seed42/checkpoint-430 | 0.472 | 0.669 | 0.300 |
| cls:/Volumes/external/h2_run/training/full_matched_seed42 | 0.472 | 0.669 | 0.300 |
