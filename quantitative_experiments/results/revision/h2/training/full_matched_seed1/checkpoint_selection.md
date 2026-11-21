# Checkpoint selection on the development set (130 questions, strict qrels)

Query prefix (JSON string): `""`

| Model | NDCG@10 | Recall@10 | Recall@1 |
| --- | --- | --- | --- |
| cls:/Volumes/external/h2_run/training/full_matched_seed1/checkpoint-85 | 0.308 | 0.415 | 0.215 |
| cls:/Volumes/external/h2_run/training/full_matched_seed1/checkpoint-170 | 0.354 | 0.500 | 0.231 |
| cls:/Volumes/external/h2_run/training/full_matched_seed1/checkpoint-255 | 0.365 | 0.523 | 0.223 |
| cls:/Volumes/external/h2_run/training/full_matched_seed1/checkpoint-340 | 0.372 | 0.523 | 0.223 |
| cls:/Volumes/external/h2_run/training/full_matched_seed1/checkpoint-425 (selected) | 0.388 | 0.554 | 0.254 |
| cls:/Volumes/external/h2_run/training/full_matched_seed1/checkpoint-430 | 0.386 | 0.554 | 0.246 |
| cls:/Volumes/external/h2_run/training/full_matched_seed1 | 0.386 | 0.554 | 0.246 |
