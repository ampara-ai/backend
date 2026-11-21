# Checkpoint selection on the development set (130 questions, strict qrels)

Query prefix (JSON string): `""`

| Model | NDCG@10 | Recall@10 | Recall@1 |
| --- | --- | --- | --- |
| cls:/Volumes/external/h2_run/training/full_matched_seed2/checkpoint-85 | 0.411 | 0.562 | 0.285 |
| cls:/Volumes/external/h2_run/training/full_matched_seed2/checkpoint-170 | 0.442 | 0.608 | 0.300 |
| cls:/Volumes/external/h2_run/training/full_matched_seed2/checkpoint-255 | 0.469 | 0.677 | 0.300 |
| cls:/Volumes/external/h2_run/training/full_matched_seed2/checkpoint-340 | 0.480 | 0.677 | 0.315 |
| cls:/Volumes/external/h2_run/training/full_matched_seed2/checkpoint-425 (selected) | 0.484 | 0.669 | 0.323 |
| cls:/Volumes/external/h2_run/training/full_matched_seed2/checkpoint-430 | 0.484 | 0.669 | 0.323 |
| cls:/Volumes/external/h2_run/training/full_matched_seed2 | 0.484 | 0.669 | 0.323 |
