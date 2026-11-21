# Comparison on eval_test_b_dataset.jsonl (200 questions, qrels eval_test_b_qrels.jsonl)

Reference: naive. Intervals: 95% bootstrap over source chunks, seeds resampled within each replicate.

## ndcg@10

| System | Seeds | Mean | Seed SD | Diff vs reference | 95% CI | p (boot) | p (Holm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| naive (reference) | 3 | 0.629 | 0.017 | – | – | – | – |
| full_matched | 3 | 0.558 | 0.079 | -0.072 | [-0.163, 0.000] | 0.0530 | – |
