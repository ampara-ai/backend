# Comparison on eval_test_dataset.jsonl (198 questions, qrels eval_test_qrels.jsonl)

Reference: naive. Intervals: 95% bootstrap over source chunks, seeds resampled within each replicate.

## ndcg@10

| System | Seeds | Mean | Seed SD | Diff vs reference | 95% CI | p (boot) | p (Holm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| naive (reference) | 3 | 0.572 | 0.018 | – | – | – | – |
| full_matched | 3 | 0.505 | 0.053 | -0.067 | [-0.136, -0.009] | 0.0214 | – |
