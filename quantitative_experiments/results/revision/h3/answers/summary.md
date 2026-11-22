# Answer quality with the generator fixed (200 questions; judge openai/gpt-5.6-luna)

Source-context support retains the legacy JSON key `gold_support`; it is not legal validation.

| Condition | Faithfulness | Source-context support | Answer relevancy |
| --- | --- | --- | --- |
| base | 0.615 | 0.324 | 0.687 |
| finetuned | 0.603 | 0.346 | 0.693 |
| base_rerank | 0.636 | 0.330 | 0.692 |
| finetuned_rerank | 0.657 | 0.329 | 0.698 |

| Condition | Metric | Scored | Missing | Not applicable |
| --- | --- | --- | --- | --- |
| base | faithfulness | 200 | 0 | 0 |
| base | gold_support | 200 | 0 | 0 |
| base | answer_relevancy | 200 | 0 | 0 |
| finetuned | faithfulness | 200 | 0 | 0 |
| finetuned | gold_support | 200 | 0 | 0 |
| finetuned | answer_relevancy | 200 | 0 | 0 |
| base_rerank | faithfulness | 200 | 0 | 0 |
| base_rerank | gold_support | 200 | 0 | 0 |
| base_rerank | answer_relevancy | 200 | 0 | 0 |
| finetuned_rerank | faithfulness | 200 | 0 | 0 |
| finetuned_rerank | gold_support | 200 | 0 | 0 |
| finetuned_rerank | answer_relevancy | 200 | 0 | 0 |

Paired differences vs base (questions with both scores):

| Condition | Metric | Paired n | Missing pairs | Diff [95% CI] | p (boot) |
| --- | --- | --- | --- | --- | --- |
| finetuned | faithfulness | 200 | 0 | -0.012 [-0.055, 0.031] | 0.5856 |
| finetuned | gold_support | 200 | 0 | +0.022 [-0.014, 0.058] | 0.2298 |
| finetuned | answer_relevancy | 200 | 0 | +0.006 [-0.007, 0.021] | 0.4218 |
| base_rerank | faithfulness | 200 | 0 | +0.020 [-0.019, 0.059] | 0.3188 |
| base_rerank | gold_support | 200 | 0 | +0.006 [-0.030, 0.042] | 0.7242 |
| base_rerank | answer_relevancy | 200 | 0 | +0.005 [-0.009, 0.019] | 0.5246 |
| finetuned_rerank | faithfulness | 200 | 0 | +0.041 [-0.001, 0.084] | 0.0546 |
| finetuned_rerank | gold_support | 200 | 0 | +0.005 [-0.033, 0.044] | 0.7894 |
| finetuned_rerank | answer_relevancy | 200 | 0 | +0.011 [-0.005, 0.028] | 0.2118 |
