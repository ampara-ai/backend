# Revision results: H1, H2 and H3

Raw outputs of the portable revision experiments, kept exactly as the scripts wrote them. Each folder has a `SHA256SUMS`
file; verify with `cd h1 && shasum -a 256 -c SHA256SUMS` (same for `h2` and `h3`).

## H1 — published checkpoint vs base (`h1/`)

`ampara-ai/bge-m3-ampara` (revision `f2fab1a2…`) against `BAAI/bge-m3` (revision `5617a9f6…`) on frozen Test B
(200 questions, 113 source clusters), strict NDCG@10, FP16 on Apple MPS.

| Base | Fine-tuned | Difference | 95% CI |
| ---: | ---: | ---: | --- |
| 0.515 | 0.643 | +0.127 | [0.076, 0.177] |

Single fixed checkpoint, so the interval does not include training-seed variation.

## H2 — full protocol vs naive baseline (`h2/`)

Six training runs (two arms × seeds 42, 1, 2; 1,376 rows and 430 updates each), checkpoint chosen per run on the
development set only, then evaluated on Test B (primary) and Test A (supporting) with strict NDCG@10.

| Test | `naive` mean | `full_matched` mean | Difference | 95% CI | raw bootstrap p |
| --- | ---: | ---: | ---: | --- | ---: |
| B (primary) | 0.629 | 0.558 | −0.072 | [−0.163, 0.000] | 0.053 |
| A (supporting) | 0.572 | 0.505 | −0.067 | [−0.136, −0.009] | 0.021 |

**This is a negative result:** the full protocol did not outperform the naive baseline and was less stable across
seeds (seed SD 0.079 vs 0.017 on Test B). Three seeds per arm give limited information, and the comparison changes
several data-construction components at once. The Holm adjustment across H1–H3 is pending and is not applied here.

The model weights and every checkpoint are in the public Hugging Face repository
[`ampara-ai/bge-m3-ampara-h2-runs`](https://huggingface.co/ampara-ai/bge-m3-ampara-h2-runs) at commit
`c3109a21dba5d08ab2c511e515090660a44bdff0`; its `SHA256SUMS` covers every file there.

## H3 — answer quality with the generator fixed (`h3/`)

Same 200 Test B questions and the same generator (`openai/gpt-oss-120b`, temperature 0, prose generation prompt) in
four conditions; only the top-10 context changes. Contexts come from the H1 rankings of the base and the published
checkpoint, with and without reranking by Voyage `rerank-2.5` (served by OpenRouter as `mongodb/rerank-2.5`). Judge:
`openai/gpt-5.6-luna` through RAGAS 0.4.3. Faithfulness is the primary metric.

| Condition | Faithfulness | Source-context support | Answer relevancy |
| --- | ---: | ---: | ---: |
| base | 0.615 | 0.324 | 0.687 |
| fine-tuned | 0.603 | 0.346 | 0.693 |
| base + rerank | 0.636 | 0.330 | 0.692 |
| fine-tuned + rerank | 0.657 | 0.329 | 0.698 |

| Faithfulness vs base | Difference | 95% CI | raw bootstrap p |
| --- | ---: | --- | ---: |
| fine-tuned (primary) | −0.012 | [−0.055, 0.031] | 0.586 |
| base + rerank (exploratory) | +0.020 | [−0.019, 0.059] | 0.319 |
| fine-tuned + rerank (exploratory) | +0.041 | [−0.001, 0.084] | 0.055 |

**Null result on the primary contrast:** the fine-tuned checkpoint did not change Faithfulness. The other two metrics
show no significant difference in any condition. Reranking improved citation precision (strict 0.81 → 0.88) and cut
answers with an unsupported citation from 9.0% to 3.8–4.4% (`h3/answers/citations.md`), but its Faithfulness gains are
exploratory and not distinguishable from zero. "Source-context support" is stored under the legacy key `gold_support`;
it is not legal validation. The Holm adjustment across H1–H3 is pending and is not applied here.

The first scoring attempt stopped after the `base` condition because the API key reached its spending limit
(`h3/log_attempt1_stopped_by_key_limit.log`). The run was resumed with `--resume` from the same commit; generations and
reranks were reused and only the missing judgments were computed (`h3/log_resume.log`).

## Notes

- Files contain absolute paths from the machine that produced them (for example `/Users/…` and `/Volumes/external/…`).
  They were left untouched on purpose so the files stay byte-identical to the originals and the hashes remain valid.
- H3 ran locally with the reranker and judge accessed through OpenRouter and the generator through Groq; no
  fine-tuning or GPU was involved.
- H2 training ran on Google Colab (NVIDIA L4); evaluation ran on Apple MPS. `full_matched_seed2` was relaunched after a
  Colab VM was lost before producing output, and `naive_seed42` was interrupted once and restarted; logs of the
  interrupted attempts were overwritten.
- These results measure source retrieval on synthetic questions, not legal correctness.
