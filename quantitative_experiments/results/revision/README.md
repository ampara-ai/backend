# Revision results: H1 and H2

Raw outputs of the portable revision experiments, kept exactly as the scripts wrote them. Each folder has a `SHA256SUMS`
file; verify with `cd h1 && shasum -a 256 -c SHA256SUMS` (same for `h2`).

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

## Notes

- Files contain absolute paths from the machine that produced them (for example `/Users/…` and `/Volumes/external/…`).
  They were left untouched on purpose so the files stay byte-identical to the originals and the hashes remain valid.
- H2 training ran on Google Colab (NVIDIA L4); evaluation ran on Apple MPS. `full_matched_seed2` was relaunched after a
  Colab VM was lost before producing output, and `naive_seed42` was interrupted once and restarted; logs of the
  interrupted attempts were overwritten.
- These results measure source retrieval on synthetic questions, not legal correctness.
