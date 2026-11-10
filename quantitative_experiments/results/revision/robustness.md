# Robustness analyses on the released results

95% bootstrap intervals over source chunks unless noted. Regenerate with `uv run python -m quantitative_experiments.revision.robustness`.

| Analysis | Measure | Result |
| --- | --- | --- |
| Memorization: corpus without training chunks | strict NDCG@10 FT − base | +0.128 [0.079, 0.178] vs +0.119 [0.063, 0.176] on the full corpus |
| #42 pooled positives from the training partition | pooled NDCG@10 FT − base without them | 352 of 499; +0.120 [0.071, 0.171] |
| #40 strict pairs | questions with >1 strict relevant | 200 pairs / 198 questions; test::chunk_tipo02_cp_art_58_01::q01: ['chunk_tipo02_cp_art_58_01', 'chunk_tipo02_cp_art_64_01']; test::chunk_tipo02_cp_art_58_01::q02: ['chunk_tipo02_cp_art_58_01', 'chunk_tipo02_cp_art_64_01'] |
| #30 Table 2 vs Table 3 | near-duplicate members beyond the first, by partition | train 14, dev 1, test 1 |
| #32 judged negatives, clustered by question | 6432 pairs, 536 questions | yes 2.3% [1.9, 2.8]; partial 16.0% [14.8, 17.2] |
| #27 lexical overlap halves | strict NDCG@10 BM25 / base / FT | low (n=111): 0.020 / 0.419 / 0.541; high (n=87): 0.119 / 0.495 / 0.610 |
| Test questions by generator | strict NDCG@10 FT − base | claude-sonnet-4-5 (n=48): +0.084 [-0.013, 0.176]; claude-sonnet-5-5 (n=150): +0.130 [0.060, 0.199] |
| #6 Faithfulness by retrieval outcome | mean Faithfulness (n) | strict hit 0.64 (137), strict miss 0.47 (37), pooled miss 0.35 (20) |
| #47 domain filter rejections | FT retriever strict Recall@10 | 24 rejected (same set in all configs: True); 0.92 on rejected vs 0.78 on accepted |
| Power of Test A | minimum detectable NDCG@10 difference (80% power) | dense_finetuned vs dense_base: 0.080; dense_finetuned vs hybrid_finetuned: 0.032; dense_finetuned_rerank100 vs dense_base_rerank100: 0.048 |
| Gain by source type | strict NDCG@10 base → FT | tipo01 (n=18): 0.552 → 0.601; tipo02 (n=35): 0.511 → 0.644; tipo03 (n=80): 0.365 → 0.552; tipo04 (n=65): 0.502 → 0.548 |
| Paired wins and losses | source chunk in top-10 for only one model | FT only 45, base only 10, both 114, neither 29 |
