# Current evaluator definition and legacy distinction

## Current full-catalog evaluator

- Candidate set: the ordered 151-course catalog minus each user's TRAIN-observed course IDs. Test labels do not define candidates.
- Relevance: unique TEST course IDs with rating >= 4. Unlabelled candidates are non-relevant.
- Eligible user: a TEST user with at least one relevant TEST item. There are 185.
- Binary DCG@K: `sum(1/log2(rank+1))` at relevant one-based ranks through K (gain 1, otherwise 0).
- IDCG@K: the same discounts at ranks 1 through `min(K, number of relevant items)`.
- NDCG@K: DCG/IDCG. NDCG@5 and @10 differ only in K.
- Averaging: unweighted macro mean over eligible users; ineligible values are NaN and skipped.
- Ties: score descending, then lexicographically ascending Course_ID.
- Missing/non-finite score: TRAIN global rating mean. Rating predictions are clipped to [1,5], ranking scores are not clipped.
- TRAIN exclusion is literal set membership. Candidate count is `151 - unique TRAIN items for user`.

The hand examples in `sample_user_manual_checks.md` establish the formula independently. Bounds follow because binary DCG cannot exceed the ideal placement of the same number of relevant items, so 0 <= NDCG <= 1.

## Legacy evaluator

The original implementations are `methods/matrix_factorization/MF_experiment_ndcg.py:33-42`, `methods/similarity_6ways/cf_experiment_ndcg.py:171-180`, `methods/rwr/RWR_experiment_ndcg.py:85-93`, and `methods/gcn/gcn.py:133-150`. They rank only each user's held-out rows, sort by predicted score, use graded gain `2^rating - 1`, have no K truncation (all held-out rows), construct IDCG by sorting held-out true ratings descending, exclude users with <=1 held-out row from the reported macro mean, and otherwise average users equally. The helper itself returns 1 for <=1 row or zero IDCG, but callers exclude single-row users.

## Three metrics that must not be mixed

1. Exact legacy: held-out-only, graded `2^rating-1`, no K, users with >=2 rows.
2. Held-out-only publication-style diagnostic: held-out-only, binary rating>=4, @5/@10, users with >=1 relevant item.
3. Current publication metric: full catalog minus TRAIN-observed, same binary definition, @5/@10, users with >=1 relevant item.

Exact old reported values are preserved in the publication summaries; reproducing them on the current split would not be an exact legacy reproduction because the legacy split and several legacy prediction/fallback behaviors differ.
