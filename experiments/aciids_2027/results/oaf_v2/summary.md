# Objective-Aligned Fusion V2 — validation only

**Final verdict: NOT SUPPORTED. TEST WAS LOADED: NO.**

## Pair supervision

TRAIN-only graded ordinal pairs: 28,076 from 314 users. Gap counts: 1=16,528, 2=8,280, 3=2,779, 4=489. These exactly match the prior feasibility audit.

Rating metrics use raw MF/LightGCN fusion. Ranking loss and ranking metrics use separately per-user standardized branch scores. No unobserved course is treated as a negative.

## Main validation points

| Point | alpha/lambda | RMSE | NDCG@5 | Recall@5 | Eligible |
|---|---|---:|---:|---:|:---:|
| MF | alpha=1 | 0.992841 | 0.202880 | 0.257843 | Yes |
| LightGCN-RMSE | alpha=0 | 1.120470 | 0.451169 | 0.647059 | No |
| Prior router best | tau=20 | 1.116060 | 0.462006 | 0.643137 | No |
| V1 fixed alpha=.25 | alpha=.25 | 1.062769 | 0.430567 | 0.578431 | Yes |
| Best V2 fixed | alpha=0.25 | 1.062769 | 0.430567 | 0.578431 | Yes |
| Best global joint | alpha=0.125426, lambda=0.01, unweighted | 1.091255 | 0.445058 | 0.612745 | Yes |

Across the complete selectable pool, the predeclared tie-break selects the corresponding `lambda_rank=0` global-alpha control: its NDCG@5 and Recall@5 are identical, its RMSE is infinitesimally lower, and lower lambda is preferred. The lambda=0.01 row above is the best strictly positive-lambda diagnostic, not the selected method.

## Required answers

1. **lambda_rank > 0 versus MSE-only:** No eligible joint point improves NDCG@5 over the best MSE-only global fusion.
2. **10% RMSE constraint:** Preserved by the reported best joint point.
3. **Best fixed alpha:** `0.25` (RMSE 1.062769, NDCG@5 0.430567).
4. **Best learned global alpha:** lambda `0.01`, initial alpha `0.1`, checkpoint alpha `0.125426`, `unweighted` (RMSE 1.091255, NDCG@5 0.445058).
5. **Versus UAF-V1 gates:** best joint NDCG@5 0.445058 versus UAF-count 0.347651 and UAF-rich 0.350683.
6. **Versus fixed alpha=.25:** Improves.
7. **Versus previous hard router:** Does not improve the best router NDCG@5.
8. **Existing Pareto frontier expanded:** Formally YES: a positive-lambda global point enters the combined frontier. Relative to the V2 MSE-only global-alpha control, however, this is only a negligible floating-point RMSE displacement with unchanged NDCG@5/Recall@5—not a meaningful ranking-frontier expansion attributable to objective alignment.
9. **Larger lambda shifts toward LightGCN:** The mean checkpoint alpha is monotonically non-increasing across lambda, but only from 0.637971 at lambda=0 to 0.636791 at lambda=1. This tiny displacement is not a meaningful systematic shift. This is a controlled within-grid association, not a causal generalization.
10. **Gap weighting:** best gap-weighted NDCG@5 0.445058 versus unweighted 0.445058; gap weighting did not help.
11. **OAF-user-count:** Not run because V2-B did not meet the predeclared material-improvement trigger.
12. **Verdict:** **NOT SUPPORTED** under the predeclared validation-only success criteria.

## Alpha versus ranking-loss weight

| lambda_rank | Mean checkpoint alpha | Std | Min | Max |
|---:|---:|---:|---:|---:|
| 0.00 | 0.637971 | 0.259811 | 0.104611 | 0.904411 |
| 0.01 | 0.637936 | 0.259808 | 0.104611 | 0.904411 |
| 0.05 | 0.637804 | 0.259795 | 0.104611 | 0.904411 |
| 0.10 | 0.637731 | 0.259833 | 0.104611 | 0.904411 |
| 0.25 | 0.637492 | 0.259898 | 0.104610 | 0.904411 |
| 0.50 | 0.637206 | 0.259995 | 0.104610 | 0.904411 |
| 1.00 | 0.636791 | 0.260103 | 0.104609 | 0.904411 |

Across completed checkpoints, raw TRAIN rating loss ranged 0.246155–0.600755 and raw ordinal ranking loss ranged 0.142074–0.455595. The preflight scale ratio was below the declared 100× stopping threshold, so no objective rescaling was introduced.

## Scientific interpretation

The result tests objective alignment for a one-dimensional global mixture. Any observed lambda–alpha relationship is specific to these frozen branch scores, this split, and the declared loss scaling. No test claim is made.
