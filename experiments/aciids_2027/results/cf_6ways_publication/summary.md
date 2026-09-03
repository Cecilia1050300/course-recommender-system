# Six-way collaborative filtering publication rerun

## Protocol

All similarities, scalers, rating matrices, and fallback statistics were fitted from TRAIN only. Each method searched K in [1, 3, 5, 7, 9] on validation, selecting lowest RMSE, then highest NDCG@5, then smaller K. The six selections were written before test was loaded; test was evaluated once per selected method with the shared evaluator.

Fallback is deterministic: valid similarity-weighted CF → target-item TRAIN mean → user TRAIN mean → global TRAIN mean. Rating evaluation clips to [1,5] in the shared evaluator; ranking uses raw scores. No jitter or artificial error penalty is used.

## Legacy versus publication

| Method | Legacy K | Legacy MAE | Legacy RMSE | Legacy NDCG | Publication K | Publication MAE | Publication RMSE | Publication NDCG@5 | Publication NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| User_Rating | 9 | 0.934411 | 1.195902 | 0.920545 | 9 | 0.979477 | 1.233450 | 0.003985 | 0.005031 |
| User_Content | 9 | 1.032660 | 1.276851 | 0.906970 | 9 | 1.081693 | 1.304234 | 0.001427 | 0.002385 |
| User_Hybrid | 7 | 0.939017 | 1.201835 | 0.917650 | 9 | 0.983697 | 1.236578 | 0.001427 | 0.003431 |
| Item_Rating | 9 | 0.885539 | 1.138862 | 0.861046 | 9 | 0.905183 | 1.186338 | 0.012522 | 0.012522 |
| Item_Content | 9 | 0.882044 | 1.103645 | 0.855232 | 9 | 0.940430 | 1.211957 | 0.010569 | 0.017389 |
| Item_Hybrid | 9 | 0.854585 | 1.075760 | 0.860213 | 7 | 0.910807 | 1.193898 | 0.160392 | 0.177006 |

**Legacy NDCG must not be compared numerically with publication NDCG@5/@10.** Legacy NDCG ranks only each user's held-out rows, uses graded gain `2^rating−1`, and excludes single-row users. Publication metrics rank the full 151-course catalog after excluding TRAIN-observed items, use binary relevance at rating ≥4, and apply the shared eligible-user rule. Legacy splitting, test-selected K, hash jitter, and artificial tier-2 error also differ.

## Ranking by test RMSE

| Rank | Method | Selected K | RMSE |
|---:|---|---:|---:|
| 1 | Item_Rating | 9 | 1.186338 |
| 2 | Item_Hybrid | 7 | 1.193898 |
| 3 | Item_Content | 9 | 1.211957 |
| 4 | User_Rating | 9 | 1.233450 |
| 5 | User_Hybrid | 9 | 1.236578 |
| 6 | User_Content | 9 | 1.304234 |

## Ranking by test NDCG@5

| Rank | Method | Selected K | NDCG@5 |
|---:|---|---:|---:|
| 1 | Item_Hybrid | 7 | 0.160392 |
| 2 | Item_Rating | 9 | 0.012522 |
| 3 | Item_Content | 9 | 0.010569 |
| 4 | User_Rating | 9 | 0.003985 |
| 5 | User_Content | 9 | 0.001427 |
| 6 | User_Hybrid | 9 | 0.001427 |

## Answers

1. **Strongest User-CF:** `User_Rating` by the primary rating criterion. Ranking-specific ordering is shown separately.
2. **Strongest Item-CF:** `Item_Rating` by the primary rating criterion.
3. **Does content help?** Not by RMSE in either family; effects are method-family and metric dependent.
4. **Does hybridization help?** Not beyond both component-only variants by RMSE.
5. **Is RMSE-best also NDCG-best?** No: `Item_Rating` is RMSE-best and `Item_Hybrid` is NDCG@5-best.
6. **Rating–ranking mismatch inside CF:** Supported descriptively: the RMSE and NDCG@5 winners differ. This is a six-method comparison, not a hyperparameter-grid correlation study.
7. **Main-table CF method:** `Item_Rating`, because K and method reporting follow the publication primary criterion. The NDCG-leading CF should remain visible in the ranking table if different.

See `validation_results.csv`, `final_test_results.csv`, `fallback_audit.csv`, and `evaluation_audit.md` for exact metrics and protocol checks.
