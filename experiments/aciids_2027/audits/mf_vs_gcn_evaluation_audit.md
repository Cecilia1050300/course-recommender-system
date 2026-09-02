# MF vs Vanilla GCN Evaluation Audit

## Purpose

This audit verifies whether the publication-scale Matrix Factorization and Vanilla GCN baselines are evaluated under identical Top-K recommendation semantics.

The audit was read-only. No model, split, evaluator, or result artifact was modified.

## Evaluation Semantics

| Evaluation semantic | Result | Finding |
|---|---|---|
| Candidate construction | SAME | Full 151-item catalog minus each user's train-observed items |
| Exclusion of train items | SAME | Identical observed-item exclusion |
| Relevance threshold | SAME | Test rating >= 4.0 |
| Test positives | SAME | Same 599 test rows, relevant counts, and 185 eligible users |
| Score direction | SAME | Higher score ranks first |
| Clipping behavior | SAME | [1,5] clipping applies to MAE/RMSE only; ranking uses supplied scores without evaluator clipping |
| Fallback behavior | SAME | Missing/non-finite predictions use train global mean |
| Per-user eligibility | SAME | Users without relevant test items are excluded |
| Precision@K | SAME | hits / K |
| Recall@K | SAME | hits / number_of_relevant_items |
| NDCG@K | SAME | Binary relevance, logarithmic discount, ideal list limited to min(K, relevant_count) |
| HitRate@K | SAME | 1 when at least one relevant item appears in Top-K |
| Tie-breaking | SAME | Course ID ascending after score descending |

## Verified Evaluation Population

- Same 458 evaluated users
- Same candidate counts per user: 121-150
- Same test user/item keys and actual ratings
- No fallback rows for either model
- No cold-start rows for either model

## Model-Side Score Processing Difference

There is one model-side preprocessing difference before the shared evaluator.

### Matrix Factorization

MF ranks raw dot-product-plus-bias scores directly.

### Vanilla GCN

GCN applies per-user MinMax scaling over non-train candidate items into [1,5] before ranking.

This MinMax transform is monotonically increasing within each user and therefore preserves candidate ordering exactly.

If all candidate scores are equal, all transformed scores become 3.0, but the original scores were already tied and both models use the same course-ID tie-breaking policy.

Therefore, the MinMax transformation does not explain GCN's higher Recall or NDCG.

The transformation can affect rating calibration and therefore MAE/RMSE.

- MF had 8 held-out predictions clipped for rating-error calculation.
- GCN had 0 because its predictions were already scaled into [1,5].

## Publication Results

| Model | Test MAE | Test RMSE | Precision@5 | Recall@5 | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| Matrix Factorization | 0.829269 | 1.024343 | 0.062703 | 0.261261 | 0.193768 |
| Vanilla GCN | 1.488966 | 1.827887 | 0.099459 | 0.471171 | 0.244365 |

## Scientific Interpretation

The MF-versus-GCN Top-K results are scientifically comparable under the current benchmark.

Both models use the same:

- test positives
- candidate construction
- train-item exclusion
- relevance threshold
- ranking metric definitions
- per-user eligibility rules
- tie-breaking policy

Therefore, the observed Top-K ranking difference reflects different learned candidate orderings rather than an evaluation-path discrepancy.

Current empirical observation:

> Better explicit-rating prediction does not necessarily imply better Top-K recommendation ranking in this sparse course-recommendation benchmark.

MF provides substantially better MAE/RMSE, while Vanilla GCN provides better Precision@5, Recall@5, and NDCG@5.

This should currently be treated as an empirical observation rather than a general conclusion. Additional recommendation-specific graph baselines, particularly LightGCN, are required before making a broader claim.

For MAE/RMSE interpretation, note that Vanilla GCN uses legacy per-user MinMax score calibration, while MF produces directly calibrated scores.

## Files and Functions Audited

- `shared/evaluator.py`
  - `EvaluationPolicy`
  - `evaluate_predictions`
- `experiments/aciids_2027/run_mf.py`
  - `fit_predict`
  - publication evaluation path
- `experiments/aciids_2027/run_gcn.py`
  - `fit_predict`
  - publication evaluation path
- `methods/gcn/gcn.py`
  - `minmax_to_rating`
  - `build_prediction_matrix`
- MF and GCN:
  - `config.json`
  - `final_test_results.json`
  - `test_predictions.csv`
  - `test_per_user_metrics.csv`

## Audit Conclusion

Top-K MF-vs-GCN comparison: **VALID**

Rating MAE/RMSE comparison: **VALID as end-to-end prediction pipelines, with calibration caveat**

Next recommended experiment: **LightGCN under the same immutable split and shared evaluator**
