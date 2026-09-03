# RWR publication rerun

## Protocol and calibration

Three fundamentally different legacy transitions are retained as separate baselines. Binary and Linear each searched eight restart probabilities; Softmax searched eight restart probabilities across six temperatures. Selection was per variant on validation RMSE, then validation NDCG@5, then smaller `c` and temperature. Test was loaded only after selections were written and evaluated once per variant.

Rating predictions preserve the legacy per-user `log(raw proximity + 1e-10)` then MinMax-to-[1,5] conversion over TRAIN-unobserved items. No labels fit this mapping, but it is a heuristic candidate-relative calibration and is not equivalent to direct explicit-rating regression. Ranking uses raw RWR proximity.

## Final test results

| Variant | c | Temperature | MAE | RMSE | P@5 | R@5 | NDCG@5 | HR@5 | P@10 | R@10 | NDCG@10 | HR@10 | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RWR_Binary | 0.1 |  | 1.981855 | 2.382367 | 0.179459 | 0.784685 | 0.613796 | 0.832432 | 0.101081 | 0.872072 | 0.642547 | 0.913514 | 3.159 |
| RWR_Linear | 0.1 |  | 1.976459 | 2.373983 | 0.182703 | 0.798198 | 0.623588 | 0.848649 | 0.101622 | 0.877477 | 0.650518 | 0.918919 | 3.121 |
| RWR_Softmax | 0.7 | 0.3 | 1.860281 | 2.249383 | 0.131892 | 0.553153 | 0.438335 | 0.600000 | 0.095676 | 0.819820 | 0.524912 | 0.854054 | 1.782 |

## Answers

1. **Best RWR RMSE:** `RWR_Softmax` (2.249383).
2. **Best RWR NDCG@5:** `RWR_Linear` (0.623588).
3. **Same variant?** No.
4. **Rating predictor or ranking model?** RWR is natively a proximity/ranking model. Its explicit rating output depends on heuristic per-user MinMax calibration.
5. **Compared with MF:** best RWR RMSE 2.249383 versus MF 1.024343; best RWR NDCG@5 0.623588 versus MF 0.193768.
6. **Compared with LightGCN:** best RWR RMSE 2.249383 versus LightGCN 1.167603; best RWR NDCG@5 0.623588 versus LightGCN 0.447355.
7. **Main-table role:** keep RWR in the main comparison table as the non-parametric graph/ranking baseline, while explicitly flagging its heuristic rating calibration. It is highly competitive on ranking but unsuitable as evidence of strong explicit-rating prediction.
8. **Rating caveat:** MAE/RMSE are sensitive to a user-specific candidate-set transformation that is neither learned nor globally calibrated. They are evaluator-compatible but scientifically less direct than MF/GCN explicit-rating predictions.

See `legacy_method_audit.md`, `validation_results.csv`, `fallback_audit.csv`, and `evaluation_audit.md` for full provenance.
