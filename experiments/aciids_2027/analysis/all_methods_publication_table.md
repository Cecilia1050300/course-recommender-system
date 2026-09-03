# ACIIDS 2027 all-method publication benchmark

All values come from structured, saved test artifacts. The table displays six decimal places; `all_methods_publication_table.csv` preserves the stored numeric precision.

## Complete seed-42 test table

| Family | Method | Selected Config | MAE ↓ | RMSE ↓ | P@5 ↑ | R@5 ↑ | NDCG@5 ↑ | HR@5 ↑ | P@10 ↑ | R@10 ↑ | NDCG@10 ↑ | HR@10 ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CF | User_Rating | K=9 | 0.979477 | 1.233450 | 0.002162 | 0.008108 | 0.003985 | 0.010811 | 0.001622 | 0.010811 | 0.005031 | 0.016216 |
| CF | User_Content | K=9 | 1.081693 | 1.304234 | 0.001081 | 0.002703 | 0.001427 | 0.005405 | 0.001081 | 0.005405 | 0.002385 | 0.010811 |
| CF | User_Hybrid | K=9 | 0.983697 | 1.236578 | 0.001081 | 0.002703 | 0.001427 | 0.005405 | 0.001622 | 0.008108 | 0.003431 | 0.016216 |
| CF | Item_Rating | K=9 | 0.905183 | 1.186338 | 0.005405 | 0.024324 | 0.012522 | 0.027027 | 0.002703 | 0.024324 | 0.012522 | 0.027027 |
| CF | Item_Content | K=9 | 0.940430 | 1.211957 | 0.004324 | 0.018919 | 0.010569 | 0.021622 | 0.004324 | 0.040541 | 0.017389 | 0.043243 |
| CF | Item_Hybrid | K=7 | 0.910807 | 1.193898 | 0.051892 | 0.202703 | 0.160392 | 0.232432 | 0.032432 | 0.252252 | 0.177006 | 0.281081 |
| RWR | Binary | c=0.1 | 1.981855 | 2.382367 | 0.179459 | 0.784685 | 0.613796 | 0.832432 | 0.101081 | 0.872072 | 0.642547 | 0.913514 |
| RWR | Linear | c=0.1 | 1.976459 | 2.373983 | 0.182703 | 0.798198 | 0.623588 | 0.848649 | 0.101622 | 0.877477 | 0.650518 | 0.918919 |
| RWR | Softmax | c=0.7; T=0.3 | 1.860281 | 2.249383 | 0.131892 | 0.553153 | 0.438335 | 0.600000 | 0.095676 | 0.819820 | 0.524912 | 0.854054 |
| Other | MF | d=32; epochs=100; lr=0.005; wd=0.001 | 0.829269 | 1.024343 | 0.062703 | 0.261261 | 0.193768 | 0.302703 | 0.042162 | 0.340541 | 0.221597 | 0.372973 |
| Other | Vanilla GCN | hidden=12; d=16; epochs=20; lr=0.005; wd=0.0001; dropout=0.1 | 1.488966 | 1.827887 | 0.099459 | 0.471171 | 0.244365 | 0.491892 | 0.061622 | 0.567568 | 0.274668 | 0.605405 |
| Other | LightGCN-RMSE | d=64; L=1; epochs=100; lr=0.01; wd=0 | 0.869719 | 1.167603 | 0.134054 | 0.632432 | 0.447355 | 0.654054 | 0.085405 | 0.772973 | 0.494881 | 0.821622 |
| Other | LightGCN-Ranking / Uniform-3L | A_uniform; d=32; L=3; epochs=100; lr=0.01; wd=0 | 1.034283 | 1.361298 | 0.191351 | 0.844144 | 0.673036 | 0.886486 | 0.105405 | 0.911712 | 0.696995 | 0.945946 |

## LightGCN-Ranking artifact identity

The authoritative complete seed-42 row above is from `experiments/aciids_2027/results/sparsity_aware_lightgcn_v1/final_test_results.json`. Its selected policy is `A_uniform`, with equal `[0.25, 0.25, 0.25, 0.25]` weights for every user bucket and fixed `d=32, L=3, lr=0.01, epochs=100, wd=0`. It is therefore mathematically the standard uniform three-layer LightGCN and reproduces the frozen seed-42 `LightGCN_Ranking` row in `robustness/per_seed_results.csv` (differences in MAE/RMSE are below `8e-9`; ranking metrics are identical).

Thus this row is both the V1 validation-selected uniform policy and the same model/configuration as the ranking-oriented 3-layer baseline. It is not the original RMSE-selected publication LightGCN, which is the separate `LightGCN-RMSE` row (`d=64, L=1`). No five-seed mean is included here.

## Consistency audit

Every included row is directly comparable: all use the immutable publication seed-42 split (TRAIN 4,845; validation 600; test 599), the 151-course full catalog, exclusion of user TRAIN-observed items, binary relevance at rating `>=4`, shared-evaluator ranking eligibility, score-descending/Course-ID-ascending tie-breaking, and the same `@5/@10` metric definitions. All rows have 599 rating rows and 185 ranking-eligible test users. CF and RWR audits additionally record the identical 64,317-row candidate manifest (121–150 candidates/user). No row is flagged non-comparable.

Rating-score semantics nevertheless differ by model: RWR maps proximity through a heuristic per-user log/MinMax transformation for MAE/RMSE, while ranking uses raw proximity. This is a scientific interpretation caveat, not an evaluator-protocol mismatch.

## Rankings

### A. RMSE (lower is better)

1. MF — 1.024343
2. LightGCN-RMSE — 1.167603
3. Item_Rating — 1.186338
4. Item_Hybrid — 1.193898
5. Item_Content — 1.211957
6. User_Rating — 1.233450
7. User_Hybrid — 1.236578
8. User_Content — 1.304234
9. LightGCN-Ranking / Uniform-3L — 1.361298
10. Vanilla GCN — 1.827887
11. Softmax — 2.249383
12. Linear — 2.373983
13. Binary — 2.382367

### B. NDCG@5 (higher is better)

1. LightGCN-Ranking / Uniform-3L — 0.673036
2. Linear — 0.623588
3. Binary — 0.613796
4. LightGCN-RMSE — 0.447355
5. Softmax — 0.438335
6. Vanilla GCN — 0.244365
7. MF — 0.193768
8. Item_Hybrid — 0.160392
9. Item_Rating — 0.012522
10. Item_Content — 0.010569
11. User_Rating — 0.003985
12. User_Content — 0.001427
13. User_Hybrid — 0.001427

### C. NDCG@10 (higher is better)

1. LightGCN-Ranking / Uniform-3L — 0.696995
2. Linear — 0.650518
3. Binary — 0.642547
4. Softmax — 0.524912
5. LightGCN-RMSE — 0.494881
6. Vanilla GCN — 0.274668
7. MF — 0.221597
8. Item_Hybrid — 0.177006
9. Item_Content — 0.017389
10. Item_Rating — 0.012522
11. User_Rating — 0.005031
12. User_Hybrid — 0.003431
13. User_Content — 0.002385

### D. Recall@5 (higher is better)

1. LightGCN-Ranking / Uniform-3L — 0.844144
2. Linear — 0.798198
3. Binary — 0.784685
4. LightGCN-RMSE — 0.632432
5. Softmax — 0.553153
6. Vanilla GCN — 0.471171
7. MF — 0.261261
8. Item_Hybrid — 0.202703
9. Item_Rating — 0.024324
10. Item_Content — 0.018919
11. User_Rating — 0.008108
12. User_Content — 0.002703
13. User_Hybrid — 0.002703

## Winners

- Overall RMSE: **MF** (1.024343).
- Overall NDCG@5: **LightGCN-Ranking / Uniform-3L** (0.673036).
- CF RMSE: **Item_Rating** (1.186338).
- CF NDCG@5: **Item_Hybrid** (0.160392).
- RWR RMSE: **Softmax** (2.249383).
- RWR NDCG@5: **Linear** (0.623588).

## Structured sources

- CF: `results/cf_6ways_publication/final_test_results.csv` and `selected_configs.json`.
- RWR: `results/rwr_publication/final_test_results.csv` and `selected_configs.json`.
- MF: `results/mf_publication/final_test_results.json` and `selected_config.json`.
- Vanilla GCN: `results/gcn_publication/final_test_results.json` and `selected_config.json`.
- LightGCN-RMSE: `results/lightgcn_publication/final_test_results.json` and `selected_config.json`.
- LightGCN-Ranking / Uniform-3L: `results/sparsity_aware_lightgcn_v1/final_test_results.json` and `selected_policy.json`; seed-42 equivalence cross-checked against `robustness/per_seed_results.csv`.
