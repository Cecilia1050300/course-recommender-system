# Sparsity-aware MF/LightGCN router V1

## Outcome: NO ELIGIBLE ROUTER

No recommender was retrained and no inference was run. Validation used only the recovered artifacts. Always-MF validation RMSE is 0.992841, so the predeclared 10% ceiling is 1.092125. Every candidate threshold exceeded that ceiling; therefore no threshold was frozen.

| Policy | tau | MAE | RMSE | NDCG@5 | Recall@5 | Users→LightGCN | Users→MF |
|---|---:|---:|---:|---:|---:|---:|---:|
| Always MF |  | 0.817948 | 0.992841 | 0.202880 | 0.257843 | 0 | 458 |
| Always LightGCN |  | 0.838711 | 1.120470 | 0.451169 | 0.647059 | 458 | 0 |
| tau=3 | 3 | 0.848165 | 1.102489 | 0.343053 | 0.457843 | 150 | 308 |
| tau=5 | 5 | 0.838754 | 1.098825 | 0.343053 | 0.457843 | 172 | 286 |
| tau=7 | 7 | 0.836830 | 1.096964 | 0.342387 | 0.463725 | 190 | 268 |
| tau=10 | 10 | 0.838484 | 1.109574 | 0.399657 | 0.528431 | 265 | 193 |
| tau=15 | 15 | 0.840581 | 1.112726 | 0.405539 | 0.534314 | 286 | 172 |
| tau=20 | 20 | 0.840577 | 1.116060 | 0.462006 | 0.643137 | 371 | 87 |

- Validation Pareto-optimal policies: Always MF, tau=5, tau=7, tau=10, tau=15, tau=20.
- Selected threshold: **none**.
- Eligible thresholds: **none**.
- Test artifacts loaded: **false**.
- Test evaluations: **0**.
- Test baseline comparison, sparsity buckets, and oracle diagnostics: **not performed**, because doing so without a frozen eligible router would violate the protocol.

## Required interpretation

1. Pareto-optimal policies are listed above and fully audited in `validation_pareto_analysis.csv`.
2. Threshold selection: **NOT SUPPORTED — NO ELIGIBLE ROUTER**.
3–10. Test routing share, test improvements, bucket effects, and oracle headroom: **INCONCLUSIVE**, because test was not accessed.
11. Simple interaction-count routing as a paper method: **NOT SUPPORTED** under the predeclared validation RMSE constraint.
12. Learned-router justification: **INCONCLUSIVE**; this failed threshold family alone is not evidence that a learned gate will work.

`final_test_results.json`, test CSVs, bucket CSV, and `oracle_upper_bound.json` are explicit status artifacts, not fabricated results.
