# CF six-way publication evaluation audit

Status: **PASS**

| Check | Verification |
|---|---|
| Immutable split | All methods use `experiments/aciids_2027/splits/seed_42`; TRAIN 4845, validation 600, test 599 rows. |
| Identical validation rating keys | True; one shared manifest with 600 unique keys. |
| Identical test rating keys | True; one shared manifest with 599 unique keys. |
| Identical candidate sets | True; candidate construction is evaluator-owned and model-independent. Validation signature `12153af0b9a82af12cb51c7a90fc491f93d0ec5872f4fbf2f74c73f301e1bbd1`; test signature `12153af0b9a82af12cb51c7a90fc491f93d0ec5872f4fbf2f74c73f301e1bbd1`. |
| Candidate counts | Validation 64317 rows (121–150 per user); test 64317 rows (121–150 per user). |
| Identical relevance labels | True; shared validation/test manifests and threshold `rating >= 4.0`. |
| TRAIN-observed exclusions | True; shared evaluator excludes each user's TRAIN-observed items. |
| Deterministic fallback | True; item TRAIN mean → user TRAIN mean → global TRAIN mean. No hash jitter. |
| Ranking tie-breaking | Raw score descending, Course ID ascending, in `shared/evaluator.py`. |
| Similarity/scaling leakage | None; rating matrices, scalers, similarities, and fallback statistics use TRAIN only. |
| K selection | Validation RMSE only, then validation NDCG@5, then smaller K. `selected_configs.json` was written before test load. |
| Test use | One evaluation per method after its K was frozen. |
| Legacy evaluator behavior removed | No artificial error=5, no [0,5] model clipping, and no held-out-list legacy NDCG. |

All six methods use the same 151-item catalog and the same shared `EvaluationPolicy`.
