# Validation prediction artifact recovery

## Outcome

Both frozen publication models were reproduced successfully using seed 42, deterministic CUDA, the unchanged seed-42 TRAIN/validation manifests, their existing `fit_predict` implementations, and the shared evaluator. No test manifest or test result was loaded. No hyperparameter search or router analysis was performed.

| Model | MAE | saved MAE | abs. diff | RMSE | saved RMSE | abs. diff | NDCG@5 | saved NDCG@5 | abs. diff | Recall@5 | saved Recall@5 | abs. diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MF | 0.817947893540 | 0.817947893540 | 0 | 0.992840511723 | 0.992840511723 | 1.11e-16 | 0.202879953071 | 0.202879953071 | 0 | 0.257843137255 | 0.257843137255 | 5.55e-17 |
| LightGCN-RMSE | 0.838710524042 | 0.838710526625 | 2.58e-09 | 1.120470175393 | 1.120470180709 | 5.32e-09 | 0.451168581993 | 0.451168581993 | 5.55e-17 | 0.647058823529 | 0.647058823529 | 0 |

All ten aggregate floating-point metrics reproduced within absolute tolerance 1e-07; maximum absolute differences were 1.11e-16 (MF) and 5.32e-09 (LightGCN-RMSE).

## Artifact counts and alignment

- Validation rating rows: 600 for each model.
- Validation users: 458 for each model.
- Ranking-eligible users: 170 for each model.
- Candidate-count range: 121–150 per validation user.
- Candidate rows: 64317 for each model.
- MF and LightGCN candidate sets: **exactly aligned** per user.
- Rating keys, eligible users, relevance labels, TRAIN exclusions, and candidate counts: **exactly aligned**.

## Sufficiency for the router

The recovered validation artifacts are sufficient for the previously specified validation-only sparsity-router experiment: each model now has observed-rating predictions, per-user shared-evaluator metrics, and raw full-catalog candidate scores over identical candidate sets. Threshold selection and test evaluation have deliberately not been started.
