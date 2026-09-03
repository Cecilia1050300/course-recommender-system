# UAF-V1 evaluation audit

Status: **PASS**

- Scope: validation-only; **TEST WAS LOADED: NO**.
- TRAIN: `/home/training/research/course-recommender-system/experiments/aciids_2027/splits/seed_42/train.csv`; validation: `/home/training/research/course-recommender-system/experiments/aciids_2027/splits/seed_42/validation.csv`.
- Frozen MF implementation: `experiments/aciids_2027/run_mf.py:fit_predict` with `{'latent_dim': 32, 'epochs': 100, 'learning_rate': 0.005, 'weight_decay': 0.001}`.
- Frozen LightGCN implementation: `experiments/aciids_2027/run_lightgcn.py:fit_predict` with `{'embedding_dim': 64, 'number_of_layers': 1, 'epochs': 100, 'learning_rate': 0.01, 'weight_decay': 0.0}`.
- Recovered validation artifacts: `/home/training/research/course-recommender-system/experiments/aciids_2027/artifacts/validation_predictions/mf` and `/home/training/research/course-recommender-system/experiments/aciids_2027/artifacts/validation_predictions/lightgcn_rmse`.
- Previous router source: `/home/training/research/course-recommender-system/experiments/aciids_2027/results/sparsity_router_v1/validation_router_results.csv`.
- Shared evaluation: `shared/evaluator.py:evaluate_predictions`, full 151-item catalog, TRAIN-item exclusion, relevance `rating >= 4`, binary NDCG, identical eligible-user and tie-breaking semantics.
- Reproduction maximum differences: MF `1.11e-16`; LightGCN `4.16e-09` (tolerance `1e-07`).
- Recovered candidate-score maximum differences: MF `8.88e-16`; LightGCN `9.54e-07`.
- Features and their standardization statistics use TRAIN only. Gate gradients use TRAIN observed ratings only. Validation labels are used solely for early stopping, hyperparameter selection, and method evaluation.
- Rating fusion uses raw model predictions followed by shared `[1,5]` clipping. Ranking fusion uses per-user, per-model z-scores over TRAIN-unobserved full-catalog candidates; normalized scores never enter MAE/RMSE.
