# OAF-V2 evaluation audit

Status: **PASS**

- Seed: 42; deterministic execution requested; device `cuda`.
- Scope: TRAIN and validation only. **TEST WAS LOADED: NO**.
- Frozen MF: `experiments/aciids_2027/run_mf.py:fit_predict`, `{'latent_dim': 32, 'epochs': 100, 'learning_rate': 0.005, 'weight_decay': 0.001}`.
- Frozen LightGCN-RMSE: `experiments/aciids_2027/run_lightgcn.py:fit_predict`, `{'embedding_dim': 64, 'number_of_layers': 1, 'epochs': 100, 'learning_rate': 0.01, 'weight_decay': 0.0}`.
- Recovered artifacts: `/home/training/research/course-recommender-system/experiments/aciids_2027/artifacts/validation_predictions/mf` and `/home/training/research/course-recommender-system/experiments/aciids_2027/artifacts/validation_predictions/lightgcn_rmse`.
- Ranking-oriented uniform 3-layer LightGCN was located at `experiments/aciids_2027/results/sparsity_aware_lightgcn_v1/validation_policy_results.csv` (policy `A_uniform`); it is a documented reference only and is not a V2 base branch.
- Base validation reproduction maximum differences: MF `1.11e-16`, LightGCN `7.15e-09` (required <= `1e-07`).
- Recovered candidate alignment differences: MF `8.88e-16`, LightGCN `2.38e-06`; recovered values overwrite replayed validation candidates and are authoritative.
- MF/LightGCN recovered validation artifacts have identical user/item candidate keys, relevance labels, TRAIN exclusions, and ranking-eligible users per `validation_artifact_audit.md`.
- Evaluation uses `shared/evaluator.py:evaluate_predictions`: full 151-course catalog, TRAIN-observed exclusion, relevance rating >=4, binary NDCG, deterministic score-descending then Course-ID ascending order.
- Ordinal supervision contains only strict rating-ordered pairs among TRAIN-observed courses. No validation pair construction, unobserved-as-negative sampling, or test access occurs.
- Raw fusion is used for MAE/RMSE. Per-user z-normalized fusion is used for ordinal loss and ranking metrics only. Moments are fit over each user's TRAIN-unobserved catalog scores and applied to all item scores.
- Preflight unnormalized rating/ranking loss-scale ratios ranged `1.289`–`1.369`, below the predeclared extreme threshold 100; no post-hoc rescaling was applied.
- Global checkpointing monitors validation RMSE with patience 20 and restores the best checkpoint. Validation labels never enter gradient updates.
