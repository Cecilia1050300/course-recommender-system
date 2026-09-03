# User-level MF versus LightGCN suitability analysis

## Scope and artifact availability

This is a post-hoc, read-only diagnostic. It trained no recommender and ran no inference. Features use only `experiments/aciids_2027/splits/seed_42/train.csv`. Outcomes use the frozen seed-42 `test_per_user_metrics.csv` files for publication MF and RMSE-selected LightGCN.

The frozen ranking-oriented LightGCN configuration has aggregate rows in the robustness outputs but **no saved per-user predictions or per-user metrics**. Consequently `mf_vs_lightgcn_ranking_user_comparison.csv` contains one status row with unavailable metric fields. No inference was rerun to fill it.

## TRAIN-only feature construction

- Users: 460; interactions: 4845; TRAIN catalog: 151 items.
- `rating_std` is population standard deviation (`ddof=0`). Entropy is Shannon entropy in bits over rating values 1–5.
- An observed ordered pair is an unordered pair of two courses from the same user whose ratings differ, oriented from the higher-rated course to the lower-rated course. Equal-rating pairs are excluded.
- Item degree is the number of TRAIN interactions. Head items are the first 31 items (top 20%, rounded up) after sorting by descending TRAIN degree and then item ID; tail items are the last 31 items under that same deterministic ordering. The sets are disjoint. Head minimum degree=73; tail maximum degree=2.
- `fraction_of_catalog_observed` uses the TRAIN catalog denominator only.

## Per-user outcomes: MF versus LightGCN-RMSE

Rating users (n=458): MF 247 (53.9%), LightGCN 211 (46.1%), tie 0 (0.0%).

NDCG@5-eligible users (n=185): MF 27 (14.6%), LightGCN 94 (50.8%), tie 64 (34.6%). Ranking eligibility is exactly non-missing per-user metrics from both shared-evaluator artifacts; ineligible users are excluded.

Recall@5 winners: MF 14 (7.6%), LightGCN 79 (42.7%), tie 92 (49.7%). HitRate@5 winners: MF 12 (6.5%), LightGCN 77 (41.6%), tie 96 (51.9%).

Ties are retained descriptively and excluded from binary prediction. Rating differences are `LightGCN MAE − MF MAE` (positive favors MF). Ranking differences are `LightGCN NDCG@5 − MF NDCG@5` (positive favors LightGCN).

## Feature associations

`feature_analysis.csv` reports winner-group mean/median/sample SD, Cohen's d, Mann–Whitney U, and Spearman association with the continuous performance difference. Cohen's d is LightGCN-winner minus MF-winner, so its sign describes which winner group has the larger feature. Benjamini–Hochberg correction is applied separately within the rating and ranking analyses to the 28 Mann–Whitney tests and separately to the 28 Spearman tests.

The five largest absolute Spearman associations with LightGCN's NDCG@5 advantage are:

- `train_interaction_count`: Spearman ρ=-0.499, BH-adjusted p=8.133e-08
- `fraction_of_catalog_observed`: Spearman ρ=-0.499, BH-adjusted p=8.133e-08
- `number_of_observed_ordered_pairs`: Spearman ρ=-0.491, BH-adjusted p=1.003e-07
- `median_degree_of_interacted_items`: Spearman ρ=0.480, BH-adjusted p=1.839e-07
- `normalized_pair_count_per_interaction`: Spearman ρ=-0.474, BH-adjusted p=2.301e-07

Interaction count alone has ρ=-0.499 with NDCG@5 difference. Association is descriptive, not causal.

## Predictive feasibility

Binary diagnostic sample: n=121 non-tied eligible users (94 LightGCN winners, 27 MF winners); stratified 5-fold CV, shuffled with seed 42. Each metric below comes from pooled out-of-fold predictions. The positive class is LightGCN winner.

| Model | Accuracy | Balanced accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| majority_class | 0.777 | 0.500 | 0.500 | 0.777 | 1.000 | 0.874 |
| interaction_count_only | 0.884 | 0.807 | 0.882 | 0.908 | 0.947 | 0.927 |
| logistic_all_features | 0.901 | 0.817 | 0.855 | 0.910 | 0.968 | 0.938 |
| shallow_decision_tree | 0.777 | 0.764 | 0.801 | 0.914 | 0.787 | 0.846 |

The logistic coefficients are standardized; tree importance is Gini importance. Both are recorded in `suitability_prediction_results.csv` and fitted on the full diagnostic sample only for interpretation, not for the CV performance estimates.

## Leakage audit

- **Verified:** no explanatory feature uses validation/test ratings; all 28 features are computed from TRAIN rows only.
- **Verified:** no explanatory feature contains MF/LightGCN scores or predictions.
- **Verified:** no feature is constructed from ranking outcomes or evaluator metrics.
- Saved test outcomes are used only to define the post-hoc supervised winner label. This analyzes explainability/predictability; it does not select or tune a recommender.
- The simple classifier hyperparameters are fixed analytical defaults; no test-label-driven hyperparameter search was conducted.

## Multi-seed limitation

Multi-seed user-level analysis is **unavailable**. The robustness directory contains aggregate and sparsity-bucket results but 0 per-user prediction/metric files. Per the no-inference constraint, seed-level winner fractions and top-five user-feature associations were not reconstructed.

## Explicit answers

### A. Are there distinct user groups where MF and LightGCN are preferable? — SUPPORTED

Both MF and LightGCN win for nonzero subsets of eligible users. The group distribution and adjusted feature tests quantify whether these groups are distinguishable; this is heterogeneity evidence, not proof that a deployable gate will generalize.

### B. Is simple interaction sparsity alone enough to predict the winner? — SUPPORTED

Interaction-count-only balanced accuracy is 0.807, versus 0.500 for the majority baseline.

### C. Do richer TRAIN-only user features improve winner prediction? — NOT SUPPORTED

All-feature logistic balanced accuracy is 0.817, compared with 0.807 for count only. Performance is cross-validated within one seed's user set.

### D. Which features are most strongly associated with LightGCN ranking advantage? — SUGGESTIVE

The five associations listed above are the strongest in this dataset. BH-adjusted p-values and effect directions should be used together; correlated features and a single split prevent causal interpretation.

### E. Can we predict the ranking winner substantially above majority? — SUPPORTED

The best balanced-accuracy gain over majority is 0.317. Accuracy alone is not used for this verdict because winner classes may be imbalanced.

### F. Is there enough signal for a future user-specific objective/gate? — SUGGESTIVE

This verdict requires both meaningful out-of-fold improvement and usable absolute discrimination. A future gate is not justified as a publication conclusion without per-user multi-seed replication; ranking-oriented LightGCN also lacks a saved user-level artifact.

## Interpretation limits

This is a single-split, post-hoc association study. Users are the CV units but share an interaction graph, so folds are not fully independent in a graph-statistical sense. Test-derived labels are legitimate here only as diagnostic outcomes. Results must not be presented as prospective model-selection performance, causal effects, or evidence for the unavailable ranking-oriented comparison.
