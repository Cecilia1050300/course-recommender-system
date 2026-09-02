# Multi-seed robustness summary

Seeds: [42, 123, 2026, 3407, 7777]. Frozen configurations; no hyperparameter search and no validation loading.
GPU: NVIDIA RTX A6000. Total runtime: 26.526 seconds.

## Mean ± sample standard deviation

| Model | MAE | RMSE | Precision@5 | Recall@5 | NDCG@5 | HitRate@5 | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MF | 0.824317 ± 0.026294 | 1.014167 ± 0.027280 | 0.069116 ± 0.005125 | 0.281604 ± 0.024685 | 0.215839 ± 0.018108 | 0.325407 ± 0.029285 | 0.046069 ± 0.002985 | 0.367126 ± 0.023112 | 0.245735 ± 0.018835 | 0.411594 ± 0.026731 |
| LightGCN_RMSE | 0.865423 ± 0.015607 | 1.179393 ± 0.019391 | 0.133076 ± 0.002679 | 0.609828 ± 0.024941 | 0.446610 ± 0.031076 | 0.646280 ± 0.018758 | 0.083444 ± 0.003219 | 0.731913 ± 0.038234 | 0.489186 ± 0.028770 | 0.779579 ± 0.039567 |
| LightGCN_Ranking | 1.044411 ± 0.020132 | 1.371805 ± 0.025063 | 0.173271 ± 0.014722 | 0.760366 ± 0.072491 | 0.602625 ± 0.053855 | 0.802959 ± 0.073189 | 0.104567 ± 0.002152 | 0.893669 ± 0.012277 | 0.649358 ± 0.037145 | 0.926401 ± 0.014103 |
| Vanilla_GCN | 1.595486 ± 0.062816 | 1.953871 ± 0.079990 | 0.094525 ± 0.014610 | 0.441925 ± 0.062565 | 0.265763 ± 0.076750 | 0.470383 ± 0.070685 | 0.061864 ± 0.005833 | 0.561684 ± 0.045978 | 0.305130 ± 0.074031 | 0.609646 ± 0.053689 |

## Pairwise robustness

- MF_lower_RMSE_than_LightGCN_RMSE: 5/5 seeds
- LightGCN_RMSE_higher_NDCG5_than_MF: 5/5 seeds
- LightGCN_Ranking_higher_NDCG5_than_MF: 5/5 seeds
- LightGCN_Ranking_worse_RMSE_than_MF: 5/5 seeds
- LightGCN RMSE-oriented strongest NDCG@5 advantage over MF in a 1–10 bucket: 5/5 seeds
- LightGCN ranking-oriented strongest NDCG@5 advantage over MF in a 1–10 bucket: 5/5 seeds
- Vanilla GCN strongest NDCG@5 advantage over MF in a 1–10 bucket: 5/5 seeds

## A. Directly supported findings

MF has lower RMSE than LightGCN-RMSE in 5/5 seeds. Both LightGCN variants have higher NDCG@5 than MF in 5/5 seeds, while LightGCN-Ranking has worse RMSE than MF in 5/5. The per-seed tables directly support these counts on the five deterministic splits.

## B. Findings robust across most/all seeds

The MF rating advantage and LightGCN ranking advantage are robust in all five seeds. For both LightGCN variants and Vanilla GCN, the strongest NDCG@5 advantage over MF occurs in a 1–10 train-interaction bucket in 5/5 seeds.

## C. Findings that are seed-sensitive

Metric magnitudes vary by seed, especially Vanilla GCN NDCG@5 (sample standard deviation 0.076750). Vanilla GCN beats MF on overall NDCG@5 in 4/5 seeds, failing to do so for seed 3407, so that overall pairwise result is seed-sensitive relative to the unanimous LightGCN results.

## D. Findings not supported

These runs do not support causal claims, population-wide generalization, uncertainty beyond five seeds, or conclusions about alternative hyperparameters. Hyperparameters were frozen and validation was not used.
