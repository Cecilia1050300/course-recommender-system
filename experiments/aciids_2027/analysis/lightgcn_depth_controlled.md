# Controlled LightGCN propagation-depth analysis

Source: `validation_results.csv` only. Fixed hyperparameters: embedding dimension 32, learning rate 0.01, 100 epochs, and weight decay 0. No model was retrained and no test artifact was used.

| Layers | MAE ↓ | RMSE ↓ | P@5 | P@10 | R@5 ↑ | R@10 | NDCG@5 ↑ | NDCG@10 | HR@5 | HR@10 | Training loss | Runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.866517 | 1.171333 | 0.137647 | 0.090588 | 0.602941 | 0.761765 | 0.461803 | 0.515982 | 0.641176 | 0.811765 | 0.669175 | 1.058609 |
| 2 | 0.898885 | 1.180469 | 0.175294 | 0.104118 | 0.759804 | 0.864706 | 0.592096 | 0.629792 | 0.805882 | 0.900000 | 0.917511 | 1.112832 |
| 3 | 0.990223 | 1.289821 | 0.197647 | 0.107647 | 0.833333 | 0.889216 | 0.648351 | 0.668304 | 0.876471 | 0.911765 | 1.138293 | 1.256148 |

## Relative changes

Positive changes in MAE/RMSE are deterioration because lower is better. Positive changes in Recall/NDCG are improvement because higher is better.

| Comparison | MAE | RMSE | Recall@5 | NDCG@5 |
|---|---:|---:|---:|---:|
| Layers 1 → 2 | +3.735% (worse) | +0.780% (worse) | +26.016% (better) | +28.214% (better) |
| Layers 2 → 3 | +10.161% (worse) | +9.263% (worse) | +9.677% (better) | +9.501% (better) |
| Layers 1 → 3 | +14.276% (worse) | +10.116% (worse) | +38.211% (better) | +40.396% (better) |

## Interpretation

**A. Does greater propagation depth appear to trade rating accuracy for ranking quality?** Yes, within this fixed hyperparameter slice. Every increase in depth worsens both lower-is-better rating metrics and improves both higher-is-better ranking metrics examined.

This is a within-grid controlled comparison, not a full causal experiment. It holds four recorded hyperparameters fixed, but uses one split and seed and does not establish that depth alone will have the same effect under other settings or datasets.
