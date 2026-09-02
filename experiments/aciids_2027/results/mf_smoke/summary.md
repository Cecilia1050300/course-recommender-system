# MF benchmark summary

- Mode: smoke
- Seed: 42
- Device: cuda
- Selected hyperparameters: `{'epochs': 4, 'dim': 8, 'lr': 0.01, 'weight_decay': 0.001}`
- Runtime: 4.602 seconds

## Validation grid

| epochs | dim | lr | weight_decay | MAE | RMSE | Rating_Rows | Missing_Predictions | Fallback_Predictions | Cold_User_Rows | Cold_Item_Rows | Ranking_Eligible_Users | Precision@5 | Recall@5 | NDCG@5 | HitRate@5 | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 8 | 0.01 | 0.001 | 1.1512859582901 | 1.3570392290394475 | 600 | 0 | 0 | 0 | 0 | 170 | 0.029411764705882353 | 0.12156862745098038 | 0.08933463663196589 | 0.1411764705882353 | 0.027647058823529413 | 0.21470588235294116 | 0.12108177571207242 | 0.2647058823529412 |
| 4 | 8 | 0.01 | 0.001 | 1.1349054054419199 | 1.3400805741096777 | 600 | 0 | 0 | 0 | 0 | 170 | 0.050588235294117656 | 0.1980392156862745 | 0.13637319158622707 | 0.23529411764705882 | 0.031764705882352945 | 0.24705882352941178 | 0.15366522260947732 | 0.29411764705882354 |

## Final test metrics

| MAE | RMSE | Rating_Rows | Missing_Predictions | Fallback_Predictions | Cold_User_Rows | Cold_Item_Rows | Ranking_Eligible_Users | Precision@5 | Recall@5 | NDCG@5 | HitRate@5 | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1428973209081787 | 1.344602637456203 | 599 | 0 | 0 | 0 | 0 | 185 | 0.04 | 0.16486486486486487 | 0.11637170263360143 | 0.1837837837837838 | 0.02864864864864865 | 0.2342342342342342 | 0.13972804787574952 | 0.2648648648648649 |
