# Sparsity-bucket model comparison

Users are bucketed by observed train interactions. Rating metrics are micro-averaged over saved test prediction rows. Ranking metrics are macro-averaged over users with at least one relevant test item, matching the publication evaluator. This analysis uses only existing train counts, test predictions, and per-user metric artifacts; no inference or new test evaluation was performed.

| Model | Train interactions | Users | Rows | Eligible | MAE | RMSE | P@5 | R@5 | NDCG@5 | HR@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MF | 1–5 | 172 | 172 | 58 | 1.069318 | 1.227770 | 0.027586 | 0.137931 | 0.103857 | 0.137931 |
| MF | 6–10 | 93 | 93 | 25 | 0.766460 | 0.928054 | 0.056000 | 0.280000 | 0.187155 | 0.280000 |
| MF | 11–20 | 106 | 139 | 44 | 0.763678 | 0.975616 | 0.072727 | 0.352273 | 0.251474 | 0.363636 |
| MF | >20 | 87 | 195 | 58 | 0.694244 | 0.896829 | 0.093103 | 0.307471 | 0.242753 | 0.431034 |
| Vanilla GCN | 1–5 | 172 | 172 | 58 | 1.815243 | 2.185295 | 0.186207 | 0.931034 | 0.485772 | 0.931034 |
| Vanilla GCN | 6–10 | 93 | 93 | 25 | 1.570388 | 1.888371 | 0.104000 | 0.520000 | 0.232558 | 0.520000 |
| Vanilla GCN | 11–20 | 106 | 139 | 44 | 1.156174 | 1.425802 | 0.054545 | 0.261364 | 0.147801 | 0.272727 |
| Vanilla GCN | >20 | 87 | 195 | 58 | 1.399563 | 1.703338 | 0.044828 | 0.149425 | 0.081305 | 0.206897 |
| LightGCN | 1–5 | 172 | 172 | 58 | 1.144268 | 1.538808 | 0.168966 | 0.844828 | 0.603236 | 0.844828 |
| LightGCN | 6–10 | 93 | 93 | 25 | 0.819593 | 1.057491 | 0.168000 | 0.840000 | 0.600640 | 0.840000 |
| LightGCN | 11–20 | 106 | 139 | 44 | 0.805656 | 1.023766 | 0.168182 | 0.795455 | 0.538911 | 0.795455 |
| LightGCN | >20 | 87 | 195 | 58 | 0.697126 | 0.904816 | 0.058621 | 0.206897 | 0.155948 | 0.275862 |

## Answers

**A. Does greater LightGCN propagation depth appear to trade rating accuracy for ranking quality in the controlled slice?** Yes within that validation slice: layers 1→3 worsen MAE and RMSE while improving Recall@5 and NDCG@5 at each step. This is a within-grid controlled comparison, not a full causal experiment.

**B. Is LightGCN's ranking advantage over MF concentrated among sparse users, dense users, or broadly distributed?** It is concentrated among sparse-to-moderate users in these artifacts. The NDCG@5 advantage is +0.499 for 1–5, +0.413 for 6–10, and +0.287 for 11–20 interactions. For >20 interactions, LightGCN is lower than MF by 0.087. Recall@5 has the same direction in every bucket.

**C. Does Vanilla GCN show the same sparsity pattern as LightGCN?** Broadly yes, but more sharply. Vanilla GCN has its strongest ranking performance for 1–5 interactions, a smaller advantage over MF at 6–10, and falls below MF for 11–20 and >20. Unlike LightGCN, its NDCG@5 declines monotonically as train interactions increase.

**D. Strength of findings**

- **Directly supported:** the reported within-artifact bucket counts and metrics; the reversal of LightGCN versus MF ranking performance in the >20 bucket; and the broadly sparse-user-concentrated ranking pattern for both graph models on this split.
- **Suggestive only:** that graph propagation is responsible for the sparse-user pattern, or that these bucket differences will reproduce across seeds, splits, datasets, and alternative selection objectives.
- **Not supported:** causal claims, statistical significance, universal cold/sparse-user superiority, or conclusions about users absent from train. There are no train-cold users in this benchmark.
