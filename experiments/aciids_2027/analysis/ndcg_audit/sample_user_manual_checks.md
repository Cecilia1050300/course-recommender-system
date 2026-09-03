# Sample-user manual checks

Status: **PARTIALLY BLOCKED BY FROZEN-ARTIFACT PROVENANCE**

The publication artifacts retain held-out scores and aggregate per-user full-catalog metrics, but not MF or LightGCN full-catalog candidate scores/checkpoints. Therefore top-10 item/score lists and independent DCG reconstruction cannot be truthfully recovered without retraining, which this audit forbids.

| Role | User | Train interactions | Relevant test items | Candidates | MF evaluator NDCG@5 | LightGCN evaluator NDCG@5 |
|---|---|---:|---|---:|---:|---:|
| sparse | 40211120 | 1 | 40BA011 | 150 | 0.000000000000 | 1.000000000000 |
| medium | 40311135 | 16 | 40BA081 | 135 | 0.000000000000 | 1.000000000000 |
| dense | 40211126 | 30 | 40BA081 | 121 | 0.000000000000 | 0.000000000000 |
| MF-good | 40511146 | 1 | 40B0291 | 150 | 1.000000000000 | 1.000000000000 |
| LightGCN-good | 40511253 | 1 | 40BA012 | 150 | 0.000000000000 | 1.000000000000 |

## Independent arithmetic checks

For relevance `[1,0,1,0,0]`, DCG@5 = `1 + 1/log2(4) = 1.5`; IDCG@5 = `1 + 1/log2(3) = 1.630929753571`; NDCG@5 = `0.919720789149`.
For relevance `[0,1,0,0,0]`, DCG@5 = `1/log2(3) = 0.630929753571`; IDCG@5 = `1`; NDCG@5 = `0.630929753571`. These match the evaluator formula to floating-point tolerance.
