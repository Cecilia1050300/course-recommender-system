# Legacy RWR method audit

Source audited: `methods/rwr/RWR_experiment_ndcg.py`. This audit was completed before the publication rerun.

## Exact variants

| Publication identifier | Legacy transition key | Edge transformation | Row normalization | Tunable parameters |
|---|---|---|---|---|
| `RWR_Binary` | `pure_binary` | Every positive user–course rating edge becomes 1 | Divide by row sum | restart probability `c` |
| `RWR_Linear` | `pure_linear` | Preserve the positive rating value as the edge weight | Divide by row sum | restart probability `c` |
| `RWR_Softmax` | `softmax` | For each node, apply masked softmax to positive incident rating weights divided by temperature | Softmax itself sums each nonempty row to 1 | restart probability `c`; temperature |

The graph is an undirected weighted bipartite block adjacency matrix
`A = [[0, R], [R.T, 0]]`; there are no self-loops, content edges, embeddings, or trainable parameters. Isolated rows remain all zero.

## RWR equation and grid

For a target user restart vector `E`, initialized with `P=E`, the legacy iteration is:

`P_new = (1-c) W.T P + c E`

It stops at Euclidean change `< 1e-8` or 200 iterations. The exact restart grid is:

`c = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]`.

Softmax additionally searches:

`temperature = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]`.

Thus Binary and Linear have eight configurations each; Softmax has 48, for 64 legitimate configurations total.

## Legacy rating conversion

For each user, the legacy code takes the item-node proximity scores, applies `log(score + 1e-10)`, and performs a separate MinMax transformation over that user's unobserved courses:

`predicted_rating = 1 + 4 * (log_score - user_unobserved_min) / (user_unobserved_max - user_unobserved_min)`.

If all unobserved values are equal, it assigns 3.0. Observed TRAIN items remain zero in the legacy prediction vector.

This mapping requires no validation/test labels and is directly reproducible from a TRAIN-only graph plus the user's TRAIN-observed mask. It is nevertheless a **heuristic, per-user candidate-set calibration**, not a learned or probabilistically justified mapping from proximity to explicit ratings. It makes MAE/RMSE scientifically weaker than for MF/GCN models whose raw outputs directly optimize ratings. For the publication rerun, this exact mapping is retained for rating predictions, while raw RWR proximity is used for ranking; logarithm and MinMax are monotonic for non-degenerate candidates but are not needed for ranking.

## Legacy fallback and evaluator behavior

- Unknown user/item: prediction 0 and artificial absolute error 5.
- Degenerate per-user unobserved score range: prediction 3.
- Legacy NDCG ranks only held-out rows for users with at least two held-out rows and uses graded gain `2^rating-1`.

The publication rerun removes the artificial error-5 evaluator rule and unknown-key behavior is delegated to the shared evaluator. Warm-start immutable splits make unknown users/items absent. The source-defined degenerate mapping to 3 remains part of the rating conversion and is reported as a fallback/calibration event.
