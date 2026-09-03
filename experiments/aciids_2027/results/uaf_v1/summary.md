# UAF-V1 validation-only results

**Verdict: NOT SUPPORTED. TEST WAS LOADED: NO.**

The gate alone was trained; MF and LightGCN parameters remained frozen. Because recovered artifacts omit TRAIN-observed scores, the authoritative frozen implementations were replayed once and reproduced within `1e-7` before gate development. Rating fusion uses raw scores; ranking fusion uses per-user candidate z-scores.

## Best validation points

| Method | MAE | RMSE | NDCG@5 | Recall@5 | Eligible |
|---|---:|---:|---:|---:|:---:|
| Always-MF | 0.817948 | 0.992841 | 0.202880 | 0.257843 | Yes |
| Always-LightGCN | 0.838711 | 1.120470 | 0.451169 | 0.647059 | No |
| Fixed-alpha-0.25 | 0.808992 | 1.062769 | 0.430567 | 0.578431 | Yes |
| UAF-count | 0.796926 | 1.005136 | 0.347651 | 0.441176 | Yes |
| UAF-rich | 0.803135 | 1.018581 | 0.350683 | 0.463725 | Yes |

The prior hard router has no selected B2 point because its predeclared 10% rule produced `NO ELIGIBLE ROUTER`; all six historical threshold points remain in `pareto_validation.csv` for comparison. Its maximum validation NDCG@5 was 0.462006.

## Gate diagnostics

| Gate | Mean alpha | Std | Min | Max | Spearman(alpha,count) |
|---|---:|---:|---:|---:|---:|
| UAF-count | 0.632422 | 0.082554 | 0.533007 | 0.773461 | 1.000000 |
| UAF-rich | 0.646324 | 0.164202 | 0.389618 | 0.961687 | 0.945943 |

| Gate | Bucket | Users | Mean | Std | Min | Max | Spearman(alpha,count) | alpha<.25 | .25<=alpha<.5 | .5<=alpha<.75 | alpha>=.75 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UAF-count | ALL | 460 | 0.632422 | 0.082554 | 0.533007 | 0.773461 | 1.000000 | 0.00% | 0.00% | 93.91% | 6.09% |
| UAF-count | 1-5 | 174 | 0.540035 | 0.014379 | 0.533007 | 0.578170 |  | 0.00% | 0.00% | 100.00% | 0.00% |
| UAF-count | 6-10 | 93 | 0.626697 | 0.017028 | 0.591485 | 0.641861 |  | 0.00% | 0.00% | 100.00% | 0.00% |
| UAF-count | 11-20 | 106 | 0.695946 | 0.016712 | 0.652968 | 0.725146 |  | 0.00% | 0.00% | 100.00% | 0.00% |
| UAF-count | >20 | 87 | 0.745917 | 0.008154 | 0.731241 | 0.773461 |  | 0.00% | 0.00% | 67.82% | 32.18% |
| UAF-rich | ALL | 460 | 0.646324 | 0.164202 | 0.389618 | 0.961687 | 0.945943 | 0.00% | 27.17% | 41.96% | 30.87% |
| UAF-rich | 1-5 | 174 | 0.483004 | 0.062491 | 0.389618 | 0.692108 |  | 0.00% | 71.84% | 28.16% | 0.00% |
| UAF-rich | 6-10 | 93 | 0.617907 | 0.050131 | 0.509523 | 0.802032 |  | 0.00% | 0.00% | 96.77% | 3.23% |
| UAF-rich | 11-20 | 106 | 0.749083 | 0.081173 | 0.623430 | 0.916762 |  | 0.00% | 0.00% | 50.00% | 50.00% |
| UAF-rich | >20 | 87 | 0.878138 | 0.056749 | 0.741320 | 0.961687 |  | 0.00% | 0.00% | 1.15% | 98.85% |

The Spearman coefficient is an all-user statistic and is therefore shown only on each gate's `ALL` row. Individual selected-gate values are saved in `user_gate_values.csv`.

## Required answers

1. **RMSE constraint:** At least one learned UAF satisfied it.
2. **NDCG@5 over Always-MF:** Yes.
3. **Over previous hard router:** No; the best historical threshold remains higher.
4. **Pareto expansion:** Yes; a learned UAF point is Pareto-optimal in the combined audit.
5. **Rich versus count-only:** Rich has slightly higher NDCG@5 but does not unambiguously outperform count-only because its RMSE is worse. Their RMSE values are 1.018581 and 1.005136, respectively.
6. **Alpha and sparsity:** UAF-count Spearman rho is 1.000; UAF-rich rho is 0.946. These are validation-development associations, not causal effects.
7. **Overall:** **NOT SUPPORTED**, based exactly on the predeclared constraint and comparison rules.

## Final required printout

- Always-MF validation: MAE 0.817948, RMSE 0.992841, P@5 0.065882, R@5 0.257843, NDCG@5 0.202880, HR@5 0.311765, P@10 0.047647, R@10 0.374510, NDCG@10 0.242429, HR@10 0.435294.
- Always-LightGCN validation: MAE 0.838711, RMSE 1.120470, P@5 0.141176, R@5 0.647059, NDCG@5 0.451169, HR@5 0.676471, P@10 0.088824, R@10 0.779412, NDCG@10 0.496575, HR@10 0.835294.
- Best fixed fusion: Fixed-alpha-0.25, MAE 0.808992, RMSE 1.062769, P@5 0.131765, R@5 0.578431, NDCG@5 0.430567, HR@5 0.623529, P@10 0.088824, R@10 0.761765, NDCG@10 0.493567, HR@10 0.811765.
- Best UAF-count: MAE 0.796926, RMSE 1.005136, P@5 0.104706, R@5 0.441176, NDCG@5 0.347651, HR@5 0.494118, P@10 0.072353, R@10 0.598039, NDCG@10 0.401363, HR@10 0.647059.
- Best UAF-rich: MAE 0.803135, RMSE 1.018581, P@5 0.109412, R@5 0.463725, NDCG@5 0.350683, HR@5 0.523529, P@10 0.069412, R@10 0.582353, NDCG@10 0.391323, HR@10 0.635294.
- Eligibility ceiling: 1.092125.
- Selected method: Fixed-alpha-0.25.
- TEST WAS LOADED: **NO**.
