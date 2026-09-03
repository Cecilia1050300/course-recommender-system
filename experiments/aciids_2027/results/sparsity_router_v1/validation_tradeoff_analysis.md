# Validation-only RMSE–NDCG trade-off analysis

## Scope

This analysis reads only `validation_router_results.csv` and `validation_pareto_analysis.csv`. No test artifact was loaded, no model was run, and the failed predeclared 10% rule remains unchanged.

## Validation policies

| Policy | Users→LightGCN | %→LightGCN | RMSE | MAE | NDCG@5 | Recall@5 | HitRate@5 | NDCG@10 | Pareto | Dominates MF? | Dominates LightGCN? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| Always MF | 0 | 0.00% | 0.992841 | 0.817948 | 0.202880 | 0.257843 | 0.311765 | 0.242429 | Yes | No | No |
| tau=3 | 150 | 32.75% | 1.102489 | 0.848165 | 0.343053 | 0.457843 | 0.511765 | 0.384782 | No | No | No |
| tau=5 | 172 | 37.55% | 1.098825 | 0.838754 | 0.343053 | 0.457843 | 0.511765 | 0.384782 | Yes | No | No |
| tau=7 | 190 | 41.48% | 1.096964 | 0.836830 | 0.342387 | 0.463725 | 0.517647 | 0.386268 | Yes | No | No |
| tau=10 | 265 | 57.86% | 1.109574 | 0.838484 | 0.399657 | 0.528431 | 0.582353 | 0.443332 | Yes | No | No |
| tau=15 | 286 | 62.45% | 1.112726 | 0.840581 | 0.405539 | 0.534314 | 0.588235 | 0.450915 | Yes | No | No |
| tau=20 | 371 | 81.00% | 1.116060 | 0.840577 | 0.462006 | 0.643137 | 0.694118 | 0.498019 | Yes | No | **Yes** |
| Always LightGCN | 458 | 100.00% | 1.120470 | 0.838711 | 0.451169 | 0.647059 | 0.676471 | 0.496575 | No | No | No |

Dominance uses validation RMSE (lower) and NDCG@5 (higher), requiring at least one strict improvement. Consequently tau=20 dominates Always LightGCN: its RMSE is lower and its NDCG@5 is higher. No router dominates Always MF because every router has higher RMSE.

## Router trade-off quantities

RMSE degradation is relative to Always MF. RMSE recovery is 0 at Always LightGCN and 1 at Always MF. NDCG retention measures the fraction of Always LightGCN's gain over MF; values above 1 are possible when a router exceeds Always LightGCN.

| Router | RMSE degradation vs MF | RMSE recovery vs LightGCN | NDCG@5 gain over MF | LightGCN NDCG gain retained |
|---|---:|---:|---:|---:|
| tau=3 | 11.044% | 0.141 | 0.140173 | 0.565 (56.5%) |
| tau=5 | 10.675% | 0.170 | 0.140173 | 0.565 (56.5%) |
| tau=7 | **10.487%** | **0.184** | 0.139508 | 0.562 (56.2%) |
| tau=10 | 11.758% | 0.085 | 0.196777 | 0.793 (79.3%) |
| tau=15 | 12.075% | 0.061 | 0.202659 | 0.816 (81.6%) |
| tau=20 | 12.411% | 0.035 | 0.259126 | 1.044 (104.4%) |

The recovery measure is not monotonic in tau: replacing MF with LightGCN for additional users can improve or worsen aggregate squared error depending on those users' errors.

## Incremental Pareto trade-off

Policies are ordered by increasing validation RMSE. The slope is `ΔNDCG@5 / ΔRMSE`: ranking gain per additional unit of RMSE.

| Adjacent Pareto policies | ΔRMSE | ΔNDCG@5 | ΔNDCG@5 / ΔRMSE |
|---|---:|---:|---:|
| Always MF → tau=7 | 0.104124 | 0.139508 | 1.340 |
| tau=7 → tau=5 | 0.001861 | 0.000666 | 0.358 |
| tau=5 → tau=10 | 0.010749 | 0.056604 | 5.266 |
| tau=10 → tau=15 | 0.003152 | 0.005882 | 1.866 |
| tau=15 → tau=20 | 0.003334 | 0.056467 | 16.937 |

There is **no clear smooth knee or general diminishing-return region**. Marginal slopes are highly non-monotonic. In particular, tau=7→tau=5 and tau=10→tau=15 yield relatively little NDCG, whereas tau=5→tau=10 and tau=15→tau=20 yield much larger gains. This irregular discrete frontier does not support choosing a policy by visual knee alone.

## Explicit answers

### A. Why did the predeclared 10% constrained V1 fail? — SUPPORTED

The ceiling was `1.10 × 0.992841 = 1.092125`. Every router exceeded it. The closest was tau=7 at RMSE 1.096964, or 10.487% worse than MF.

### B. Was the failure marginal or substantial? — SUPPORTED: marginal

Tau=7 missed the permitted degradation by 0.487 percentage points, corresponding to RMSE 0.004840 above the ceiling. The rule still failed exactly as declared; “marginal” describes distance from the boundary, not permission to relax it retrospectively.

### C. Does any router dominate Always LightGCN? — SUPPORTED

Tau=20 does: RMSE improves from 1.120470 to 1.116060 while NDCG@5 improves from 0.451169 to 0.462006.

### D. Does any router dominate Always MF? — NOT SUPPORTED

No router reaches MF's RMSE, although every router improves NDCG@5.

### E. Is there a clear validation Pareto knee? — NOT SUPPORTED

The discrete marginal slopes fluctuate rather than decline consistently. Any V2 choice therefore needs a separately declared utility, constraint, or scalarization—not post-hoc knee selection.

### F. Is the routing hypothesis still empirically promising? — SUGGESTIVE

Six policies lie on the validation Pareto frontier, and tau=20 strictly dominates Always LightGCN. However, none satisfies the original MF-relative RMSE constraint, and this is one validation split.

### G. Is a separately declared V2 Pareto-selection experiment justified? — SUPPORTED

The validation results justify testing a new, prospectively specified Pareto-based rule because routing supplies distinct non-dominated operating points and one router dominates the graph-only reference. They do not justify selecting tau=20 retrospectively or accessing test under V1. A V2 protocol must declare its selection rule before any test access and preserve V1 as a failed constrained experiment.

## Boundary

This document supports experiment-design justification only. It selects no policy, changes no V1 decision, and provides no test evidence.
