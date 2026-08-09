# Course Recommender System

This repository contains experiments for a course recommendation system. The project is organized by method so each experiment has its own folder, while shared datasets stay at the project root.

## Folder Layout

```text
.
|-- methods/
|   |-- similarity_6ways/        # 6 similarity-based recommendation methods
|   |   `-- results/             # CSV outputs for the similarity experiments
|   |-- matrix_factorization/    # MF / SVD / SGD experiments
|   |   `-- results/             # MF reports and plots
|   |-- rwr/                     # Random Walk with Restart experiments
|   `-- gcn/                     # GCN recommendation experiments
|-- shared/                      # preprocessing, plotting, and cross-method utilities
|   `-- plots/                   # shared dataset visualization outputs
|-- old/                         # older exploratory files kept for reference
|-- vq_vae/                      # unrelated/experimental model files kept separately
|-- rating_matrix.csv            # shared rating matrix
|-- rating_matrix_train.csv      # shared train matrix
|-- rating_matrix_truth.csv      # shared truth matrix
|-- test_set.csv                 # shared evaluation set
|-- transaction.csv              # shared raw transaction data
|-- flattened.csv                # shared user/content feature table
`-- item_content_100d.csv        # shared course/content embeddings
```

## Method Groups

### 1. Similarity 6 Ways

Location: `methods/similarity_6ways/`

Includes user/item and rating/content/hybrid similarity experiments:

- `User_Rating`
- `User_Content`
- `User_Hybrid`
- `Item_Rating`
- `Item_Content`
- `Item_Hybrid`

Main scripts include:

- `recommender_comparison_6ways.py`
- `comparison_6ways_0518.py`
- `comparison_6ways_0601.py`
- `Hyperparameter_Tuning_6ways.py`
- `cf_experiment_ndcg.py`
- `Adjusted_Cosine.py`
- `test_CF.py`

Results are in `methods/similarity_6ways/results/`.

### 2. Matrix Factorization

Location: `methods/matrix_factorization/`

Main scripts include:

- `MF_0601.py`
- `MF_Comparison.py`
- `MF_Comparison copy.py`
- `MF_experiment_ndcg.py`
- `SVD_SGD.py`

Results and plots are in `methods/matrix_factorization/results/`.

### 3. Random Walk with Restart

Location: `methods/rwr/`

Main scripts include:

- `graph_rwr.py`
- `rwr_course_recommendation.py`
- `RWR_experiment_ndcg.py`
- `rwr_improved_experiments.py`
- `rwr_pure_linear.py`

**`RWR_experiment_ndcg.py` is the authoritative benchmark.** It evaluates
every strategy on the full `old/rating_matrix - rating_matrix.csv` matrix
against all 1813 rows of `test_set.csv` (407 students).

`rwr_pure_linear.py` and `rwr_improved_experiments.py` load a much smaller
demo matrix instead (`rating_matrix_train.csv` / `rating_matrix_truth.csv`,
8 students x 12 courses), and their `__main__` block only reports metrics
for a single hardcoded student (`student_idx=0`, 5 masked ratings). The
"0.94 MAE" result from earlier commits came from that single-student demo
run, not a full-dataset evaluation — on the full dataset RWR's real MAE is
1.8-2.0 (see the benchmark table below). Don't quote the old number without
this caveat.

### 4. GCN

Location: `methods/gcn/`

Main scripts include:

- `gcn.py`

GCN evaluation now uses the same formal dataset split as MF and RWR:

- Matrix: `old/rating_matrix - rating_matrix.csv`
- Test set: `test_set.csv`
- Test ratings are masked from the training graph before evaluation.
- Reported metrics: `MAE`, `RMSE`, `NDCG`, failed prediction count, and test row count.
- Output files are written to `methods/gcn/results/`.
- The maintained GCN version uses pure PyTorch. PyTorch Geometric is not required.

## Verified Benchmark (all methods, same train/test split)

`RWR_experiment_ndcg.py`, `gcn.py`, `MF_experiment_ndcg.py`, and
`cf_experiment_ndcg.py` all evaluate against the same data: the full
460 x 151 matrix (`old/rating_matrix - rating_matrix.csv`) with every
rating in `test_set.csv` (1813 rows / 407 students) masked out before
prediction. These are the only four scripts whose numbers are directly
comparable across methods. Last run 2026-08-10:

| Method                    | MAE  | RMSE | NDCG |
|---------------------------|------|------|------|
| Matrix Factorization      | 0.80 | 1.03 | 0.91 |
| CF Item_Hybrid (best K)   | 0.85 | 1.08 | 0.86 |
| CF User_Rating (best K)   | 0.93 | 1.20 | 0.92 |
| GCN                       | 1.43 | 1.77 | 0.84 |
| RWR Softmax (best temp/c) | 1.79 | 2.18 | 0.89 |
| RWR Linear                | 2.01 | 2.38 | 0.86 |
| RWR Binary                | 2.02 | 2.39 | 0.84 |

**NDCG caveat:** `calculate_ndcg()` returns 1.0 for any student with only
one masked test rating, since a single item has no ranking order to
score. 78 of the 407 students in `test_set.csv` have exactly one test row.
All four experiment scripts now exclude those students from the averaged
NDCG (see each script's `evaluate_*`/`run_*` function) rather than counting
them as a perfect score — the table above already reflects the correction.
NDCG still compresses into a fairly narrow band (0.84-0.92) regardless of
MAE, so treat MAE/RMSE as the primary signal when comparing methods, not
NDCG alone.

**RWR and GCN are the weak methods here**, and a manual spot check
(`shared/manual_verification.py`) shows why: both systematically
over-predict for students who gave low true grades (several sampled rows
have `Actual=1` with `Predicted` above 4.5), suggesting they rank by graph
connectivity/popularity rather than this particular student's actual
preference. This is the main thing to dig into before the next meeting —
candidates are the Min-Max rescaling step, the restart probability `c`,
and (for GCN) the very small embedding dimension (`embedding_dim=4`).

## Manual Verification

`shared/manual_verification.py` randomly samples rows (fixed seed) from
each method's `results/*_details.csv` (true rating vs. predicted rating)
so predictions can be checked by hand instead of trusting the aggregate
metrics alone. Run it after the four experiment scripts above; it writes
`shared/manual_verification_sample.csv`.

## Shared Files

The root CSV files are intentionally kept outside method folders because multiple methods read them directly:

- `rating_matrix.csv`
- `rating_matrix_train.csv`
- `rating_matrix_truth.csv`
- `test_set.csv`
- `transaction.csv`
- `flattened.csv`
- `item_content_100d.csv`

Utility scripts and shared plots live in `shared/`.

## Notes

Some older scripts still use direct relative paths, so the safest way to run experiments is from the project root.
