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
