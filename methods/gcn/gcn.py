"""
GCN course recommender experiment.

This is the single maintained GCN script for the project. It uses pure PyTorch
instead of PyTorch Geometric so it is easier to run and explain in a lab
environment.

Evaluation protocol:
- Use old/rating_matrix - rating_matrix.csv as the full rating matrix.
- Use test_set.csv as the held-out test set.
- Mask every test_set rating from the training graph before training.
- Report MAE, RMSE, NDCG, failed prediction count, and test row count.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = PROJECT_ROOT / "old" / "rating_matrix - rating_matrix.csv"
TEST_PATH = PROJECT_ROOT / "test_set.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_data():
    rating_df = pd.read_csv(MATRIX_PATH, index_col=0).fillna(0)
    rating_df.index = rating_df.index.astype(str)
    rating_df.columns = rating_df.columns.astype(str)

    test_df = pd.read_csv(TEST_PATH)
    test_df["Student_ID"] = test_df["Student_ID"].astype(str)
    test_df["Course_ID"] = test_df["Course_ID"].astype(str)
    return rating_df, test_df


def make_masked_train(rating_df, test_df):
    train_df = rating_df.copy()
    masked_count = 0

    for _, row in test_df.iterrows():
        user_id = row["Student_ID"]
        course_id = row["Course_ID"]
        if user_id in train_df.index and course_id in train_df.columns:
            train_df.loc[user_id, course_id] = 0.0
            masked_count += 1

    return train_df, masked_count


def build_gcn_matrix(train_matrix):
    num_users, num_items = train_matrix.shape
    total_nodes = num_users + num_items

    adjacency = np.zeros((total_nodes, total_nodes), dtype=np.float64)
    binary_edges = (train_matrix > 0).astype(np.float64)
    adjacency[:num_users, num_users:] = binary_edges
    adjacency[num_users:, :num_users] = binary_edges.T

    adjacency_with_loops = adjacency + np.eye(total_nodes)
    degrees = adjacency_with_loops.sum(axis=1)
    d_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
    d_inv_sqrt[degrees == 0] = 0.0
    normalized = np.diag(d_inv_sqrt).dot(adjacency_with_loops).dot(np.diag(d_inv_sqrt))
    return torch.FloatTensor(normalized)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, normalized_adjacency):
        return normalized_adjacency @ features @ self.weight


class GCNRecommender(nn.Module):
    def __init__(self, total_nodes, hidden_dim=12, embedding_dim=4, dropout=0.1):
        super().__init__()
        self.features = nn.Parameter(torch.eye(total_nodes), requires_grad=False)
        self.gcn1 = GCNLayer(total_nodes, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, embedding_dim)
        self.dropout = dropout

    def forward(self, normalized_adjacency):
        hidden = self.gcn1(self.features, normalized_adjacency)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.gcn2(hidden, normalized_adjacency)


def minmax_to_rating(raw_scores, candidate_mask):
    ratings = np.zeros_like(raw_scores, dtype=float)
    candidate_values = raw_scores[candidate_mask]

    if candidate_values.size == 0:
        return ratings

    min_score = candidate_values.min()
    max_score = candidate_values.max()
    if max_score != min_score:
        ratings[candidate_mask] = 1.0 + 4.0 * (
            (candidate_values - min_score) / (max_score - min_score)
        )
    else:
        ratings[candidate_mask] = 3.0
    return ratings


def build_prediction_matrix(embeddings, train_df):
    train_matrix = train_df.values.astype(np.float64)
    user_ids = train_df.index.tolist()
    item_ids = train_df.columns.tolist()
    num_users, _ = train_matrix.shape

    user_embeddings = embeddings[:num_users].detach().cpu().numpy()
    item_embeddings = embeddings[num_users:].detach().cpu().numpy()
    raw_scores = user_embeddings.dot(item_embeddings.T)

    prediction_rows = []
    for user_idx in range(len(user_ids)):
        candidate_mask = train_matrix[user_idx] == 0
        prediction_rows.append(minmax_to_rating(raw_scores[user_idx], candidate_mask))

    return pd.DataFrame(prediction_rows, index=user_ids, columns=item_ids)


def calculate_ndcg(user_records):
    if len(user_records) <= 1:
        return 1.0

    y_true = np.array([record["true"] for record in user_records], dtype=float)
    y_pred = np.array([record["pred"] for record in user_records], dtype=float)

    ranked_idx = np.argsort(y_pred)[::-1]
    dcg = sum(
        (2 ** y_true[i] - 1) / np.log2(rank + 2)
        for rank, i in enumerate(ranked_idx)
    )
    ideal_scores = np.sort(y_true)[::-1]
    idcg = sum(
        (2 ** score - 1) / np.log2(rank + 2)
        for rank, score in enumerate(ideal_scores)
    )
    return float(dcg / idcg) if idcg > 0 else 1.0


def evaluate_prediction_matrix(pred_df, test_df):
    errors = []
    details = []
    ndcg_bundles = {}
    failed_count = 0

    for _, row in test_df.iterrows():
        user_id = row["Student_ID"]
        course_id = row["Course_ID"]
        actual = float(row["Actual_Grade"])

        if user_id in pred_df.index and course_id in pred_df.columns:
            predicted = float(np.clip(pred_df.loc[user_id, course_id], 1.0, 5.0))
            error = abs(actual - predicted)
        else:
            predicted = 0.0
            error = 5.0
            failed_count += 1

        errors.append(error)
        details.append(
            {
                "Student_ID": user_id,
                "Course_ID": course_id,
                "Actual": actual,
                "Predicted": round(predicted, 4),
                "Error": round(error, 4),
            }
        )
        ndcg_bundles.setdefault(user_id, []).append(
            {"true": actual, "pred": predicted}
        )

    error_array = np.array(errors, dtype=float)
    # 只對有 >=2 筆蓋牌測試資料的學生算 NDCG，單筆學生沒有排序可比較，
    # calculate_ndcg 對他們會硬回傳 1.0，混進平均會虛灌分數。
    ndcg_scores = [calculate_ndcg(r) for r in ndcg_bundles.values() if len(r) > 1]
    summary = {
        "MAE": float(error_array.mean()),
        "RMSE": float(np.sqrt(np.mean(error_array ** 2))),
        "NDCG": float(np.mean(ndcg_scores)) if ndcg_scores else float("nan"),
        "NDCG_Users": len(ndcg_scores),
        "Failed_Predictions": failed_count,
        "Test_Rows": int(len(test_df)),
    }
    return summary, pd.DataFrame(details)


def save_report(summary, details_df):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Method": "GCN_Pure_Torch", **summary}]).to_csv(
        RESULTS_DIR / "GCN_Pure_Torch_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    details_df.to_csv(
        RESULTS_DIR / "GCN_Pure_Torch_details.csv",
        index=False,
        encoding="utf-8-sig",
    )


def run_experiment(epochs=20, lr=0.01, weight_decay=1e-4):
    torch.manual_seed(42)
    np.random.seed(42)

    rating_df, test_df = load_data()
    train_df, masked_count = make_masked_train(rating_df, test_df)
    train_matrix = train_df.values.astype(np.float64)
    num_users, num_items = train_matrix.shape

    normalized_adjacency = build_gcn_matrix(train_matrix)
    model = GCNRecommender(total_nodes=num_users + num_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_user_idx, train_item_idx = np.where(train_matrix > 0)
    train_user_idx_t = torch.LongTensor(train_user_idx)
    train_item_idx_t = torch.LongTensor(train_item_idx)
    train_targets_t = torch.FloatTensor(train_matrix[train_user_idx, train_item_idx])

    print(f"Rating matrix: {num_users} users x {num_items} courses")
    print(f"Masked test ratings from training graph: {masked_count}")

    last_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        embeddings = model(normalized_adjacency)
        user_embeddings = embeddings[:num_users]
        item_embeddings = embeddings[num_users:]
        predictions = torch.sum(
            user_embeddings[train_user_idx_t] * item_embeddings[train_item_idx_t],
            dim=1,
        )
        loss = F.mse_loss(predictions, train_targets_t)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

        if epoch == 1 or epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred_df = build_prediction_matrix(model(normalized_adjacency), train_df)
                summary, _ = evaluate_prediction_matrix(pred_df, test_df)
            print(
                f"Epoch {epoch:2d} | Loss: {last_loss:.4f} | "
                f"MAE: {summary['MAE']:.4f} | RMSE: {summary['RMSE']:.4f} | "
                f"NDCG: {summary['NDCG']:.4f}"
            )

    model.eval()
    with torch.no_grad():
        final_pred_df = build_prediction_matrix(model(normalized_adjacency), train_df)

    summary, details_df = evaluate_prediction_matrix(final_pred_df, test_df)
    summary["Training_Loss"] = last_loss
    summary["Masked_Ratings"] = masked_count
    save_report(summary, details_df)
    return summary


def print_summary(summary):
    print("\n" + "=" * 70)
    print("GCN Pure PyTorch Final Result")
    print("=" * 70)
    print(f"MAE  : {summary['MAE']:.4f}")
    print(f"RMSE : {summary['RMSE']:.4f}")
    print(f"NDCG : {summary['NDCG']:.4f} (based on {summary['NDCG_Users']} students with >=2 test items)")
    print(f"Failed predictions: {summary['Failed_Predictions']}")
    print(f"Test rows: {summary['Test_Rows']}")
    print("=" * 70)


if __name__ == "__main__":
    final_summary = run_experiment()
    print_summary(final_summary)
