"""Common explicit-rating and full-catalog ranking evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationPolicy:
    rating_min: float = 1.0
    rating_max: float = 5.0
    relevant_threshold: float = 4.0
    top_ks: tuple[int, ...] = (5, 10)
    missing_prediction: str = "train_global_mean"


def evaluate_predictions(
    predictions: pd.DataFrame,
    evaluation: pd.DataFrame,
    train: pd.DataFrame,
    all_items: Iterable[str],
    train_global_mean: float,
    policy: EvaluationPolicy = EvaluationPolicy(),
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """Evaluate ratings and ranking with one documented policy.

    Candidates are all catalog items except items observed for that user in
    train. Relevance is rating >= relevant_threshold in the supplied evaluation
    manifest. Items without an evaluation label are treated as non-relevant.
    Ranking aggregates exclude users with no relevant evaluation item.
    """
    required = {"Student_ID", "Course_ID", "Rating"}
    if not required.issubset(evaluation.columns) or not required.issubset(train.columns):
        raise ValueError(f"manifests must contain {sorted(required)}")

    pred = predictions.copy()
    pred.index = pred.index.astype(str)
    pred.columns = pred.columns.astype(str)
    eval_df = evaluation.copy()
    train_df = train.copy()
    for frame in (eval_df, train_df):
        frame["Student_ID"] = frame["Student_ID"].astype(str)
        frame["Course_ID"] = frame["Course_ID"].astype(str)

    details = []
    missing = 0
    fallback = 0
    cold_users = 0
    cold_items = 0
    train_users = set(train_df["Student_ID"])
    train_items = set(train_df["Course_ID"])
    for row in eval_df.itertuples(index=False):
        user, item, actual = row.Student_ID, row.Course_ID, float(row.Rating)
        is_cold_user, is_cold_item = user not in train_users, item not in train_items
        cold_users += int(is_cold_user)
        cold_items += int(is_cold_item)
        if not is_cold_user and not is_cold_item and user in pred.index and item in pred.columns and np.isfinite(pred.loc[user, item]):
            raw = float(pred.loc[user, item])
        else:
            raw = float(train_global_mean)
            fallback += 1
            missing += int(user not in pred.index or item not in pred.columns or not np.isfinite(pred.loc[user, item]))
        clipped = float(np.clip(raw, policy.rating_min, policy.rating_max))
        details.append({
            "Student_ID": user, "Course_ID": item, "Actual": actual,
            "Predicted_Raw": raw, "Predicted": clipped,
            "Absolute_Error": abs(actual - clipped),
            "Cold_User": is_cold_user, "Cold_Item": is_cold_item,
        })
    detail_df = pd.DataFrame(details)
    detail_df["Squared_Error"] = np.square(detail_df["Actual"] - detail_df["Predicted"])
    rating_by_user = detail_df.groupby("Student_ID").agg(
        User_MAE=("Absolute_Error", "mean"),
        User_MSE=("Squared_Error", "mean"),
        User_Rating_Rows=("Course_ID", "size"),
    )
    rating_by_user["User_RMSE"] = np.sqrt(rating_by_user.pop("User_MSE"))

    catalog = [str(item) for item in all_items]
    observed = train_df.groupby("Student_ID")["Course_ID"].agg(set).to_dict()
    relevant = eval_df[eval_df["Rating"] >= policy.relevant_threshold].groupby("Student_ID")["Course_ID"].agg(set).to_dict()
    per_user = []
    for user in sorted(eval_df["Student_ID"].unique()):
        candidates = [item for item in catalog if item not in observed.get(user, set())]
        scores = []
        for item in candidates:
            value = pred.loc[user, item] if user in pred.index and item in pred.columns else train_global_mean
            value = float(value) if np.isfinite(value) else float(train_global_mean)
            scores.append((item, value))
        ranked = [item for item, _ in sorted(scores, key=lambda pair: (-pair[1], pair[0]))]
        rel = relevant.get(user, set())
        row_metrics: dict[str, float | int | str] = {
            "Student_ID": user, "Relevant_Items": len(rel), "Candidates": len(candidates)
        }
        for k in policy.top_ks:
            top = ranked[:k]
            hits = sum(item in rel for item in top)
            row_metrics[f"Precision@{k}"] = hits / k if rel else np.nan
            row_metrics[f"Recall@{k}"] = hits / len(rel) if rel else np.nan
            row_metrics[f"HitRate@{k}"] = float(hits > 0) if rel else np.nan
            dcg = sum((1.0 / np.log2(rank + 2)) for rank, item in enumerate(top) if item in rel)
            ideal_len = min(k, len(rel))
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_len))
            row_metrics[f"NDCG@{k}"] = dcg / idcg if rel and idcg else np.nan
        per_user.append(row_metrics)
    per_user_df = pd.DataFrame(per_user)
    per_user_df = per_user_df.merge(rating_by_user, left_on="Student_ID", right_index=True, how="left")

    errors = detail_df["Actual"] - detail_df["Predicted"]
    metrics: dict[str, float | int] = {
        "MAE": float(np.abs(errors).mean()),
        "RMSE": float(np.sqrt(np.square(errors).mean())),
        "Rating_Rows": len(detail_df),
        "Missing_Predictions": missing,
        "Fallback_Predictions": fallback,
        "Cold_User_Rows": cold_users,
        "Cold_Item_Rows": cold_items,
        "Ranking_Eligible_Users": int((per_user_df["Relevant_Items"] > 0).sum()),
    }
    for k in policy.top_ks:
        for name in ("Precision", "Recall", "NDCG", "HitRate"):
            metrics[f"{name}@{k}"] = float(per_user_df[f"{name}@{k}"].mean(skipna=True))
    return metrics, detail_df, per_user_df
