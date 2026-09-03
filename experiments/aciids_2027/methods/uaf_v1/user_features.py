"""TRAIN-only user features for UAF-V1."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "train_interaction_count",
    "log1p_train_interaction_count",
    "train_rating_mean",
    "train_rating_std",
    "train_rating_entropy",
    "train_positive_ratio",
    "mean_train_item_popularity",
]
COUNT_ONLY_COLUMNS = ["log1p_train_interaction_count"]
RICH_COLUMNS = FEATURE_COLUMNS


def _entropy(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(probabilities * np.log(probabilities)).sum())


def build_train_user_features(train: pd.DataFrame) -> pd.DataFrame:
    """Build features without consulting validation or test data."""
    data = train.copy()
    data["Student_ID"] = data["Student_ID"].astype(str)
    data["Course_ID"] = data["Course_ID"].astype(str)
    popularity = data.groupby("Course_ID").size()
    grouped = data.groupby("Student_ID", sort=True)
    features = grouped["Rating"].agg(
        train_interaction_count="size",
        train_rating_mean="mean",
        train_rating_std=lambda x: float(np.std(x.to_numpy(dtype=float), ddof=0)),
        train_rating_entropy=_entropy,
        train_positive_ratio=lambda x: float((x >= 4).mean()),
    )
    features["log1p_train_interaction_count"] = np.log1p(features["train_interaction_count"])
    mean_popularity = data.assign(item_popularity=data.Course_ID.map(popularity)).groupby(
        "Student_ID"
    )["item_popularity"].mean()
    features["mean_train_item_popularity"] = mean_popularity
    return features[FEATURE_COLUMNS].astype(float)


def standardize_features(
    features: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Fit and apply population standardization over TRAIN users."""
    selected = features[columns].copy()
    statistics: dict[str, dict[str, float]] = {}
    for column in columns:
        mean = float(selected[column].mean())
        std = float(selected[column].std(ddof=0))
        scale = std if std > 0 else 1.0
        selected[column] = (selected[column] - mean) / scale
        statistics[column] = {"mean": mean, "std": std, "scale_used": scale}
    return selected, statistics


def bucket_name(count: float) -> str:
    if count <= 5:
        return "1-5"
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    return ">20"
