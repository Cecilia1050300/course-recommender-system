"""Observed-rating ordinal pairs for OAF-V2."""

from __future__ import annotations

import numpy as np
import pandas as pd


EXPECTED = {"eligible_users": 314, "total_pairs": 28076,
            "gap_counts": {1: 16528, 2: 8280, 3: 2779, 4: 489}}


def build_ordinal_pairs(train: pd.DataFrame) -> pd.DataFrame:
    """Return every directed (preferred, less-preferred) TRAIN-observed pair."""
    rows: list[dict] = []
    for user, group in train.groupby("Student_ID", sort=True):
        values = list(group[["Course_ID", "Rating"]].itertuples(index=False, name=None))
        for item_i, rating_i in values:
            for item_j, rating_j in values:
                if rating_i > rating_j:
                    rows.append({"Student_ID": str(user), "preferred_item": str(item_i),
                                 "less_preferred_item": str(item_j),
                                 "rating_gap": int(rating_i - rating_j)})
    return pd.DataFrame(rows)


def pair_statistics(pairs: pd.DataFrame) -> dict:
    gaps = pairs.rating_gap.value_counts().sort_index()
    return {
        "eligible_users": int(pairs.Student_ID.nunique()),
        "total_pairs": int(len(pairs)),
        "gap_counts": {str(gap): int(gaps.get(gap, 0)) for gap in [1, 2, 3, 4]},
        "gap_percentages": {str(gap): float(100 * gaps.get(gap, 0) / len(pairs))
                            for gap in [1, 2, 3, 4]},
    }


def verify_pair_statistics(statistics: dict) -> None:
    actual_gaps = {int(key): value for key, value in statistics["gap_counts"].items()}
    if (statistics["eligible_users"] != EXPECTED["eligible_users"] or
            statistics["total_pairs"] != EXPECTED["total_pairs"] or
            actual_gaps != EXPECTED["gap_counts"]):
        raise RuntimeError(f"Ordinal-pair audit mismatch: {statistics}; expected {EXPECTED}")
