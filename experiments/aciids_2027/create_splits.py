"""Create the immutable ACIIDS 2027 per-user warm-start manifests."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.reproducibility import set_global_seed, write_json

MATRIX = ROOT / "old" / "rating_matrix - rating_matrix.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "splits" / "seed_42"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preserve_or_write_manifest(path: Path, expected: pd.DataFrame) -> None:
    """Create a manifest once; refuse to mutate an existing split."""
    if path.exists():
        existing = pd.read_csv(path, dtype={"Student_ID": str, "Course_ID": str})
        try:
            pd.testing.assert_frame_equal(existing, expected, check_dtype=False)
        except AssertionError as error:
            raise RuntimeError(f"immutable split differs from regenerated data: {path}") from error
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    expected.to_csv(temporary, index=False)
    temporary.replace(path)


def build_manifests(matrix: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    records = matrix.stack().reset_index()
    records.columns = ["Student_ID", "Course_ID", "Rating"]
    records = records[records["Rating"] > 0].copy()
    records[["Student_ID", "Course_ID"]] = records[["Student_ID", "Course_ID"]].astype(str)
    rng = np.random.default_rng(seed)
    assignments = pd.Series("train", index=records.index, dtype=object)
    for _, indices in records.groupby("Student_ID", sort=True).groups.items():
        idx = np.array(list(indices))
        n = len(idx)
        if n < 3:
            continue
        shuffled = rng.permutation(idx)
        holdout_each = max(1, int(np.floor(n * 0.10)))
        holdout_each = min(holdout_each, (n - 1) // 2)
        assignments.loc[shuffled[:holdout_each]] = "validation"
        assignments.loc[shuffled[holdout_each:2 * holdout_each]] = "test"

    # Enforce item warm-start: every held-out item must occur in train.
    for item, indices in records.groupby("Course_ID", sort=True).groups.items():
        idx = list(indices)
        if not (assignments.loc[idx] == "train").any():
            move_idx = sorted(idx)[0]
            assignments.loc[move_idx] = "train"

    manifests = {name: records.loc[assignments == name].sort_values(["Student_ID", "Course_ID"]).reset_index(drop=True)
                 for name in ("train", "validation", "test")}
    combined = pd.concat([df.assign(Split=name) for name, df in manifests.items()])
    if len(combined) != len(records) or combined.duplicated(["Student_ID", "Course_ID"]).any():
        raise RuntimeError("split integrity check failed")
    return manifests


def describe(manifests: dict[str, pd.DataFrame], n_users: int, n_items: int) -> tuple[dict, pd.DataFrame]:
    train_users, train_items = set(manifests["train"].Student_ID), set(manifests["train"].Course_ID)
    stats = {"catalog_users": n_users, "catalog_items": n_items, "splits": {}}
    rows = []
    for name, df in manifests.items():
        counts = df.groupby("Student_ID").size()
        stats["splits"][name] = {
            "ratings": len(df), "users": df.Student_ID.nunique(), "items": df.Course_ID.nunique(),
            "density": len(df) / (n_users * n_items),
            "cold_start_users_vs_train": len(set(df.Student_ID) - train_users),
            "cold_start_items_vs_train": len(set(df.Course_ID) - train_items),
            "per_user_interactions": {
                "min": int(counts.min()), "max": int(counts.max()), "mean": float(counts.mean()),
                "median": float(counts.median()), "p25": float(counts.quantile(.25)), "p75": float(counts.quantile(.75)),
            },
        }
        all_counts = counts.reindex(sorted(train_users | set(manifests["validation"].Student_ID) | set(manifests["test"].Student_ID)), fill_value=0)
        rows.extend({"Student_ID": user, "Split": name, "Interactions": int(value)} for user, value in all_counts.items())
    return stats, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    repro = set_global_seed(args.seed)
    matrix = pd.read_csv(MATRIX, index_col=0).fillna(0)
    matrix.index, matrix.columns = matrix.index.astype(str), matrix.columns.astype(str)
    manifests = build_manifests(matrix, args.seed)
    stats, distribution = describe(manifests, len(matrix), len(matrix.columns))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, df in manifests.items():
        preserve_or_write_manifest(args.output / f"{name}.csv", df)
    preserve_or_write_manifest(args.output / "per_user_interactions.csv", distribution)
    manifest_hashes = {f"{name}.csv": sha256(args.output / f"{name}.csv") for name in manifests}
    write_json(args.output / "split_statistics.json", {
        "strategy": "per_user_warm_start_80_10_10_approx",
        "immutable": True,
        "source": str(MATRIX.relative_to(ROOT)),
        "source_sha256": sha256(MATRIX),
        "manifest_sha256": manifest_hashes,
        "reproducibility": repro,
        **stats,
    })
    print(pd.DataFrame(stats["splits"]).T.to_string())


if __name__ == "__main__":
    main()
