"""Publication-protocol migration of the legacy two-layer vanilla GCN."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.gcn.gcn import GCNRecommender, build_gcn_matrix, build_prediction_matrix
from shared.evaluator import EvaluationPolicy, evaluate_predictions
from shared.reproducibility import set_global_seed, write_json

MATRIX = ROOT / "old" / "rating_matrix - rating_matrix.csv"
DEFAULT_SPLITS = Path(__file__).resolve().parent / "splits" / "seed_42"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "gcn_publication"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def make_train_matrix(train: pd.DataFrame, user_ids: list[str], item_ids: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=user_ids, columns=item_ids)
    for row in train.itertuples(index=False):
        matrix.loc[row.Student_ID, row.Course_ID] = float(row.Rating)
    return matrix


def fit_predict(
    train_matrix: pd.DataFrame,
    hidden_dim: int,
    embedding_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, float]:
    set_global_seed(seed)
    values = train_matrix.to_numpy(dtype=np.float64)
    user_idx, item_idx = np.where(values > 0)
    users = torch.tensor(user_idx, dtype=torch.long, device=device)
    items = torch.tensor(item_idx, dtype=torch.long, device=device)
    targets = torch.tensor(values[user_idx, item_idx], dtype=torch.float32, device=device)
    adjacency = build_gcn_matrix(values).to(device)
    model = GCNRecommender(
        total_nodes=sum(values.shape), hidden_dim=hidden_dim,
        embedding_dim=embedding_dim, dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    last_loss = float("nan")
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(adjacency)
        predictions = torch.sum(
            embeddings[users] * embeddings[values.shape[0] + items], dim=1
        )
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        predictions = build_prediction_matrix(model(adjacency), train_matrix)
    return predictions, last_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    overall_started = time.perf_counter()
    reproducibility = set_global_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Vanilla GCN publication baseline requires CUDA")
    device = torch.device("cuda")
    catalog = pd.read_csv(MATRIX, index_col=0).fillna(0)
    user_ids = catalog.index.astype(str).tolist()
    item_ids = catalog.columns.astype(str).tolist()
    train = pd.read_csv(args.splits / "train.csv", dtype={"Student_ID": str, "Course_ID": str})
    validation = pd.read_csv(args.splits / "validation.csv", dtype={"Student_ID": str, "Course_ID": str})
    train_matrix = make_train_matrix(train, user_ids, item_ids)
    train_mean = float(train.Rating.mean())
    policy = EvaluationPolicy()
    search_space = {
        "hidden_dim": [12, 32],
        "embedding_dim": [4, 16],
        "epochs": [20, 50, 100],
        "learning_rate": [.005, .01],
        "weight_decay": [1e-4],
        "dropout": [.1],
    }

    keys = tuple(search_space)
    validation_rows = []
    for values in itertools.product(*(search_space[key] for key in keys)):
        params = dict(zip(keys, values))
        started = time.perf_counter()
        predictions, loss = fit_predict(train_matrix, **params, seed=args.seed, device=device)
        metrics, _, _ = evaluate_predictions(
            predictions, validation, train, item_ids, train_mean, policy
        )
        torch.cuda.synchronize(device)
        validation_rows.append({
            **params, "training_loss": loss,
            "runtime_seconds": time.perf_counter() - started, **metrics,
        })

    validation_results = pd.DataFrame(validation_rows)
    selected = validation_results.sort_values(
        ["RMSE", "MAE", "hidden_dim", "embedding_dim", "epochs", "learning_rate"]
    ).iloc[0]
    selected_params = {
        key: (int(selected[key]) if key in {"hidden_dim", "embedding_dim", "epochs"} else float(selected[key]))
        for key in keys
    }

    # Test data is deliberately unavailable until validation selection is final.
    test = pd.read_csv(args.splits / "test.csv", dtype={"Student_ID": str, "Course_ID": str})
    test_started = time.perf_counter()
    final_predictions, final_loss = fit_predict(
        train_matrix, **selected_params, seed=args.seed, device=device
    )
    test_metrics, details, per_user = evaluate_predictions(
        final_predictions, test, train, item_ids, train_mean, policy
    )
    torch.cuda.synchronize(device)
    test_runtime = time.perf_counter() - test_started

    args.output.mkdir(parents=True, exist_ok=True)
    validation_results.to_csv(args.output / "validation_results.csv", index=False)
    details.to_csv(args.output / "test_predictions.csv", index=False)
    per_user.to_csv(args.output / "test_per_user_metrics.csv", index=False)
    selected_payload = {
        "selection_metric": "validation_RMSE",
        "selected_hyperparameters": selected_params,
        "validation_MAE": float(selected.MAE),
        "validation_RMSE": float(selected.RMSE),
    }
    final_payload = {
        "selected_hyperparameters": selected_params,
        "test_evaluations": 1,
        "training_loss": final_loss,
        "runtime_seconds": test_runtime,
        "metrics": test_metrics,
    }
    write_json(args.output / "selected_config.json", selected_payload)
    write_json(args.output / "final_test_results.json", final_payload)
    metadata = {
        "method": "two_layer_vanilla_gcn",
        "seed": args.seed,
        "reproducibility": reproducibility,
        "execution_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "search_space": search_space,
        "configuration_count": len(validation_results),
        "evaluation_policy": policy.__dict__,
        "split_directory": str(args.splits),
        "total_runtime_seconds": time.perf_counter() - overall_started,
    }
    write_json(args.output / "config.json", metadata)
    summary = [
        "# Vanilla GCN publication baseline", "",
        f"- Seed: {args.seed}", f"- Device: {device} ({metadata['gpu_name']})",
        f"- Validation configurations: {len(validation_results)}",
        f"- Selected hyperparameters: `{selected_params}`",
        f"- Validation MAE / RMSE: {selected.MAE:.6f} / {selected.RMSE:.6f}",
        f"- Final test runtime: {test_runtime:.3f} seconds", "",
        "## Validation grid", "", markdown_table(validation_results), "",
        "## Final test metrics", "", markdown_table(pd.DataFrame([test_metrics])), "",
    ]
    (args.output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Selected: {selected_params}")
    print(pd.DataFrame([test_metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
