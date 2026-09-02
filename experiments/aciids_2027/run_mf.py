"""MF baseline using only shared manifests and the shared evaluator."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.evaluator import EvaluationPolicy, evaluate_predictions
from shared.reproducibility import set_global_seed, write_json

MATRIX = ROOT / "old" / "rating_matrix - rating_matrix.csv"
DEFAULT_SPLITS = Path(__file__).resolve().parent / "splits" / "seed_42"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


class MatrixFactorization(nn.Module):
    def __init__(self, users: int, items: int, dim: int, mean: float):
        super().__init__()
        self.user = nn.Embedding(users, dim)
        self.item = nn.Embedding(items, dim)
        self.user_bias = nn.Embedding(users, 1)
        self.item_bias = nn.Embedding(items, 1)
        self.global_bias = nn.Parameter(torch.tensor(mean, dtype=torch.float32))
        nn.init.normal_(self.user.weight, std=.01)
        nn.init.normal_(self.item.weight, std=.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user(users) * self.item(items)).sum(1) + self.user_bias(users).squeeze(1) + self.item_bias(items).squeeze(1) + self.global_bias


def fit_predict(train: pd.DataFrame, user_ids: list[str], item_ids: list[str], epochs: int, dim: int, lr: float, weight_decay: float, seed: int, device: torch.device) -> pd.DataFrame:
    set_global_seed(seed)
    u_map, i_map = {v: i for i, v in enumerate(user_ids)}, {v: i for i, v in enumerate(item_ids)}
    u = torch.tensor([u_map[x] for x in train.Student_ID], dtype=torch.long, device=device)
    i = torch.tensor([i_map[x] for x in train.Course_ID], dtype=torch.long, device=device)
    r = torch.tensor(train.Rating.to_numpy(), dtype=torch.float32, device=device)
    model = MatrixFactorization(len(user_ids), len(item_ids), dim, float(r.mean())).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(u, i), r)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        matrix = model.user.weight @ model.item.weight.T + model.user_bias.weight + model.item_bias.weight.T + model.global_bias
    return pd.DataFrame(matrix.cpu().numpy(), index=user_ids, columns=item_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS / "mf_smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Use publication-scale epoch/dimension grid")
    args = parser.parse_args()
    started = time.perf_counter()
    repro = set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = pd.read_csv(MATRIX, index_col=0).fillna(0)
    user_ids, item_ids = matrix.index.astype(str).tolist(), matrix.columns.astype(str).tolist()
    train = pd.read_csv(args.splits / "train.csv", dtype={"Student_ID": str, "Course_ID": str})
    validation = pd.read_csv(args.splits / "validation.csv", dtype={"Student_ID": str, "Course_ID": str})
    grid = ({"epochs": [10, 20, 50, 100], "dim": 64} if args.full else {"epochs": [2, 4], "dim": 8})
    fixed = {"lr": .01, "weight_decay": 1e-3}
    policy = EvaluationPolicy()
    train_mean = float(train.Rating.mean())
    validation_rows = []
    for epochs in grid["epochs"]:
        predictions = fit_predict(train, user_ids, item_ids, epochs, grid["dim"], **fixed, seed=args.seed, device=device)
        metrics, _, _ = evaluate_predictions(predictions, validation, train, item_ids, train_mean, policy)
        validation_rows.append({"epochs": epochs, "dim": grid["dim"], **fixed, **metrics})
    validation_df = pd.DataFrame(validation_rows)
    selected = validation_df.sort_values(["RMSE", "MAE", "epochs"]).iloc[0]
    selected_params = {"epochs": int(selected.epochs), "dim": grid["dim"], **fixed}

    # The test manifest is first loaded only after hyperparameter selection.
    test = pd.read_csv(args.splits / "test.csv", dtype={"Student_ID": str, "Course_ID": str})
    final_predictions = fit_predict(train, user_ids, item_ids, **selected_params, seed=args.seed, device=device)
    test_metrics, details, per_user = evaluate_predictions(final_predictions, test, train, item_ids, train_mean, policy)
    runtime = time.perf_counter() - started
    split_stats = json.loads((args.splits / "split_statistics.json").read_text())
    config = {
        "method": "biased_matrix_factorization", "mode": "full" if args.full else "smoke",
        "seed": args.seed, "reproducibility": repro, "execution_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "grid": grid, "fixed_hyperparameters": fixed,
        "selected_hyperparameters": selected_params,
        "evaluation_policy": policy.__dict__, "split_directory": str(args.splits),
        "cold_start_statistics": {name: split_stats["splits"][name] for name in ("validation", "test")},
        "runtime_seconds": runtime,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(args.output / "validation_metrics.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(args.output / "test_metrics.csv", index=False)
    details.to_csv(args.output / "test_predictions.csv", index=False)
    per_user.to_csv(args.output / "test_per_user_metrics.csv", index=False)
    write_json(args.output / "config.json", config)
    summary = ["# MF benchmark summary", "", f"- Mode: {config['mode']}", f"- Seed: {args.seed}", f"- Device: {device}",
               f"- Selected hyperparameters: `{selected_params}`", f"- Runtime: {runtime:.3f} seconds", "", "## Validation grid", "",
               markdown_table(validation_df), "", "## Final test metrics", "", markdown_table(pd.DataFrame([test_metrics])), ""]
    (args.output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Selected: {selected_params}")
    print(pd.DataFrame([test_metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
