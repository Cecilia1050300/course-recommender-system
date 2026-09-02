"""Publication-grade explicit-rating LightGCN baseline.

Raw LightGCN dot-product scores are passed unchanged for ranking. For MAE/RMSE,
the shared evaluator applies its documented [1, 5] clipping; no per-user scaling
or validation/test-fitted calibration is used.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.evaluator import EvaluationPolicy, evaluate_predictions
from shared.reproducibility import set_global_seed, write_json

MATRIX = ROOT / "old" / "rating_matrix - rating_matrix.csv"
DEFAULT_SPLITS = Path(__file__).resolve().parent / "splits" / "seed_42"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "lightgcn_publication"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def build_normalized_adjacency(
    train: pd.DataFrame,
    user_map: dict[str, int],
    item_map: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """Build D^-1/2 A D^-1/2 for the loop-free binary bipartite graph."""
    num_users, num_items = len(user_map), len(item_map)
    user_nodes = torch.tensor([user_map[x] for x in train.Student_ID], device=device)
    item_nodes = torch.tensor(
        [num_users + item_map[x] for x in train.Course_ID], device=device
    )
    sources = torch.cat((user_nodes, item_nodes))
    targets = torch.cat((item_nodes, user_nodes))
    degree = torch.bincount(sources, minlength=num_users + num_items).float()
    weights = degree[sources].rsqrt() * degree[targets].rsqrt()
    adjacency = torch.sparse_coo_tensor(
        torch.stack((sources, targets)), weights,
        (num_users + num_items, num_users + num_items), device=device,
    )
    return adjacency.coalesce()


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, layers: int):
        super().__init__()
        self.num_users = num_users
        self.layers = layers
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=.01)

    def forward(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        layer_embedding = self.embedding.weight
        embeddings = [layer_embedding]
        for _ in range(self.layers):
            layer_embedding = torch.sparse.mm(adjacency, layer_embedding)
            embeddings.append(layer_embedding)
        final = torch.stack(embeddings, dim=0).mean(dim=0)
        return final[:self.num_users], final[self.num_users:]


def fit_predict(
    train: pd.DataFrame,
    user_ids: list[str],
    item_ids: list[str],
    embedding_dim: int,
    number_of_layers: int,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, float]:
    set_global_seed(seed)
    user_map = {value: index for index, value in enumerate(user_ids)}
    item_map = {value: index for index, value in enumerate(item_ids)}
    users = torch.tensor([user_map[x] for x in train.Student_ID], device=device)
    items = torch.tensor([item_map[x] for x in train.Course_ID], device=device)
    targets = torch.tensor(train.Rating.to_numpy(), dtype=torch.float32, device=device)
    adjacency = build_normalized_adjacency(train, user_map, item_map, device)
    model = LightGCN(
        len(user_ids), len(item_ids), embedding_dim, number_of_layers
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    last_loss = float("nan")
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        user_embeddings, item_embeddings = model(adjacency)
        scores = (user_embeddings[users] * item_embeddings[items]).sum(dim=1)
        loss = F.mse_loss(scores, targets)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model(adjacency)
        scores = user_embeddings @ item_embeddings.T
    return pd.DataFrame(scores.cpu().numpy(), index=user_ids, columns=item_ids), last_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    overall_started = time.perf_counter()
    reproducibility = set_global_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("LightGCN publication baseline requires CUDA")
    device = torch.device("cuda")
    catalog = pd.read_csv(MATRIX, index_col=0).fillna(0)
    user_ids = catalog.index.astype(str).tolist()
    item_ids = catalog.columns.astype(str).tolist()
    train = pd.read_csv(args.splits / "train.csv", dtype={"Student_ID": str, "Course_ID": str})
    validation = pd.read_csv(args.splits / "validation.csv", dtype={"Student_ID": str, "Course_ID": str})
    policy = EvaluationPolicy()
    train_mean = float(train.Rating.mean())
    search_space = {
        "embedding_dim": [16, 32, 64],
        "number_of_layers": [1, 2, 3],
        "learning_rate": [.005, .01],
        "epochs": [50, 100],
        "weight_decay": [0.0, 1e-4],
    }

    keys = tuple(search_space)
    validation_rows = []
    for values in itertools.product(*(search_space[key] for key in keys)):
        params = dict(zip(keys, values))
        started = time.perf_counter()
        predictions, loss = fit_predict(
            train, user_ids, item_ids, **params, seed=args.seed, device=device
        )
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
        ["RMSE", "MAE", "embedding_dim", "number_of_layers", "learning_rate", "epochs", "weight_decay"]
    ).iloc[0]
    selected_params = {
        key: (int(selected[key]) if key in {"embedding_dim", "number_of_layers", "epochs"} else float(selected[key]))
        for key in keys
    }

    # Test is loaded only after validation selection has been finalized.
    test = pd.read_csv(args.splits / "test.csv", dtype={"Student_ID": str, "Course_ID": str})
    test_started = time.perf_counter()
    final_predictions, final_loss = fit_predict(
        train, user_ids, item_ids, **selected_params, seed=args.seed, device=device
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
        "score_calibration": {
            "ranking": "raw_dot_product_no_scaling_or_clipping",
            "rating_metrics": "raw_dot_product_then_shared_evaluator_clip_to_[1,5]",
            "fitted_calibration": "none",
        },
        "metrics": test_metrics,
    }
    write_json(args.output / "selected_config.json", selected_payload)
    write_json(args.output / "final_test_results.json", final_payload)
    metadata = {
        "method": "lightgcn_explicit_rating_mse",
        "seed": args.seed,
        "reproducibility": reproducibility,
        "execution_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "search_space": search_space,
        "configuration_count": len(validation_results),
        "training_objective": "MSE_on_observed_train_ratings",
        "score_calibration": final_payload["score_calibration"],
        "evaluation_policy": policy.__dict__,
        "split_directory": str(args.splits),
        "total_runtime_seconds": time.perf_counter() - overall_started,
    }
    write_json(args.output / "config.json", metadata)
    summary = [
        "# LightGCN publication baseline", "",
        f"- Seed: {args.seed}", f"- Device: {device} ({metadata['gpu_name']})",
        f"- Validation configurations: {len(validation_results)}",
        f"- Selected hyperparameters: `{selected_params}`",
        f"- Validation MAE / RMSE: {selected.MAE:.6f} / {selected.RMSE:.6f}",
        f"- Final test runtime: {test_runtime:.3f} seconds",
        "- Ranking scores: raw dot products (no scaling or clipping)",
        "- Rating calibration: raw dot products clipped to [1,5] only by the shared evaluator", "",
        "## Validation grid", "", markdown_table(validation_results), "",
        "## Final test metrics", "", markdown_table(pd.DataFrame([test_metrics])), "",
    ]
    (args.output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Selected: {selected_params}")
    print(pd.DataFrame([test_metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
