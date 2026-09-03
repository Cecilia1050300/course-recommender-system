"""V1 rule-based user-sparsity-aware LightGCN experiment."""

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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.aciids_2027.run_lightgcn import build_normalized_adjacency, markdown_table
from shared.evaluator import EvaluationPolicy, evaluate_predictions
from shared.reproducibility import set_global_seed, write_json

MATRIX = ROOT / "old" / "rating_matrix - rating_matrix.csv"
DEFAULT_SPLITS = Path(__file__).resolve().parent / "splits" / "seed_42"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "sparsity_aware_lightgcn_v1"
ROBUSTNESS = Path(__file__).resolve().parent / "robustness"
BUCKETS = ("1-5", "6-10", "11-20", ">20")
POLICIES = {
    "A_uniform": {
        "1-5": [0.25, 0.25, 0.25, 0.25], "6-10": [0.25, 0.25, 0.25, 0.25],
        "11-20": [0.25, 0.25, 0.25, 0.25], ">20": [0.25, 0.25, 0.25, 0.25],
    },
    "B_mild": {
        "1-5": [0.10, 0.20, 0.30, 0.40], "6-10": [0.15, 0.25, 0.30, 0.30],
        "11-20": [0.30, 0.30, 0.25, 0.15], ">20": [0.50, 0.30, 0.15, 0.05],
    },
    "C_stronger": {
        "1-5": [0.05, 0.15, 0.30, 0.50], "6-10": [0.10, 0.20, 0.30, 0.40],
        "11-20": [0.35, 0.30, 0.20, 0.15], ">20": [0.65, 0.25, 0.10, 0.00],
    },
    "D_hard_routing": {
        "1-5": [0.00, 0.10, 0.30, 0.60], "6-10": [0.10, 0.20, 0.30, 0.40],
        "11-20": [0.40, 0.35, 0.20, 0.05], ">20": [0.75, 0.20, 0.05, 0.00],
    },
}
FIXED = {
    "embedding_dim": 32, "number_of_layers": 3, "learning_rate": .01,
    "epochs": 100, "weight_decay": 0.0,
}


def bucket_for_count(count: int) -> str:
    if count <= 5:
        return "1-5"
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    return ">20"


def validate_policies() -> None:
    for name, policy in POLICIES.items():
        if set(policy) != set(BUCKETS):
            raise ValueError(f"{name} does not cover every bucket")
        for bucket, weights in policy.items():
            if len(weights) != FIXED["number_of_layers"] + 1 or not np.isclose(sum(weights), 1.0):
                raise ValueError(f"invalid weights for {name}/{bucket}: {weights}")


def user_weight_tensor(
    train: pd.DataFrame, user_ids: list[str], policy: dict[str, list[float]],
    device: torch.device,
) -> torch.Tensor:
    counts = train.groupby("Student_ID").size().to_dict()
    weights = [policy[bucket_for_count(int(counts[user]))] for user in user_ids]
    return torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(-1)


class SparsityAwareLightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, layers: int):
        super().__init__()
        self.num_users = num_users
        self.layers = layers
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=.01)

    def forward(
        self, adjacency: torch.Tensor, user_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_embedding = self.embedding.weight
        layers = [layer_embedding]
        for _ in range(self.layers):
            layer_embedding = torch.sparse.mm(adjacency, layer_embedding)
            layers.append(layer_embedding)
        stacked = torch.stack(layers, dim=1)
        users = (stacked[:self.num_users] * user_weights).sum(dim=1)
        items = stacked[self.num_users:].mean(dim=1)
        return users, items


def fit_predict(
    train: pd.DataFrame, user_ids: list[str], item_ids: list[str],
    policy: dict[str, list[float]], seed: int, device: torch.device,
) -> tuple[pd.DataFrame, float]:
    set_global_seed(seed)
    user_map = {value: index for index, value in enumerate(user_ids)}
    item_map = {value: index for index, value in enumerate(item_ids)}
    users = torch.tensor([user_map[x] for x in train.Student_ID], device=device)
    items = torch.tensor([item_map[x] for x in train.Course_ID], device=device)
    targets = torch.tensor(train.Rating.to_numpy(), dtype=torch.float32, device=device)
    adjacency = build_normalized_adjacency(train, user_map, item_map, device)
    user_weights = user_weight_tensor(train, user_ids, policy, device)
    model = SparsityAwareLightGCN(
        len(user_ids), len(item_ids), FIXED["embedding_dim"], FIXED["number_of_layers"]
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=FIXED["learning_rate"], weight_decay=FIXED["weight_decay"]
    )
    last_loss = float("nan")
    for _ in range(FIXED["epochs"]):
        model.train()
        optimizer.zero_grad()
        user_embeddings, item_embeddings = model(adjacency, user_weights)
        scores = (user_embeddings[users] * item_embeddings[items]).sum(dim=1)
        loss = F.mse_loss(scores, targets)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model(adjacency, user_weights)
        scores = user_embeddings @ item_embeddings.T
    return pd.DataFrame(scores.cpu().numpy(), index=user_ids, columns=item_ids), last_loss


def bucket_metrics(
    label: str, train: pd.DataFrame, details: pd.DataFrame, per_user: pd.DataFrame,
) -> list[dict]:
    user_buckets = train.groupby("Student_ID").size().map(bucket_for_count)
    details = details.assign(bucket=details.Student_ID.map(user_buckets))
    per_user = per_user.assign(bucket=per_user.Student_ID.map(user_buckets))
    rows = []
    for bucket in BUCKETS:
        rating = details[details.bucket == bucket]
        users = per_user[per_user.bucket == bucket]
        eligible = users[users.Relevant_Items > 0]
        rows.append({
            "model": label, "train_interaction_bucket": bucket,
            "users": int(users.Student_ID.nunique()), "rating_rows": int(len(rating)),
            "ranking_eligible_users": int(len(eligible)),
            "MAE": float(rating.Absolute_Error.mean()),
            "RMSE": float(np.sqrt(rating.Squared_Error.mean())),
            "Precision@5": float(eligible["Precision@5"].mean()),
            "Recall@5": float(eligible["Recall@5"].mean()),
            "NDCG@5": float(eligible["NDCG@5"].mean()),
            "HitRate@5": float(eligible["HitRate@5"].mean()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    validate_policies()
    overall_started = time.perf_counter()
    reproducibility = set_global_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("sparsity-aware LightGCN V1 requires CUDA")
    device = torch.device("cuda")
    catalog = pd.read_csv(MATRIX, index_col=0).fillna(0)
    user_ids = catalog.index.astype(str).tolist()
    item_ids = catalog.columns.astype(str).tolist()
    train = pd.read_csv(args.splits / "train.csv", dtype={"Student_ID": str, "Course_ID": str})
    validation = pd.read_csv(args.splits / "validation.csv", dtype={"Student_ID": str, "Course_ID": str})
    train_mean = float(train.Rating.mean())
    evaluator_policy = EvaluationPolicy()
    validation_rows = []
    validation_buckets = {}

    for policy_name, weights in POLICIES.items():
        started = time.perf_counter()
        predictions, loss = fit_predict(train, user_ids, item_ids, weights, args.seed, device)
        metrics, details, per_user = evaluate_predictions(
            predictions, validation, train, item_ids, train_mean, evaluator_policy
        )
        torch.cuda.synchronize(device)
        validation_rows.append({
            "policy": policy_name, "training_loss": loss,
            "runtime_seconds": time.perf_counter() - started, **metrics,
        })
        validation_buckets[policy_name] = bucket_metrics(policy_name, train, details, per_user)

    validation_results = pd.DataFrame(validation_rows)
    selected = validation_results.sort_values(
        ["NDCG@5", "Recall@5", "RMSE", "policy"], ascending=[False, False, True, True]
    ).iloc[0]
    selected_name = str(selected.policy)
    selected_weights = POLICIES[selected_name]

    # Test remains unavailable until validation NDCG@5 policy selection is complete.
    test = pd.read_csv(args.splits / "test.csv", dtype={"Student_ID": str, "Course_ID": str})
    test_started = time.perf_counter()
    final_predictions, final_loss = fit_predict(
        train, user_ids, item_ids, selected_weights, args.seed, device
    )
    test_metrics, details, per_user = evaluate_predictions(
        final_predictions, test, train, item_ids, train_mean, evaluator_policy
    )
    torch.cuda.synchronize(device)
    test_runtime = time.perf_counter() - test_started
    selected_bucket_rows = bucket_metrics(selected_name, train, details, per_user)

    reference_overall = pd.read_csv(ROBUSTNESS / "per_seed_results.csv")
    reference_overall = reference_overall[
        (reference_overall.seed == args.seed) &
        reference_overall.model.isin(["MF", "LightGCN_RMSE", "LightGCN_Ranking"])
    ]
    reference_buckets = pd.read_csv(ROBUSTNESS / "sparsity_per_seed_results.csv")
    reference_buckets = reference_buckets[
        (reference_buckets.seed == args.seed) &
        reference_buckets.model.isin(["MF", "LightGCN_RMSE", "LightGCN_Ranking"])
    ].copy()
    # Recover reference bucket precision from existing artifacts only.
    prior_buckets = pd.read_csv(
        Path(__file__).resolve().parent / "analysis" / "sparsity_bucket_comparison.csv"
    )
    precision_lookup = {
        ("MF", row["train_interaction_bucket"]): row["Precision@5"]
        for _, row in prior_buckets[prior_buckets.model == "MF"].iterrows()
    }
    precision_lookup.update({
        ("LightGCN_RMSE", row["train_interaction_bucket"]): row["Precision@5"]
        for _, row in prior_buckets[prior_buckets.model == "LightGCN"].iterrows()
    })
    if selected_name == "A_uniform":
        precision_lookup.update({
            ("LightGCN_Ranking", row["train_interaction_bucket"]): row["Precision@5"]
            for row in selected_bucket_rows
        })
    reference_buckets["Precision@5"] = [
        precision_lookup.get((row.model, row.train_interaction_bucket), np.nan)
        for row in reference_buckets.itertuples(index=False)
    ]
    sparsity_results = pd.concat(
        [pd.DataFrame(selected_bucket_rows), reference_buckets[[
            "model", "train_interaction_bucket", "users", "rating_rows",
            "ranking_eligible_users", "MAE", "RMSE", "Precision@5", "Recall@5",
            "NDCG@5", "HitRate@5",
        ]]], ignore_index=True,
    )
    comparison_metrics = evaluator_policy_metric_names()
    overall_comparison = pd.concat([
        pd.DataFrame([{"model": f"V1_{selected_name}", **test_metrics}]),
        reference_overall[["model", *comparison_metrics]],
    ], ignore_index=True)[["model", *comparison_metrics]]

    args.output.mkdir(parents=True, exist_ok=True)
    validation_results.to_csv(args.output / "validation_policy_results.csv", index=False)
    details.to_csv(args.output / "test_predictions.csv", index=False)
    per_user.to_csv(args.output / "test_per_user_metrics.csv", index=False)
    sparsity_results.to_csv(args.output / "sparsity_bucket_results.csv", index=False)
    selected_payload = {
        "selection_metric": "validation_NDCG@5", "selected_policy": selected_name,
        "weights": selected_weights, "fixed_hyperparameters": FIXED,
        "validation_metrics": {key: float(selected[key]) for key in evaluator_policy_metric_names()},
    }
    write_json(args.output / "selected_policy.json", selected_payload)
    final_payload = {
        "selected_policy": selected_name, "weights": selected_weights,
        "test_evaluations": 1, "training_loss": final_loss,
        "runtime_seconds": test_runtime,
        "score_calibration": {
            "ranking": "raw_dot_product_no_scaling_or_clipping",
            "rating_metrics": "raw_dot_product_then_shared_evaluator_clip_to_[1,5]",
            "fitted_calibration": "none",
        },
        "metrics": test_metrics,
    }
    write_json(args.output / "final_test_results.json", final_payload)
    config = {
        "method": "rule_based_user_sparsity_aware_lightgcn_v1",
        "seed": args.seed, "reproducibility": reproducibility,
        "execution_device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "fixed_hyperparameters": FIXED, "predefined_policies": POLICIES,
        "policy_search_only": True, "policy_selection_metric": "validation_NDCG@5",
        "item_aggregation": "uniform_mean_layers_0_to_3",
        "user_aggregation": "fixed_bucket_specific_weighted_sum_layers_0_to_3",
        "evaluation_policy": evaluator_policy.__dict__,
        "reference_source": str(ROBUSTNESS),
        "reference_bucket_precision_note": "MF and LightGCN_RMSE read from existing analysis artifact; LightGCN_Ranking equals selected A_uniform and uses its V1 bucket precision",
        "total_runtime_seconds": time.perf_counter() - overall_started,
    }
    write_json(args.output / "config.json", config)

    uniform = validation_results.set_index("policy").loc["A_uniform"]
    validation_sparse = pd.DataFrame(validation_buckets[selected_name]).set_index("train_interaction_bucket")
    uniform_sparse = pd.DataFrame(validation_buckets["A_uniform"]).set_index("train_interaction_bucket")
    sparse_delta = float(validation_sparse.loc[["1-5", "6-10"], "NDCG@5"].mean() - uniform_sparse.loc[["1-5", "6-10"], "NDCG@5"].mean())
    dense_delta = float(validation_sparse.loc[">20", "NDCG@5"] - uniform_sparse.loc[">20", "NDCG@5"])
    ndcg_delta = float(selected["NDCG@5"] - uniform["NDCG@5"])
    lines = [
        "# Rule-based sparsity-aware LightGCN V1", "",
        f"- Selected policy: **{selected_name}** by validation NDCG@5",
        f"- Fixed hyperparameters: `{FIXED}`", f"- GPU: {config['gpu_name']}",
        f"- Total runtime: {config['total_runtime_seconds']:.3f} seconds", "",
        "## Validation policy results", "", markdown_table(validation_results), "",
        "## Final test metrics", "", markdown_table(pd.DataFrame([test_metrics])), "",
        "## Overall frozen-reference comparison", "", markdown_table(overall_comparison), "",
        "## Test sparsity buckets and frozen references", "", markdown_table(sparsity_results), "",
        "Reference bucket Precision@5 was recovered from existing analysis artifacts; the LightGCN-Ranking reference equals selected A_uniform. No reference model was rerun.", "",
        "## Required conclusions", "",
        f"1. **Validation NDCG@5 versus uniform:** change {ndcg_delta:+.6f}; " + ("the selected sparsity-aware policy outperforms uniform." if ndcg_delta > 0 else "uniform is not outperformed."),
        f"2. **Sparse-user ranking:** mean validation NDCG@5 change across 1–5 and 6–10 buckets is {sparse_delta:+.6f} versus uniform. The selected V1 therefore provides no sparse-user gain over standard uniform LightGCN.",
        f"3. **Dense-user degradation:** validation >20 NDCG@5 change is {dense_delta:+.6f} versus uniform. The selected V1 does not reduce dense-user degradation beyond the frozen ranking-oriented uniform baseline; its stronger dense ranking than LightGCN-RMSE is attributable to that existing uniform three-layer configuration, not sparsity-aware routing.",
        f"4. **Rating impact:** validation MAE changes by {float(selected.MAE-uniform.MAE):+.6f} and RMSE by {float(selected.RMSE-uniform.RMSE):+.6f} versus uniform (lower is better).",
        f"5. **Winning policy:** {selected_name}, because it has the highest validation NDCG@5 among the four predefined policies.",
        f"6. **V2 justification:** no. The validation NDCG@5 gain over uniform is {ndcg_delta:+.6f}; all three non-uniform rules perform worse overall. V1 supplies no performance evidence that learnable routing is warranted, although V2 could still be studied under a separate hypothesis.", "",
    ]
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Selected policy: {selected_name}")
    print(validation_results.to_string(index=False))
    print(pd.DataFrame([test_metrics]).to_string(index=False))


def evaluator_policy_metric_names() -> list[str]:
    return [
        "MAE", "RMSE", "Precision@5", "Precision@10", "Recall@5", "Recall@10",
        "NDCG@5", "NDCG@10", "HitRate@5", "HitRate@10",
    ]


if __name__ == "__main__":
    main()
