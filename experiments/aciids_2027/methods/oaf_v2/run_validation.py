"""Validation-only Objective-Aligned Fusion V2.

No test path or test artifact is defined by this experiment.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.aciids_2027.methods.oaf_v2.objective_aligned_fusion import (
    GlobalAlpha, UserCountAlpha, joint_losses,
)
from experiments.aciids_2027.methods.oaf_v2.ordinal_pairs import (
    build_ordinal_pairs, pair_statistics, verify_pair_statistics,
)
from experiments.aciids_2027.methods.uaf_v1.user_features import build_train_user_features
from experiments.aciids_2027.run_lightgcn import fit_predict as fit_lightgcn
from experiments.aciids_2027.run_mf import fit_predict as fit_mf
from shared.evaluator import EvaluationPolicy, evaluate_predictions
from shared.reproducibility import set_global_seed, write_json

SEED = 42
SPLITS = ROOT / "experiments/aciids_2027/splits/seed_42"
RECOVERED = ROOT / "experiments/aciids_2027/artifacts/validation_predictions"
RESULTS = ROOT / "experiments/aciids_2027/results"
UAF_V1 = RESULTS / "uaf_v1"
OUTPUT = RESULTS / "oaf_v2"
CATALOG = ROOT / "old/rating_matrix - rating_matrix.csv"
POLICY = EvaluationPolicy()
MF_CONFIG = {"latent_dim": 32, "epochs": 100, "learning_rate": 0.005, "weight_decay": 0.001}
LG_CONFIG = {"embedding_dim": 64, "number_of_layers": 1, "epochs": 100,
             "learning_rate": 0.01, "weight_decay": 0.0}
FIXED_ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
INITIAL_ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]
LAMBDAS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
LEARNING_RATES = [0.001, 0.005, 0.01, 0.05]
MAX_EPOCHS = [50, 100, 200]
PAIR_MODES = ["unweighted", "gap_weighted"]
REPRODUCTION_ATOL = 1e-7
MATERIAL_NDCG_GAIN = 0.005


def recovered_matrix(model: str, users: list[str], items: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(RECOVERED / model / "validation_candidate_scores.csv",
                        dtype={"user_id": str, "course_id": str})
    return frame.pivot(index="user_id", columns="course_id", values="score").reindex(
        index=users, columns=items)


def expected_metrics(model: str) -> dict:
    payload = json.loads((RECOVERED / model / "reproduction_check.json").read_text())
    return {name: values["saved"] for name, values in payload["metric_comparison"].items()}


def reproduce(train: pd.DataFrame, validation: pd.DataFrame, users: list[str],
              items: list[str], device: torch.device):
    mf = fit_mf(train, users, items, epochs=MF_CONFIG["epochs"], dim=MF_CONFIG["latent_dim"],
                lr=MF_CONFIG["learning_rate"], weight_decay=MF_CONFIG["weight_decay"],
                seed=SEED, device=device)
    lg, _ = fit_lightgcn(train, users, items, embedding_dim=LG_CONFIG["embedding_dim"],
                         number_of_layers=LG_CONFIG["number_of_layers"],
                         learning_rate=LG_CONFIG["learning_rate"], epochs=LG_CONFIG["epochs"],
                         weight_decay=LG_CONFIG["weight_decay"], seed=SEED, device=device)
    checks, candidate_checks = {}, {}
    for name, matrix in (("mf", mf), ("lightgcn_rmse", lg)):
        metrics, _, _ = evaluate_predictions(
            matrix, validation, train, items, float(train.Rating.mean()), POLICY)
        expected = expected_metrics(name)
        differences = {key: abs(float(metrics[key]) - float(expected[key])) for key in expected}
        checks[name] = max(differences.values())
        if checks[name] > REPRODUCTION_ATOL:
            raise RuntimeError(f"{name} reproduction failed: {differences}")
        recovered = recovered_matrix(name, users, items)
        mask = recovered.notna().to_numpy()
        recovered_values = recovered.to_numpy()
        candidate_checks[name] = float(np.max(np.abs(matrix.to_numpy()[mask] - recovered_values[mask])))
        matrix.values[mask] = recovered_values[mask]
    return mf, lg, checks, candidate_checks


def normalization(matrix: pd.DataFrame, train: pd.DataFrame,
                  users: list[str], items: list[str]) -> pd.DataFrame:
    """Fit per-user moments on TRAIN-unobserved catalog scores; transform all items."""
    observed = train.groupby("Student_ID")["Course_ID"].agg(set).to_dict()
    result = pd.DataFrame(index=matrix.index, columns=matrix.columns, dtype=float)
    for user in users:
        candidates = [item for item in items if item not in observed.get(user, set())]
        reference = matrix.loc[user, candidates].to_numpy(dtype=float)
        result.loc[user] = ((matrix.loc[user].to_numpy(dtype=float) - reference.mean()) /
                            (reference.std(ddof=0) + 1e-8))
    return result


def fused_matrix(first: pd.DataFrame, second: pd.DataFrame, alpha) -> pd.DataFrame:
    if np.isscalar(alpha):
        values = float(alpha) * first.to_numpy() + (1.0 - float(alpha)) * second.to_numpy()
    else:
        weights = pd.Series(alpha).reindex(first.index).to_numpy(dtype=float)[:, None]
        values = weights * first.to_numpy() + (1.0 - weights) * second.to_numpy()
    return pd.DataFrame(values, index=first.index, columns=first.columns)


def evaluate_fusion(alpha, mf: pd.DataFrame, lg: pd.DataFrame,
                    z_mf: pd.DataFrame, z_lg: pd.DataFrame,
                    validation: pd.DataFrame, train: pd.DataFrame,
                    items: list[str]) -> dict:
    rating, _, _ = evaluate_predictions(fused_matrix(mf, lg, alpha), validation, train,
                                        items, float(train.Rating.mean()), POLICY)
    ranking, _, _ = evaluate_predictions(fused_matrix(z_mf, z_lg, alpha), validation, train,
                                         items, float(train.Rating.mean()), POLICY)
    result = dict(rating)
    for k in POLICY.top_ks:
        for metric in ["Precision", "Recall", "NDCG", "HitRate"]:
            result[f"{metric}@{k}"] = ranking[f"{metric}@{k}"]
    result["Ranking_Eligible_Users"] = ranking["Ranking_Eligible_Users"]
    return result


def tensor_data(train: pd.DataFrame, validation: pd.DataFrame, pairs: pd.DataFrame,
                mf: pd.DataFrame, lg: pd.DataFrame, z_mf: pd.DataFrame,
                z_lg: pd.DataFrame, users: list[str], items: list[str], device: torch.device) -> dict:
    u_map, i_map = {x: i for i, x in enumerate(users)}, {x: i for i, x in enumerate(items)}
    train_u = np.array([u_map[x] for x in train.Student_ID])
    train_i = np.array([i_map[x] for x in train.Course_ID])
    val_u = np.array([u_map[x] for x in validation.Student_ID])
    val_i = np.array([i_map[x] for x in validation.Course_ID])
    pair_u = np.array([u_map[x] for x in pairs.Student_ID])
    pair_i = np.array([i_map[x] for x in pairs.preferred_item])
    pair_j = np.array([i_map[x] for x in pairs.less_preferred_item])
    as_tensor = lambda value, dtype=torch.float32: torch.tensor(value, dtype=dtype, device=device)
    return {
        "train_u": as_tensor(train_u, torch.long),
        "mf_train": as_tensor(mf.to_numpy()[train_u, train_i]),
        "lg_train": as_tensor(lg.to_numpy()[train_u, train_i]),
        "train_targets": as_tensor(train.Rating.to_numpy()),
        "val_u": as_tensor(val_u, torch.long),
        "mf_val": as_tensor(mf.to_numpy()[val_u, val_i]),
        "lg_val": as_tensor(lg.to_numpy()[val_u, val_i]),
        "val_targets": as_tensor(validation.Rating.to_numpy()),
        "pair_u": as_tensor(pair_u, torch.long),
        "mf_pair_diff": as_tensor(z_mf.to_numpy()[pair_u, pair_i] - z_mf.to_numpy()[pair_u, pair_j]),
        "lg_pair_diff": as_tensor(z_lg.to_numpy()[pair_u, pair_i] - z_lg.to_numpy()[pair_u, pair_j]),
        "pair_gap": as_tensor(pairs.rating_gap.to_numpy()),
    }


def train_global(initial_alpha: float, lambda_rank: float, learning_rate: float,
                 maximum_epochs: int, pair_mode: str, tensors: dict,
                 device: torch.device) -> tuple[dict, list[dict]]:
    set_global_seed(SEED, deterministic=True)
    model = GlobalAlpha(initial_alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    best_state, best_rmse, best_epoch, patience = None, float("inf"), 0, 20
    history = []
    for epoch in range(1, maximum_epochs + 1):
        optimizer.zero_grad()
        alpha = model()
        weights = tensors["pair_gap"] if pair_mode == "gap_weighted" else None
        rating_loss, ranking_loss = joint_losses(
            alpha, alpha, tensors["mf_train"], tensors["lg_train"], tensors["train_targets"],
            tensors["mf_pair_diff"], tensors["lg_pair_diff"], weights)
        total = rating_loss + lambda_rank * ranking_loss
        total.backward()
        optimizer.step()
        with torch.no_grad():
            checkpoint_alpha = model()
            val_pred = checkpoint_alpha * tensors["mf_val"] + (1.0 - checkpoint_alpha) * tensors["lg_val"]
            val_rmse = float(torch.sqrt(torch.mean(torch.square(
                val_pred.clamp(1.0, 5.0) - tensors["val_targets"]))).item())
        history.append({"stage": "global", "pair_mode": pair_mode,
                        "initial_alpha": initial_alpha, "lambda_rank": lambda_rank,
                        "learning_rate": learning_rate, "maximum_epochs": maximum_epochs,
                        "epoch": epoch, "alpha": float(checkpoint_alpha.item()),
                        "train_rating_loss": float(rating_loss.item()),
                        "train_ranking_loss": float(ranking_loss.item()),
                        "train_total_loss": float(total.item()), "validation_RMSE_monitor": val_rmse})
        if val_rmse < best_rmse - 1e-12:
            best_rmse, best_epoch, best_state, patience = val_rmse, epoch, copy.deepcopy(model.state_dict()), 20
        else:
            patience -= 1
            if patience == 0:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        best_alpha = float(model().item())
        weights = tensors["pair_gap"] if pair_mode == "gap_weighted" else None
        rating_loss, ranking_loss = joint_losses(
            model(), model(), tensors["mf_train"], tensors["lg_train"], tensors["train_targets"],
            tensors["mf_pair_diff"], tensors["lg_pair_diff"], weights)
    return {"initial_alpha": initial_alpha, "final_alpha": history[-1]["alpha"],
            "best_checkpoint_alpha": best_alpha, "lambda_rank": lambda_rank,
            "learning_rate": learning_rate, "maximum_epochs": maximum_epochs,
            "best_epoch": best_epoch, "epochs_run": epoch, "pair_mode": pair_mode,
            "train_rating_loss": float(rating_loss.item()),
            "train_ranking_loss": float(ranking_loss.item())}, history


def train_user_count(lambda_rank: float, learning_rate: float, maximum_epochs: int,
                     pair_mode: str, tensors: dict, count_feature: torch.Tensor,
                     device: torch.device) -> tuple[dict, pd.Series, list[dict]]:
    set_global_seed(SEED, deterministic=True)
    model = UserCountAlpha().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state, best_rmse, best_epoch, patience, history = None, float("inf"), 0, 20, []
    for epoch in range(1, maximum_epochs + 1):
        optimizer.zero_grad()
        alpha = model(count_feature)
        weights = tensors["pair_gap"] if pair_mode == "gap_weighted" else None
        rating_loss, ranking_loss = joint_losses(
            alpha[tensors["train_u"]], alpha[tensors["pair_u"]],
            tensors["mf_train"], tensors["lg_train"], tensors["train_targets"],
            tensors["mf_pair_diff"], tensors["lg_pair_diff"], weights)
        total = rating_loss + lambda_rank * ranking_loss
        total.backward(); optimizer.step()
        with torch.no_grad():
            checkpoint_alpha = model(count_feature)
            val_alpha = checkpoint_alpha[tensors["val_u"]]
            val_pred = val_alpha * tensors["mf_val"] + (1.0 - val_alpha) * tensors["lg_val"]
            val_rmse = float(torch.sqrt(torch.mean(torch.square(
                val_pred.clamp(1.0, 5.0) - tensors["val_targets"]))).item())
        history.append({"stage": "user_count", "pair_mode": pair_mode,
                        "initial_alpha": np.nan, "lambda_rank": lambda_rank,
                        "learning_rate": learning_rate, "maximum_epochs": maximum_epochs,
                        "epoch": epoch, "alpha": float(checkpoint_alpha.mean().item()),
                        "train_rating_loss": float(rating_loss.item()),
                        "train_ranking_loss": float(ranking_loss.item()),
                        "train_total_loss": float(total.item()), "validation_RMSE_monitor": val_rmse})
        if val_rmse < best_rmse - 1e-12:
            best_rmse, best_epoch, best_state, patience = val_rmse, epoch, copy.deepcopy(model.state_dict()), 20
        else:
            patience -= 1
            if patience == 0: break
    model.load_state_dict(best_state)
    with torch.no_grad(): final_alpha = model(count_feature).cpu().numpy()
    return {"lambda_rank": lambda_rank, "learning_rate": learning_rate,
            "maximum_epochs": maximum_epochs, "best_epoch": best_epoch,
            "epochs_run": epoch, "pair_mode": pair_mode,
            "mean_alpha": float(final_alpha.mean())}, pd.Series(final_alpha), history


def pareto(frame: pd.DataFrame, ceiling: float) -> pd.DataFrame:
    output = []
    for index, row in frame.iterrows():
        dominated_by, dominates = [], []
        for other_index, other in frame.iterrows():
            if index == other_index: continue
            other_wins = (other.RMSE <= row.RMSE and other["NDCG@5"] >= row["NDCG@5"] and
                          (other.RMSE < row.RMSE or other["NDCG@5"] > row["NDCG@5"]))
            row_wins = (row.RMSE <= other.RMSE and row["NDCG@5"] >= other["NDCG@5"] and
                        (row.RMSE < other.RMSE or row["NDCG@5"] > other["NDCG@5"]))
            if other_wins: dominated_by.append(str(other.point))
            if row_wins: dominates.append(str(other.point))
        output.append({**row.to_dict(), "eligible": bool(row.RMSE <= ceiling),
                       "dominated": bool(dominated_by), "pareto_optimal": not dominated_by,
                       "dominated_by": ";".join(dominated_by), "dominates": ";".join(dominates)})
    return pd.DataFrame(output)


def main() -> None:
    started = time.perf_counter()
    reproducibility = set_global_seed(SEED, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog = pd.read_csv(CATALOG, index_col=0).fillna(0)
    users, items = catalog.index.astype(str).tolist(), catalog.columns.astype(str).tolist()
    train = pd.read_csv(SPLITS / "train.csv", dtype={"Student_ID": str, "Course_ID": str})
    validation = pd.read_csv(SPLITS / "validation.csv", dtype={"Student_ID": str, "Course_ID": str})
    pairs = build_ordinal_pairs(train)
    pair_stats = pair_statistics(pairs); verify_pair_statistics(pair_stats)
    mf, lg, reproduction, candidate_checks = reproduce(train, validation, users, items, device)
    z_mf, z_lg = normalization(mf, train, users, items), normalization(lg, train, users, items)
    tensors = tensor_data(train, validation, pairs, mf, lg, z_mf, z_lg, users, items, device)

    fixed_rows = []
    for alpha in FIXED_ALPHAS:
        fixed_rows.append({"point": f"V2-fixed-alpha-{alpha:.2f}", "family": "v2_fixed",
                           "alpha": alpha, **evaluate_fusion(alpha, mf, lg, z_mf, z_lg,
                                                             validation, train, items)})
    fixed = pd.DataFrame(fixed_rows)

    # Preflight loss-scale audit before the objective grid.
    scale_rows = []
    for initial in INITIAL_ALPHAS:
        alpha = torch.tensor(initial, device=device)
        rating_loss, ranking_loss = joint_losses(
            alpha, alpha, tensors["mf_train"], tensors["lg_train"], tensors["train_targets"],
            tensors["mf_pair_diff"], tensors["lg_pair_diff"])
        scale_rows.append((float(rating_loss.item()), float(ranking_loss.item())))
    scale_ratios = [max(a, b) / max(min(a, b), 1e-12) for a, b in scale_rows]
    if max(scale_ratios) > 100:
        raise RuntimeError(f"Extreme unnormalized objective-scale imbalance: {scale_rows}")

    global_rows, history_rows, metric_cache = [], [], {}
    for pair_mode, initial, lambda_rank, learning_rate, maximum_epochs in product(
            PAIR_MODES, INITIAL_ALPHAS, LAMBDAS, LEARNING_RATES, MAX_EPOCHS):
        trained, history = train_global(initial, lambda_rank, learning_rate,
                                        maximum_epochs, pair_mode, tensors, device)
        key = round(trained["best_checkpoint_alpha"], 10)
        if key not in metric_cache:
            metric_cache[key] = evaluate_fusion(trained["best_checkpoint_alpha"], mf, lg,
                                                z_mf, z_lg, validation, train, items)
        global_rows.append({"point": "V2-global", "family": "v2_global", **trained,
                            **metric_cache[key]})
        history_rows.extend(history)
    global_results = pd.DataFrame(global_rows)

    # Baselines are read from validated validation-only artifacts.
    v1 = pd.read_csv(UAF_V1 / "validation_results.csv")
    v1_fixed = v1[v1.method.isin(["Always-MF", "Always-LightGCN", "Fixed-alpha-0.25",
                                  "Fixed-alpha-0.50", "Fixed-alpha-0.75"])].copy()
    v1_fixed["point"] = "V1-" + v1_fixed.method
    v1_fixed["family"] = "existing"
    v1_gates = v1[v1.method.isin(["UAF-count", "UAF-rich"])].sort_values(
        ["method", "RMSE", "NDCG@5"], ascending=[True, True, False]).groupby("method").head(1).copy()
    v1_gates["point"] = "V1-" + v1_gates.method; v1_gates["family"] = "existing"
    routers = pd.read_csv(RESULTS / "sparsity_router_v1/validation_router_results.csv")
    routers = routers[routers.policy.str.startswith("tau=")].copy()
    routers["point"] = "Router-" + routers.policy; routers["family"] = "existing"
    baseline_columns = ["point", "family", "RMSE", "NDCG@5", "Recall@5", "MAE",
                        "Precision@5", "HitRate@5", "Precision@10", "Recall@10",
                        "NDCG@10", "HitRate@10"]
    existing = pd.concat([v1_fixed[baseline_columns], v1_gates[baseline_columns],
                          routers[baseline_columns]], ignore_index=True)
    mf_rmse = float(v1_fixed.loc[v1_fixed.method == "Always-MF", "RMSE"].iloc[0])
    ceiling = 1.10 * mf_rmse

    mse_control = global_results[global_results.lambda_rank == 0]
    positive = global_results[(global_results.lambda_rank > 0) & (global_results.RMSE <= ceiling)]
    best_mse_ndcg = float(mse_control[mse_control.RMSE <= ceiling]["NDCG@5"].max())
    material_positive = positive[positive["NDCG@5"] >= best_mse_ndcg + MATERIAL_NDCG_GAIN]
    run_user_count = not material_positive.empty

    user_count_results = pd.DataFrame(); user_alpha = pd.DataFrame()
    if run_user_count:
        chosen = material_positive.sort_values(
            ["NDCG@5", "Recall@5", "RMSE", "lambda_rank"],
            ascending=[False, False, True, True]).iloc[0]
        feature = np.log1p(build_train_user_features(train).reindex(users).train_interaction_count)
        standardized = ((feature - feature.mean()) / feature.std(ddof=0)).to_numpy()[:, None]
        count_tensor = torch.tensor(standardized, dtype=torch.float32, device=device)
        trained, alpha, history = train_user_count(
            float(chosen.lambda_rank), float(chosen.learning_rate), int(chosen.maximum_epochs),
            str(chosen.pair_mode), tensors, count_tensor, device)
        alpha.index = users
        metrics = evaluate_fusion(alpha, mf, lg, z_mf, z_lg, validation, train, items)
        user_count_results = pd.DataFrame([{"point": "OAF-user-count", "family": "v2_user_count",
                                            **trained, **metrics}])
        user_alpha = pd.DataFrame({"Student_ID": users, "alpha": alpha.values,
                                   "train_interaction_count": build_train_user_features(train).reindex(users).train_interaction_count.astype(int).values})
        history_rows.extend(history)

    best_rows = []
    for (pair_mode, lambda_rank), group in global_results.groupby(["pair_mode", "lambda_rank"]):
        best_rows.append(group.sort_values(["RMSE", "NDCG@5"], ascending=[True, False]).iloc[0])
    pareto_candidates = pd.concat([
        existing,
        fixed[baseline_columns],
        pd.DataFrame(best_rows)[baseline_columns + ["lambda_rank", "initial_alpha",
                                                     "best_checkpoint_alpha", "pair_mode"]],
        user_count_results.reindex(columns=baseline_columns + ["lambda_rank", "pair_mode"]),
    ], ignore_index=True, sort=False)
    pareto_results = pareto(pareto_candidates, ceiling)
    existing_frontier = pareto(existing, ceiling)
    joint_on_frontier = pareto_results[(pareto_results.family == "v2_global") &
                                       (pareto_results.lambda_rank > 0) & pareto_results.pareto_optimal]
    frontier_expanded = not joint_on_frontier.empty

    eligible_joint = global_results[(global_results.lambda_rank > 0) &
                                    (global_results.RMSE <= ceiling)]
    best_joint = eligible_joint.sort_values(
        ["NDCG@5", "Recall@5", "RMSE", "lambda_rank"],
        ascending=[False, False, True, True]).iloc[0]
    best_fixed = fixed[fixed.RMSE <= ceiling].sort_values(
        ["NDCG@5", "Recall@5", "RMSE"], ascending=[False, False, True]).iloc[0]
    previous_best = routers.sort_values(["NDCG@5", "RMSE"], ascending=[False, True]).iloc[0]
    fixed_025 = v1_fixed[v1_fixed.method == "Fixed-alpha-0.25"].iloc[0]
    global_mse_best = mse_control[mse_control.RMSE <= ceiling].sort_values(
        ["NDCG@5", "Recall@5", "RMSE"], ascending=[False, False, True]).iloc[0]

    improves_mse = best_joint["NDCG@5"] > global_mse_best["NDCG@5"]
    improves_fixed = best_joint["NDCG@5"] > fixed_025["NDCG@5"]
    improves_router = best_joint["NDCG@5"] > previous_best["NDCG@5"]
    if eligible_joint.empty or not improves_mse:
        verdict = "NOT SUPPORTED"
    elif frontier_expanded and improves_router:
        verdict = "SUPPORTED"
    else:
        verdict = "SUGGESTIVE"

    gap_best = global_results[(global_results.pair_mode == "gap_weighted") &
                              (global_results.lambda_rank > 0) & (global_results.RMSE <= ceiling)].sort_values(
                                  ["NDCG@5", "RMSE"], ascending=[False, True]).iloc[0]
    unweighted_best = global_results[(global_results.pair_mode == "unweighted") &
                                     (global_results.lambda_rank > 0) & (global_results.RMSE <= ceiling)].sort_values(
                                         ["NDCG@5", "RMSE"], ascending=[False, True]).iloc[0]
    alpha_lambda = global_results.groupby("lambda_rank").best_checkpoint_alpha.mean()
    monotonic_shift = bool(alpha_lambda.is_monotonic_decreasing)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(OUTPUT / "fixed_alpha_validation.csv", index=False)
    global_results.to_csv(OUTPUT / "global_alpha_joint_objective.csv", index=False)
    pd.DataFrame(history_rows).to_csv(OUTPUT / "training_history.csv", index=False)
    pareto_results.to_csv(OUTPUT / "pareto_validation.csv", index=False)
    if run_user_count:
        user_count_results.to_csv(OUTPUT / "user_count_joint_objective.csv", index=False)
        user_alpha.to_csv(OUTPUT / "user_alpha_values.csv", index=False)
    write_json(OUTPUT / "pair_statistics.json", {
        **pair_stats, "source": str(SPLITS / "train.csv"),
        "observed_only": True, "unobserved_negative_sampling": False,
        "matches_existing_feasibility_audit": True,
    })
    selection_pool = pd.concat([fixed.assign(lambda_rank=np.nan, initial_alpha=np.nan,
                                              final_alpha=np.nan, best_checkpoint_alpha=fixed.alpha),
                                global_results, user_count_results], ignore_index=True, sort=False)
    eligible_pool = selection_pool[selection_pool.RMSE <= ceiling].copy()
    eligible_pool["complexity"] = eligible_pool.family.map(
        {"v2_fixed": 0, "v2_global": 1, "v2_user_count": 2})
    selected = eligible_pool.sort_values(
        ["NDCG@5", "Recall@5", "RMSE", "complexity", "lambda_rank"],
        ascending=[False, False, True, True, True]).iloc[0]
    selected_payload = {
        "scope": "validation_only", "test_loaded": False, "seed": SEED,
        "rmse_mf": mf_rmse, "eligibility_ceiling": ceiling,
        "selection_rule": "eligible RMSE <= 1.10*MF; max NDCG@5; ties Recall@5, RMSE, simplicity, lower lambda",
        "selected": {key: (None if pd.isna(value) else value) for key, value in selected.to_dict().items()},
        "best_v2_joint": {key: (None if pd.isna(value) else value) for key, value in best_joint.to_dict().items()},
        "frontier_expanded": frontier_expanded, "user_count_run": run_user_count,
        "material_reentry_threshold_ndcg": MATERIAL_NDCG_GAIN, "verdict": verdict,
    }
    write_json(OUTPUT / "selected_config.json", selected_payload)

    audit = f"""# OAF-V2 evaluation audit

Status: **PASS**

- Seed: 42; deterministic execution requested; device `{device}`.
- Scope: TRAIN and validation only. **TEST WAS LOADED: NO**.
- Frozen MF: `experiments/aciids_2027/run_mf.py:fit_predict`, `{MF_CONFIG}`.
- Frozen LightGCN-RMSE: `experiments/aciids_2027/run_lightgcn.py:fit_predict`, `{LG_CONFIG}`.
- Recovered artifacts: `{RECOVERED / 'mf'}` and `{RECOVERED / 'lightgcn_rmse'}`.
- Ranking-oriented uniform 3-layer LightGCN was located at `experiments/aciids_2027/results/sparsity_aware_lightgcn_v1/validation_policy_results.csv` (policy `A_uniform`); it is a documented reference only and is not a V2 base branch.
- Base validation reproduction maximum differences: MF `{reproduction['mf']:.3g}`, LightGCN `{reproduction['lightgcn_rmse']:.3g}` (required <= `{REPRODUCTION_ATOL}`).
- Recovered candidate alignment differences: MF `{candidate_checks['mf']:.3g}`, LightGCN `{candidate_checks['lightgcn_rmse']:.3g}`; recovered values overwrite replayed validation candidates and are authoritative.
- MF/LightGCN recovered validation artifacts have identical user/item candidate keys, relevance labels, TRAIN exclusions, and ranking-eligible users per `validation_artifact_audit.md`.
- Evaluation uses `shared/evaluator.py:evaluate_predictions`: full 151-course catalog, TRAIN-observed exclusion, relevance rating >=4, binary NDCG, deterministic score-descending then Course-ID ascending order.
- Ordinal supervision contains only strict rating-ordered pairs among TRAIN-observed courses. No validation pair construction, unobserved-as-negative sampling, or test access occurs.
- Raw fusion is used for MAE/RMSE. Per-user z-normalized fusion is used for ordinal loss and ranking metrics only. Moments are fit over each user's TRAIN-unobserved catalog scores and applied to all item scores.
- Preflight unnormalized rating/ranking loss-scale ratios ranged `{min(scale_ratios):.3f}`–`{max(scale_ratios):.3f}`, below the predeclared extreme threshold 100; no post-hoc rescaling was applied.
- Global checkpointing monitors validation RMSE with patience 20 and restores the best checkpoint. Validation labels never enter gradient updates.
"""
    (OUTPUT / "evaluation_audit.md").write_text(audit, encoding="utf-8")

    summary = f"""# Objective-Aligned Fusion V2 — validation only

**Final verdict: {verdict}. TEST WAS LOADED: NO.**

## Pair supervision

TRAIN-only graded ordinal pairs: {pair_stats['total_pairs']:,} from {pair_stats['eligible_users']} users. Gap counts: 1={pair_stats['gap_counts']['1']:,}, 2={pair_stats['gap_counts']['2']:,}, 3={pair_stats['gap_counts']['3']:,}, 4={pair_stats['gap_counts']['4']:,}. These exactly match the prior feasibility audit.

Rating metrics use raw MF/LightGCN fusion. Ranking loss and ranking metrics use separately per-user standardized branch scores. No unobserved course is treated as a negative.

## Main validation points

| Point | alpha/lambda | RMSE | NDCG@5 | Recall@5 | Eligible |
|---|---|---:|---:|---:|:---:|
| MF | alpha=1 | {mf_rmse:.6f} | {v1_fixed.loc[v1_fixed.method == 'Always-MF', 'NDCG@5'].iloc[0]:.6f} | {v1_fixed.loc[v1_fixed.method == 'Always-MF', 'Recall@5'].iloc[0]:.6f} | Yes |
| LightGCN-RMSE | alpha=0 | {v1_fixed.loc[v1_fixed.method == 'Always-LightGCN', 'RMSE'].iloc[0]:.6f} | {v1_fixed.loc[v1_fixed.method == 'Always-LightGCN', 'NDCG@5'].iloc[0]:.6f} | {v1_fixed.loc[v1_fixed.method == 'Always-LightGCN', 'Recall@5'].iloc[0]:.6f} | No |
| Prior router best | {previous_best.policy} | {previous_best.RMSE:.6f} | {previous_best['NDCG@5']:.6f} | {previous_best['Recall@5']:.6f} | {'Yes' if previous_best.RMSE <= ceiling else 'No'} |
| V1 fixed alpha=.25 | alpha=.25 | {fixed_025.RMSE:.6f} | {fixed_025['NDCG@5']:.6f} | {fixed_025['Recall@5']:.6f} | Yes |
| Best V2 fixed | alpha={best_fixed.alpha:.2f} | {best_fixed.RMSE:.6f} | {best_fixed['NDCG@5']:.6f} | {best_fixed['Recall@5']:.6f} | Yes |
| Best global joint | alpha={best_joint.best_checkpoint_alpha:.6f}, lambda={best_joint.lambda_rank:g}, {best_joint.pair_mode} | {best_joint.RMSE:.6f} | {best_joint['NDCG@5']:.6f} | {best_joint['Recall@5']:.6f} | Yes |

## Required answers

1. **lambda_rank > 0 versus MSE-only:** {'Yes, the best eligible joint point improves NDCG@5.' if improves_mse else 'No eligible joint point improves NDCG@5 over the best MSE-only global fusion.'}
2. **10% RMSE constraint:** {'Preserved by the reported best joint point.' if best_joint.RMSE <= ceiling else 'Not preserved.'}
3. **Best fixed alpha:** `{best_fixed.alpha:.2f}` (RMSE {best_fixed.RMSE:.6f}, NDCG@5 {best_fixed['NDCG@5']:.6f}).
4. **Best learned global alpha:** lambda `{best_joint.lambda_rank:g}`, initial alpha `{best_joint.initial_alpha:g}`, checkpoint alpha `{best_joint.best_checkpoint_alpha:.6f}`, `{best_joint.pair_mode}` (RMSE {best_joint.RMSE:.6f}, NDCG@5 {best_joint['NDCG@5']:.6f}).
5. **Versus UAF-V1 gates:** best joint NDCG@5 {best_joint['NDCG@5']:.6f} versus UAF-count {v1_gates.loc[v1_gates.method == 'UAF-count', 'NDCG@5'].iloc[0]:.6f} and UAF-rich {v1_gates.loc[v1_gates.method == 'UAF-rich', 'NDCG@5'].iloc[0]:.6f}.
6. **Versus fixed alpha=.25:** {'Improves.' if improves_fixed else 'Does not improve.'}
7. **Versus previous hard router:** {'Improves.' if improves_router else 'Does not improve the best router NDCG@5.'}
8. **Existing Pareto frontier expanded:** {'Formally YES, but relative to the V2 MSE-only control the change must be checked for numerical rather than meaningful metric improvement.' if frontier_expanded else 'NO.'}
9. **Larger lambda shifts toward LightGCN:** {'Mean checkpoint alpha is monotonically non-increasing, but the displacement must be judged by its magnitude and may be negligible.' if monotonic_shift else 'No systematic monotonic shift is supported across the full grid.'} This is a controlled within-grid association, not a causal generalization.
10. **Gap weighting:** best gap-weighted NDCG@5 {gap_best['NDCG@5']:.6f} versus unweighted {unweighted_best['NDCG@5']:.6f}; {'gap weighting helped.' if gap_best['NDCG@5'] > unweighted_best['NDCG@5'] else 'gap weighting did not help.'}
11. **OAF-user-count:** {'Run because the predeclared >=0.005 NDCG material-improvement trigger was met.' if run_user_count else 'Not run because V2-B did not meet the predeclared material-improvement trigger.'}
12. **Verdict:** **{verdict}** under the predeclared validation-only success criteria.

## Scientific interpretation

The result tests objective alignment for a one-dimensional global mixture. Any observed lambda–alpha relationship is specific to these frozen branch scores, this split, and the declared loss scaling. No test claim is made.
"""
    (OUTPUT / "summary.md").write_text(summary, encoding="utf-8")

    print(f"""MF validation:
RMSE = {mf_rmse:.6f}
NDCG@5 = {v1_fixed.loc[v1_fixed.method == 'Always-MF', 'NDCG@5'].iloc[0]:.6f}

LightGCN-RMSE validation:
RMSE = {v1_fixed.loc[v1_fixed.method == 'Always-LightGCN', 'RMSE'].iloc[0]:.6f}
NDCG@5 = {v1_fixed.loc[v1_fixed.method == 'Always-LightGCN', 'NDCG@5'].iloc[0]:.6f}

Previous router best:
RMSE = {previous_best.RMSE:.6f}
NDCG@5 = {previous_best['NDCG@5']:.6f}

UAF-V1 fixed alpha=0.25:
RMSE = {fixed_025.RMSE:.6f}
NDCG@5 = {fixed_025['NDCG@5']:.6f}

Best V2 fixed alpha:
alpha = {best_fixed.alpha:.2f}
RMSE = {best_fixed.RMSE:.6f}
NDCG@5 = {best_fixed['NDCG@5']:.6f}

Best V2 global joint-objective:
lambda_rank = {best_joint.lambda_rank:g}
initial alpha = {best_joint.initial_alpha:g}
final alpha = {best_joint.final_alpha:.6f}
RMSE = {best_joint.RMSE:.6f}
NDCG@5 = {best_joint['NDCG@5']:.6f}
Recall@5 = {best_joint['Recall@5']:.6f}
eligible = YES

Pareto frontier expanded:
{'YES' if frontier_expanded else 'NO'}

OAF-user-count run:
{'YES' if run_user_count else 'NO'}

Final verdict:
{verdict}

TEST WAS LOADED:
NO""")


if __name__ == "__main__":
    main()
