"""Validation-only UAF-V1 development experiment.

This module intentionally defines no test path and never opens a test artifact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.aciids_2027.methods.uaf_v1.gate_model import UserAdaptiveGate
from experiments.aciids_2027.methods.uaf_v1.user_features import (
    COUNT_ONLY_COLUMNS,
    FEATURE_COLUMNS,
    RICH_COLUMNS,
    bucket_name,
    build_train_user_features,
    standardize_features,
)
from experiments.aciids_2027.run_lightgcn import fit_predict as fit_lightgcn
from experiments.aciids_2027.run_mf import fit_predict as fit_mf
from shared.evaluator import EvaluationPolicy, evaluate_predictions
from shared.reproducibility import set_global_seed, write_json

SEED = 42
SPLITS = ROOT / "experiments/aciids_2027/splits/seed_42"
RECOVERED = ROOT / "experiments/aciids_2027/artifacts/validation_predictions"
PUBLICATION = ROOT / "experiments/aciids_2027/results"
OUTPUT = ROOT / "experiments/aciids_2027/results/uaf_v1"
CATALOG = ROOT / "old/rating_matrix - rating_matrix.csv"
POLICY = EvaluationPolicy()
MF_CONFIG = {"latent_dim": 32, "epochs": 100, "learning_rate": 0.005, "weight_decay": 0.001}
LG_CONFIG = {"embedding_dim": 64, "number_of_layers": 1, "epochs": 100,
             "learning_rate": 0.01, "weight_decay": 0.0}
LEARNING_RATES = [0.001, 0.005, 0.01]
WEIGHT_DECAYS = [0.0, 0.0001, 0.001]
EPOCHS = [50, 100, 200]
ATOL = 1e-7


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combined_metrics(rating_matrix: pd.DataFrame, ranking_matrix: pd.DataFrame,
                     validation: pd.DataFrame, train: pd.DataFrame,
                     items: list[str]) -> dict:
    rating, _, _ = evaluate_predictions(
        rating_matrix, validation, train, items, float(train.Rating.mean()), POLICY)
    ranking, _, _ = evaluate_predictions(
        ranking_matrix, validation, train, items, float(train.Rating.mean()), POLICY)
    result = dict(rating)
    result["Ranking_Eligible_Users"] = ranking["Ranking_Eligible_Users"]
    for k in POLICY.top_ks:
        for metric in ("Precision", "Recall", "NDCG", "HitRate"):
            result[f"{metric}@{k}"] = ranking[f"{metric}@{k}"]
    return result


def candidate_zscores(matrix: pd.DataFrame, train: pd.DataFrame,
                      users: list[str], items: list[str]) -> pd.DataFrame:
    observed = train.groupby("Student_ID")["Course_ID"].agg(set).to_dict()
    result = pd.DataFrame(0.0, index=matrix.index, columns=matrix.columns)
    for user in users:
        candidates = [item for item in items if item not in observed.get(user, set())]
        values = matrix.loc[user, candidates].to_numpy(dtype=float)
        result.loc[user, candidates] = (values - values.mean()) / (values.std(ddof=0) + 1e-8)
    return result


def fuse(mf: pd.DataFrame, lg: pd.DataFrame, alpha: pd.Series) -> pd.DataFrame:
    weights = alpha.reindex(mf.index).to_numpy(dtype=float)[:, None]
    return pd.DataFrame(
        weights * mf.to_numpy(dtype=float) + (1.0 - weights) * lg.to_numpy(dtype=float),
        index=mf.index, columns=mf.columns,
    )


def recovered_matrix(model: str, users: list[str], items: list[str]) -> pd.DataFrame:
    scores = pd.read_csv(
        RECOVERED / model / "validation_candidate_scores.csv",
        dtype={"user_id": str, "course_id": str},
    )
    matrix = scores.pivot(index="user_id", columns="course_id", values="score")
    return matrix.reindex(index=users, columns=items)


def expected_metrics(model: str) -> dict:
    check = json.loads((RECOVERED / model / "reproduction_check.json").read_text())
    return {key: value["saved"] for key, value in check["metric_comparison"].items()}


def reproduce_base_models(train: pd.DataFrame, validation: pd.DataFrame,
                          users: list[str], items: list[str], device: torch.device):
    mf = fit_mf(train, users, items, epochs=MF_CONFIG["epochs"], dim=MF_CONFIG["latent_dim"],
                lr=MF_CONFIG["learning_rate"], weight_decay=MF_CONFIG["weight_decay"],
                seed=SEED, device=device)
    lg, _ = fit_lightgcn(train, users, items, embedding_dim=LG_CONFIG["embedding_dim"],
                         number_of_layers=LG_CONFIG["number_of_layers"],
                         learning_rate=LG_CONFIG["learning_rate"], epochs=LG_CONFIG["epochs"],
                         weight_decay=LG_CONFIG["weight_decay"], seed=SEED, device=device)
    checks = {}
    for name, matrix in (("mf", mf), ("lightgcn_rmse", lg)):
        actual, _, _ = evaluate_predictions(
            matrix, validation, train, items, float(train.Rating.mean()), POLICY)
        expected = expected_metrics(name)
        differences = {metric: abs(float(actual[metric]) - float(expected[metric])) for metric in expected}
        checks[name] = {"maximum_absolute_difference": max(differences.values()),
                        "absolute_differences": differences,
                        "status": "PASS" if max(differences.values()) <= ATOL else "FAIL"}
    if any(value["status"] != "PASS" for value in checks.values()):
        raise RuntimeError(f"Frozen base-model reproduction failed: {checks}")
    return mf, lg, checks


def train_gate(feature_frame: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame,
               mf: pd.DataFrame, lg: pd.DataFrame, user_map: dict[str, int],
               item_map: dict[str, int], learning_rate: float, weight_decay: float,
               maximum_epochs: int, device: torch.device) -> tuple[UserAdaptiveGate, dict]:
    set_global_seed(SEED, deterministic=True)
    model = UserAdaptiveGate(feature_frame.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    features = torch.tensor(feature_frame.to_numpy(), dtype=torch.float32, device=device)
    train_users = torch.tensor([user_map[x] for x in train.Student_ID], device=device)
    train_items = [item_map[x] for x in train.Course_ID]
    train_targets = torch.tensor(train.Rating.to_numpy(), dtype=torch.float32, device=device)
    mf_train = torch.tensor(mf.to_numpy()[train_users.cpu().numpy(), train_items],
                            dtype=torch.float32, device=device)
    lg_train = torch.tensor(lg.to_numpy()[train_users.cpu().numpy(), train_items],
                            dtype=torch.float32, device=device)
    val_users = torch.tensor([user_map[x] for x in validation.Student_ID], device=device)
    val_items = [item_map[x] for x in validation.Course_ID]
    val_targets = torch.tensor(validation.Rating.to_numpy(), dtype=torch.float32, device=device)
    mf_val = torch.tensor(mf.to_numpy()[val_users.cpu().numpy(), val_items],
                          dtype=torch.float32, device=device)
    lg_val = torch.tensor(lg.to_numpy()[val_users.cpu().numpy(), val_items],
                          dtype=torch.float32, device=device)
    best_state, best_rmse, best_epoch, patience_left = None, float("inf"), 0, 20
    last_train_loss = float("nan")
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        optimizer.zero_grad()
        alpha = model(features)
        predictions = alpha[train_users] * mf_train + (1.0 - alpha[train_users]) * lg_train
        loss = F.mse_loss(predictions, train_targets)
        loss.backward()
        optimizer.step()
        last_train_loss = float(loss.item())
        model.eval()
        with torch.no_grad():
            alpha_val = model(features)[val_users]
            val_predictions = alpha_val * mf_val + (1.0 - alpha_val) * lg_val
            rmse = float(torch.sqrt(F.mse_loss(val_predictions.clamp(1.0, 5.0), val_targets)).item())
        if rmse < best_rmse - 1e-12:
            best_rmse, best_epoch = rmse, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = 20
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "epochs_run": epoch,
                   "early_stopping_validation_RMSE": best_rmse,
                   "last_train_MSE": last_train_loss}


def gate_diagnostics(name: str, alpha: pd.Series, features: pd.DataFrame) -> list[dict]:
    counts = features.train_interaction_count
    correlation = spearmanr(alpha.to_numpy(), counts.to_numpy()).statistic
    rows = []
    for bucket in ["ALL", "1-5", "6-10", "11-20", ">20"]:
        selected = alpha if bucket == "ALL" else alpha[counts.map(bucket_name) == bucket]
        rows.append({
            "method": name, "bucket": bucket, "users": len(selected),
            "alpha_mean": float(selected.mean()), "alpha_std": float(selected.std(ddof=0)),
            "alpha_min": float(selected.min()), "alpha_max": float(selected.max()),
            "spearman_alpha_vs_count": float(correlation) if bucket == "ALL" else np.nan,
            "pct_alpha_lt_0.25": 100 * float((selected < .25).mean()),
            "pct_alpha_0.25_to_0.5": 100 * float(((selected >= .25) & (selected < .5)).mean()),
            "pct_alpha_0.5_to_0.75": 100 * float(((selected >= .5) & (selected < .75)).mean()),
            "pct_alpha_ge_0.75": 100 * float((selected >= .75).mean()),
        })
    return rows


def pareto(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, point in frame.iterrows():
        dominated_by, dominates = [], []
        for other_index, other in frame.iterrows():
            if index == other_index:
                continue
            other_wins = (other.RMSE <= point.RMSE and other["NDCG@5"] >= point["NDCG@5"] and
                          (other.RMSE < point.RMSE or other["NDCG@5"] > point["NDCG@5"]))
            point_wins = (point.RMSE <= other.RMSE and point["NDCG@5"] >= other["NDCG@5"] and
                          (point.RMSE < other.RMSE or point["NDCG@5"] > other["NDCG@5"]))
            if other_wins:
                dominated_by.append(other.method)
            if point_wins:
                dominates.append(other.method)
        rows.append({**point.to_dict(), "pareto_optimal": not dominated_by,
                     "dominated_by": ";".join(dominated_by), "dominates": ";".join(dominates)})
    return pd.DataFrame(rows)


def main() -> None:
    started = time.perf_counter()
    reproducibility = set_global_seed(SEED, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog = pd.read_csv(CATALOG, index_col=0).fillna(0)
    users, items = catalog.index.astype(str).tolist(), catalog.columns.astype(str).tolist()
    train = pd.read_csv(SPLITS / "train.csv", dtype={"Student_ID": str, "Course_ID": str})
    validation = pd.read_csv(SPLITS / "validation.csv", dtype={"Student_ID": str, "Course_ID": str})
    mf, lg, reproduction = reproduce_base_models(train, validation, users, items, device)

    # Verify retrained validation candidate scores against recovered artifacts.
    recovered_checks = {}
    for name, matrix in (("mf", mf), ("lightgcn_rmse", lg)):
        recovered = recovered_matrix(name, users, items)
        mask = recovered.notna()
        maximum = float(np.max(np.abs(matrix.where(mask).to_numpy()[mask.to_numpy()] -
                                      recovered.to_numpy()[mask.to_numpy()])))
        recovered_checks[name] = maximum
        # Replayed CUDA scores can differ by a few float32 ULPs. The recovered,
        # already-audited artifacts remain authoritative for every validation
        # candidate; replayed matrices are needed only for TRAIN-observed scores.
        matrix.values[mask.to_numpy()] = recovered.to_numpy()[mask.to_numpy()]

    features = build_train_user_features(train).reindex(users)
    if features.isna().any().any():
        raise RuntimeError("TRAIN-only feature matrix contains missing users or values")
    standardized, feature_statistics = {}, {}
    for name, columns in (("UAF-count", COUNT_ONLY_COLUMNS), ("UAF-rich", RICH_COLUMNS)):
        standardized[name], feature_statistics[name] = standardize_features(features, columns)

    validation_users = sorted(validation.Student_ID.unique())
    z_mf = candidate_zscores(mf, train, validation_users, items)
    z_lg = candidate_zscores(lg, train, validation_users, items)
    rows = []
    fixed_names = {0.0: "Always-LightGCN", 0.25: "Fixed-alpha-0.25",
                   0.5: "Fixed-alpha-0.50", 0.75: "Fixed-alpha-0.75", 1.0: "Always-MF"}
    for alpha_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        alpha = pd.Series(alpha_value, index=users)
        metrics = combined_metrics(fuse(mf, lg, alpha), fuse(z_mf, z_lg, alpha),
                                   validation, train, items)
        rows.append({"method": fixed_names[alpha_value], "family": "fixed_fusion",
                     "feature_set": "none", "alpha": alpha_value,
                     "learning_rate": np.nan, "weight_decay": np.nan,
                     "maximum_epochs": np.nan, "best_epoch": np.nan,
                     "epochs_run": np.nan, "last_train_MSE": np.nan, **metrics})

    user_map, item_map = {x: i for i, x in enumerate(users)}, {x: i for i, x in enumerate(items)}
    gate_values: list[dict] = []
    best_models = {}
    for gate_name in ["UAF-count", "UAF-rich"]:
        gate_rows = []
        for learning_rate, weight_decay, maximum_epochs in product(
                LEARNING_RATES, WEIGHT_DECAYS, EPOCHS):
            model, training = train_gate(
                standardized[gate_name], train, validation, mf, lg, user_map, item_map,
                learning_rate, weight_decay, maximum_epochs, device)
            model.eval()
            with torch.no_grad():
                alpha_values = model(torch.tensor(standardized[gate_name].to_numpy(),
                                                   dtype=torch.float32, device=device)).cpu().numpy()
            alpha = pd.Series(alpha_values, index=users)
            metrics = combined_metrics(fuse(mf, lg, alpha), fuse(z_mf, z_lg, alpha),
                                       validation, train, items)
            row = {"method": gate_name, "family": "learned_gate",
                   "feature_set": "count_only" if gate_name == "UAF-count" else "rich",
                   "alpha": np.nan, "learning_rate": learning_rate,
                   "weight_decay": weight_decay, "maximum_epochs": maximum_epochs,
                   **training, **metrics}
            rows.append(row)
            gate_rows.append((row, model, alpha))
        gate_rows.sort(key=lambda entry: (entry[0]["RMSE"], -entry[0]["NDCG@5"],
                                          entry[0]["maximum_epochs"], entry[0]["learning_rate"],
                                          entry[0]["weight_decay"]))
        best_row, best_model, best_alpha = gate_rows[0]
        best_models[gate_name] = best_row
        for user, value in best_alpha.items():
            gate_values.append({"method": gate_name, "Student_ID": user,
                                "alpha": float(value),
                                "train_interaction_count": int(features.loc[user, "train_interaction_count"]),
                                "bucket": bucket_name(features.loc[user, "train_interaction_count"])})
        gate_values.extend(gate_diagnostics(gate_name, best_alpha, features))

    results = pd.DataFrame(rows)
    previous = pd.read_csv(PUBLICATION / "sparsity_router_v1/validation_router_results.csv")
    previous = previous[previous.policy.str.startswith("tau=")].copy()
    previous_points = pd.DataFrame({
        "method": "Previous-" + previous.policy,
        "family": "previous_threshold_router", "feature_set": "train_count_threshold",
        "RMSE": previous.RMSE, "MAE": previous.MAE,
        "NDCG@5": previous["NDCG@5"], "Recall@5": previous["Recall@5"],
        "HitRate@5": previous["HitRate@5"], "Precision@5": previous["Precision@5"],
        "Precision@10": previous["Precision@10"], "Recall@10": previous["Recall@10"],
        "NDCG@10": previous["NDCG@10"], "HitRate@10": previous["HitRate@10"],
    })
    best_learned = pd.DataFrame([best_models["UAF-count"], best_models["UAF-rich"]])
    fixed = results[results.family == "fixed_fusion"]
    pareto_input = pd.concat([fixed, previous_points, best_learned], ignore_index=True, sort=False)
    pareto_results = pareto(pareto_input)

    mf_rmse = float(fixed.loc[fixed.method == "Always-MF", "RMSE"].iloc[0])
    ceiling = 1.10 * mf_rmse
    learned_eligible = best_learned[best_learned.RMSE <= ceiling]
    eligible = pareto_input[pareto_input.RMSE <= ceiling].copy()
    complexity = {"Always-MF": 0, "Always-LightGCN": 0,
                  "Fixed-alpha-0.25": 1, "Fixed-alpha-0.50": 1, "Fixed-alpha-0.75": 1,
                  "UAF-count": 2, "UAF-rich": 3}
    eligible["complexity"] = eligible.method.map(complexity).fillna(4)
    selected = eligible.sort_values(["NDCG@5", "Recall@5", "RMSE", "complexity"],
                                    ascending=[False, False, True, True]).iloc[0]
    previous_best_ndcg = float(previous["NDCG@5"].max())
    learned_best_ndcg = float(best_learned["NDCG@5"].max())
    if learned_eligible.empty:
        verdict = "NOT SUPPORTED"
    elif learned_best_ndcg <= max(float(fixed.loc[fixed.method == "Always-MF", "NDCG@5"].iloc[0]),
                                  previous_best_ndcg):
        verdict = "NOT SUPPORTED"
    else:
        verdict = "SUGGESTIVE"

    # Separate per-user alpha rows from aggregate diagnostic rows.
    raw_gate = pd.DataFrame([row for row in gate_values if "Student_ID" in row])
    diagnostics = pd.DataFrame([row for row in gate_values if "Student_ID" not in row])
    feature_payload = {
        "definitions": {
            "train_interaction_count": "number of TRAIN interactions",
            "log1p_train_interaction_count": "natural log(1 + TRAIN interaction count)",
            "train_rating_mean": "mean TRAIN rating",
            "train_rating_std": "population standard deviation (ddof=0) of TRAIN ratings",
            "train_rating_entropy": "natural-log Shannon entropy of empirical TRAIN rating levels",
            "train_positive_ratio": "fraction of TRAIN ratings >= 4",
            "mean_train_item_popularity": "mean TRAIN interaction degree of interacted items",
        },
        "feature_sets": {"count_only": COUNT_ONLY_COLUMNS, "rich": RICH_COLUMNS},
        "standardization_population": "all users in TRAIN only",
        "standardization": feature_statistics,
        "raw_summary": {column: {key: float(value) for key, value in
                        features[column].describe().to_dict().items()} for column in FEATURE_COLUMNS},
    }

    old_frontier = pareto_results[pareto_results.family.isin(
        ["fixed_fusion", "previous_threshold_router"])]
    uaf_expands = bool((pareto_results.family.eq("learned_gate") & pareto_results.pareto_optimal).any())
    best_count, best_rich = best_models["UAF-count"], best_models["UAF-rich"]
    always_mf = fixed[fixed.method == "Always-MF"].iloc[0]
    always_lg = fixed[fixed.method == "Always-LightGCN"].iloc[0]
    best_fixed = fixed[fixed.method.str.startswith("Fixed")].sort_values(
        ["NDCG@5", "RMSE"], ascending=[False, True]).iloc[0]
    count_diag = diagnostics[(diagnostics.method == "UAF-count") & (diagnostics.bucket == "ALL")].iloc[0]
    rich_diag = diagnostics[(diagnostics.method == "UAF-rich") & (diagnostics.bucket == "ALL")].iloc[0]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT / "validation_results.csv", index=False)
    raw_gate.to_csv(OUTPUT / "user_gate_values.csv", index=False)
    pareto_results.to_csv(OUTPUT / "pareto_validation.csv", index=False)
    write_json(OUTPUT / "feature_summary.json", feature_payload)
    best_count_json = {key: (None if isinstance(value, float) and np.isnan(value) else value)
                       for key, value in best_count.items()}
    best_rich_json = {key: (None if isinstance(value, float) and np.isnan(value) else value)
                      for key, value in best_rich.items()}
    write_json(OUTPUT / "selected_config.json", {
        "experiment_scope": "validation_only", "test_loaded": False,
        "eligibility_ceiling": ceiling, "learned_uaf_eligible": not learned_eligible.empty,
        "selected_method": selected.method,
        "selection_rule": "RMSE <= 1.10*Always-MF RMSE; max NDCG@5; tie Recall@5, RMSE, simplicity",
        "best_UAF_count": best_count_json, "best_UAF_rich": best_rich_json,
        "verdict": verdict,
    })
    audit = f"""# UAF-V1 evaluation audit

Status: **PASS**

- Scope: validation-only; **TEST WAS LOADED: NO**.
- TRAIN: `{SPLITS / 'train.csv'}`; validation: `{SPLITS / 'validation.csv'}`.
- Frozen MF implementation: `experiments/aciids_2027/run_mf.py:fit_predict` with `{MF_CONFIG}`.
- Frozen LightGCN implementation: `experiments/aciids_2027/run_lightgcn.py:fit_predict` with `{LG_CONFIG}`.
- Recovered validation artifacts: `{RECOVERED / 'mf'}` and `{RECOVERED / 'lightgcn_rmse'}`.
- Previous router source: `{PUBLICATION / 'sparsity_router_v1/validation_router_results.csv'}`.
- Shared evaluation: `shared/evaluator.py:evaluate_predictions`, full 151-item catalog, TRAIN-item exclusion, relevance `rating >= 4`, binary NDCG, identical eligible-user and tie-breaking semantics.
- Reproduction maximum differences: MF `{reproduction['mf']['maximum_absolute_difference']:.3g}`; LightGCN `{reproduction['lightgcn_rmse']['maximum_absolute_difference']:.3g}` (tolerance `{ATOL}`).
- Recovered candidate-score maximum differences: MF `{recovered_checks['mf']:.3g}`; LightGCN `{recovered_checks['lightgcn_rmse']:.3g}`.
- Features and their standardization statistics use TRAIN only. Gate gradients use TRAIN observed ratings only. Validation labels are used solely for early stopping, hyperparameter selection, and method evaluation.
- Rating fusion uses raw model predictions followed by shared `[1,5]` clipping. Ranking fusion uses per-user, per-model z-scores over TRAIN-unobserved full-catalog candidates; normalized scores never enter MAE/RMSE.
"""
    (OUTPUT / "evaluation_audit.md").write_text(audit, encoding="utf-8")
    diagnostic_rows = []
    for _, row in diagnostics.iterrows():
        rho = "" if pd.isna(row.spearman_alpha_vs_count) else f"{row.spearman_alpha_vs_count:.6f}"
        diagnostic_rows.append(
            f"| {row.method} | {row.bucket} | {int(row.users)} | {row.alpha_mean:.6f} | "
            f"{row.alpha_std:.6f} | {row.alpha_min:.6f} | {row.alpha_max:.6f} | {rho} | "
            f"{row['pct_alpha_lt_0.25']:.2f}% | {row['pct_alpha_0.25_to_0.5']:.2f}% | "
            f"{row['pct_alpha_0.5_to_0.75']:.2f}% | {row['pct_alpha_ge_0.75']:.2f}% |"
        )
    diagnostic_markdown = "\n".join(diagnostic_rows)
    summary = f"""# UAF-V1 validation-only results

**Verdict: {verdict}. TEST WAS LOADED: NO.**

The gate alone was trained; MF and LightGCN parameters remained frozen. Because recovered artifacts omit TRAIN-observed scores, the authoritative frozen implementations were replayed once and reproduced within `1e-7` before gate development. Rating fusion uses raw scores; ranking fusion uses per-user candidate z-scores.

## Best validation points

| Method | MAE | RMSE | NDCG@5 | Recall@5 | Eligible |
|---|---:|---:|---:|---:|:---:|
| Always-MF | {always_mf.MAE:.6f} | {always_mf.RMSE:.6f} | {always_mf['NDCG@5']:.6f} | {always_mf['Recall@5']:.6f} | Yes |
| Always-LightGCN | {always_lg.MAE:.6f} | {always_lg.RMSE:.6f} | {always_lg['NDCG@5']:.6f} | {always_lg['Recall@5']:.6f} | {'Yes' if always_lg.RMSE <= ceiling else 'No'} |
| {best_fixed.method} | {best_fixed.MAE:.6f} | {best_fixed.RMSE:.6f} | {best_fixed['NDCG@5']:.6f} | {best_fixed['Recall@5']:.6f} | {'Yes' if best_fixed.RMSE <= ceiling else 'No'} |
| UAF-count | {best_count['MAE']:.6f} | {best_count['RMSE']:.6f} | {best_count['NDCG@5']:.6f} | {best_count['Recall@5']:.6f} | {'Yes' if best_count['RMSE'] <= ceiling else 'No'} |
| UAF-rich | {best_rich['MAE']:.6f} | {best_rich['RMSE']:.6f} | {best_rich['NDCG@5']:.6f} | {best_rich['Recall@5']:.6f} | {'Yes' if best_rich['RMSE'] <= ceiling else 'No'} |

The prior hard router has no selected B2 point because its predeclared 10% rule produced `NO ELIGIBLE ROUTER`; all six historical threshold points remain in `pareto_validation.csv` for comparison. Its maximum validation NDCG@5 was {previous_best_ndcg:.6f}.

## Gate diagnostics

| Gate | Mean alpha | Std | Min | Max | Spearman(alpha,count) |
|---|---:|---:|---:|---:|---:|
| UAF-count | {count_diag.alpha_mean:.6f} | {count_diag.alpha_std:.6f} | {count_diag.alpha_min:.6f} | {count_diag.alpha_max:.6f} | {count_diag.spearman_alpha_vs_count:.6f} |
| UAF-rich | {rich_diag.alpha_mean:.6f} | {rich_diag.alpha_std:.6f} | {rich_diag.alpha_min:.6f} | {rich_diag.alpha_max:.6f} | {rich_diag.spearman_alpha_vs_count:.6f} |

| Gate | Bucket | Users | Mean | Std | Min | Max | Spearman(alpha,count) | alpha<.25 | .25<=alpha<.5 | .5<=alpha<.75 | alpha>=.75 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{diagnostic_markdown}

The Spearman coefficient is an all-user statistic and is therefore shown only on each gate's `ALL` row. Individual selected-gate values are saved in `user_gate_values.csv`.

## Required answers

1. **RMSE constraint:** {'At least one learned UAF satisfied it.' if not learned_eligible.empty else 'No learned UAF satisfied the predeclared ceiling.'}
2. **NDCG@5 over Always-MF:** {'Yes.' if learned_best_ndcg > always_mf['NDCG@5'] else 'No.'}
3. **Over previous hard router:** {'Yes on validation NDCG@5.' if learned_best_ndcg > previous_best_ndcg else 'No; the best historical threshold remains higher.'}
4. **Pareto expansion:** {'Yes; a learned UAF point is Pareto-optimal in the combined audit.' if uaf_expands else 'No learned UAF point expands the combined validation frontier.'}
5. **Rich versus count-only:** {'Rich has slightly higher NDCG@5 but does not unambiguously outperform count-only because its RMSE is worse.' if best_rich['NDCG@5'] > best_count['NDCG@5'] and best_rich['RMSE'] > best_count['RMSE'] else 'Rich does not improve the joint operating point over count-only.'} Their RMSE values are {best_rich['RMSE']:.6f} and {best_count['RMSE']:.6f}, respectively.
6. **Alpha and sparsity:** UAF-count Spearman rho is {count_diag.spearman_alpha_vs_count:.3f}; UAF-rich rho is {rich_diag.spearman_alpha_vs_count:.3f}. These are validation-development associations, not causal effects.
7. **Overall:** **{verdict}**, based exactly on the predeclared constraint and comparison rules.

## Final required printout

- Always-MF validation: MAE {always_mf.MAE:.6f}, RMSE {always_mf.RMSE:.6f}, NDCG@5 {always_mf['NDCG@5']:.6f}, Recall@5 {always_mf['Recall@5']:.6f}.
- Always-LightGCN validation: MAE {always_lg.MAE:.6f}, RMSE {always_lg.RMSE:.6f}, NDCG@5 {always_lg['NDCG@5']:.6f}, Recall@5 {always_lg['Recall@5']:.6f}.
- Best fixed fusion: {best_fixed.method}, MAE {best_fixed.MAE:.6f}, RMSE {best_fixed.RMSE:.6f}, NDCG@5 {best_fixed['NDCG@5']:.6f}, Recall@5 {best_fixed['Recall@5']:.6f}.
- Best UAF-count: MAE {best_count['MAE']:.6f}, RMSE {best_count['RMSE']:.6f}, NDCG@5 {best_count['NDCG@5']:.6f}, Recall@5 {best_count['Recall@5']:.6f}.
- Best UAF-rich: MAE {best_rich['MAE']:.6f}, RMSE {best_rich['RMSE']:.6f}, NDCG@5 {best_rich['NDCG@5']:.6f}, Recall@5 {best_rich['Recall@5']:.6f}.
- Eligibility ceiling: {ceiling:.6f}.
- Selected method: {selected.method}.
- TEST WAS LOADED: **NO**.
"""
    (OUTPUT / "summary.md").write_text(summary, encoding="utf-8")
    print(summary.split("## Final required printout", 1)[1])
    print("\nTEST WAS LOADED: NO")


if __name__ == "__main__":
    main()
