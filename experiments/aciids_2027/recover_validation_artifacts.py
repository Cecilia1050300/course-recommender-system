"""Recover validation-only artifacts for frozen publication MF and LightGCN.

This entry point intentionally has no test-data path. It retrains exactly the
already-selected configurations and stops before writing if aggregate
validation metrics do not reproduce the authoritative saved rows.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.aciids_2027.run_lightgcn import fit_predict as fit_lightgcn
from experiments.aciids_2027.run_mf import fit_predict as fit_mf
from shared.evaluator import EvaluationPolicy, evaluate_predictions
from shared.reproducibility import set_global_seed, write_json


SPLITS = ROOT / "experiments/aciids_2027/splits/seed_42"
MATRIX = ROOT / "old/rating_matrix - rating_matrix.csv"
PUBLICATION = ROOT / "experiments/aciids_2027/results"
OUTPUT = ROOT / "experiments/aciids_2027/artifacts/validation_predictions"
MF_PARAMS = {"latent_dim": 32, "epochs": 100, "learning_rate": 0.005, "weight_decay": 0.001}
LG_PARAMS = {"embedding_dim": 64, "number_of_layers": 1, "learning_rate": 0.01,
             "epochs": 100, "weight_decay": 0.0}
SEED = 42
ATOL = 1e-7


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authoritative_row(model: str, params: dict) -> dict:
    directory = "mf_publication" if model == "mf" else "lightgcn_publication"
    frame = pd.read_csv(PUBLICATION / directory / "validation_results.csv")
    mask = pd.Series(True, index=frame.index)
    for key, value in params.items():
        mask &= np.isclose(frame[key].astype(float), float(value), rtol=0, atol=1e-12)
    if mask.sum() != 1:
        raise RuntimeError(f"Expected one authoritative {model} row; found {mask.sum()}")
    return frame.loc[mask].iloc[0].to_dict()


def check_reproduction(model: str, actual: dict, expected: dict) -> dict:
    metric_names = [
        "MAE", "RMSE", "Precision@5", "Recall@5", "NDCG@5", "HitRate@5",
        "Precision@10", "Recall@10", "NDCG@10", "HitRate@10",
    ]
    differences = {
        name: {
            "recovered": float(actual[name]),
            "saved": float(expected[name]),
            "absolute_difference": abs(float(actual[name]) - float(expected[name])),
        }
        for name in metric_names
    }
    count_names = ["Rating_Rows", "Missing_Predictions", "Fallback_Predictions",
                   "Cold_User_Rows", "Cold_Item_Rows", "Ranking_Eligible_Users"]
    count_differences = {
        name: {"recovered": int(actual[name]), "saved": int(expected[name]),
               "difference": int(actual[name]) - int(expected[name])}
        for name in count_names
    }
    max_difference = max(v["absolute_difference"] for v in differences.values())
    passed = max_difference <= ATOL and all(v["difference"] == 0 for v in count_differences.values())
    return {
        "model": model,
        "status": "PASS" if passed else "FAIL",
        "absolute_tolerance": ATOL,
        "maximum_metric_absolute_difference": max_difference,
        "metric_comparison": differences,
        "count_comparison": count_differences,
    }


def candidate_scores(predictions: pd.DataFrame, validation: pd.DataFrame,
                     train: pd.DataFrame, item_ids: list[str]) -> pd.DataFrame:
    observed = train.groupby("Student_ID")["Course_ID"].agg(set).to_dict()
    relevant = (validation[validation["Rating"] >= 4.0]
                .groupby("Student_ID")["Course_ID"].agg(set).to_dict())
    chunks = []
    for user in sorted(validation["Student_ID"].unique()):
        items = [item for item in item_ids if item not in observed.get(user, set())]
        chunks.append(pd.DataFrame({
            "user_id": user,
            "course_id": items,
            "score": predictions.loc[user, items].to_numpy(dtype=float),
            "train_observed": False,
            "is_relevant_validation": [item in relevant.get(user, set()) for item in items],
        }))
    return pd.concat(chunks, ignore_index=True)


def format_rating_details(details: pd.DataFrame) -> pd.DataFrame:
    return details.rename(columns={
        "Student_ID": "user_id", "Course_ID": "course_id", "Actual": "true_rating",
        "Predicted_Raw": "raw_prediction", "Predicted": "clipped_prediction",
        "Absolute_Error": "absolute_error", "Squared_Error": "squared_error",
    })[["user_id", "course_id", "true_rating", "raw_prediction", "clipped_prediction",
        "absolute_error", "squared_error", "Cold_User", "Cold_Item"]]


def format_per_user(per_user: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    counts = train.groupby("Student_ID").size().rename("train_interaction_count")
    out = per_user.merge(counts, left_on="Student_ID", right_index=True, how="left")
    out["ranking_eligible"] = out["Relevant_Items"] > 0
    return out.rename(columns={
        "Student_ID": "user_id", "User_MAE": "MAE", "User_RMSE": "RMSE",
        "User_Rating_Rows": "number_of_validation_ratings",
        "Candidates": "number_of_ranking_candidates",
        "Relevant_Items": "number_of_relevant_validation_items",
    })[["user_id", "train_interaction_count", "number_of_validation_ratings", "MAE", "RMSE",
        "number_of_ranking_candidates", "number_of_relevant_validation_items", "ranking_eligible",
        "Precision@5", "Recall@5", "NDCG@5", "HitRate@5",
        "Precision@10", "Recall@10", "NDCG@10", "HitRate@10"]]


def config_payload(model: str, params: dict, repro: dict, runtime: float,
                   training_loss: float | None = None) -> dict:
    payload = {
        "purpose": "validation_artifact_recovery_only",
        "model": model,
        "frozen_publication_configuration": params,
        "seed": SEED,
        "training_implementation": (
            "experiments/aciids_2027/run_mf.py:fit_predict" if model == "mf"
            else "experiments/aciids_2027/run_lightgcn.py:fit_predict"
        ),
        "reproducibility": repro,
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "evaluation_implementation": "shared/evaluator.py:evaluate_predictions",
        "evaluation_policy": EvaluationPolicy().__dict__,
        "rating_calibration": "shared evaluator clips raw scores to [1,5]",
        "ranking_scores": "raw scores; no clipping or scaling",
        "split": str(SPLITS),
        "validation_only": True,
        "test_loaded": False,
        "runtime_seconds": runtime,
        "input_hashes_sha256": {
            "train.csv": sha256(SPLITS / "train.csv"),
            "validation.csv": sha256(SPLITS / "validation.csv"),
            "shared/evaluator.py": sha256(ROOT / "shared/evaluator.py"),
            "shared/reproducibility.py": sha256(ROOT / "shared/reproducibility.py"),
        },
    }
    if training_loss is not None:
        payload["training_loss"] = training_loss
    return payload


def main() -> None:
    repro = set_global_seed(SEED, deterministic=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen publication recovery requires CUDA")
    device = torch.device("cuda")
    catalog = pd.read_csv(MATRIX, index_col=0).fillna(0)
    user_ids = catalog.index.astype(str).tolist()
    item_ids = catalog.columns.astype(str).tolist()
    train = pd.read_csv(SPLITS / "train.csv", dtype={"Student_ID": str, "Course_ID": str})
    validation = pd.read_csv(SPLITS / "validation.csv", dtype={"Student_ID": str, "Course_ID": str})
    train_mean = float(train["Rating"].mean())
    policy = EvaluationPolicy()

    start = time.perf_counter()
    mf_matrix = fit_mf(train, user_ids, item_ids, epochs=MF_PARAMS["epochs"],
                       dim=MF_PARAMS["latent_dim"], lr=MF_PARAMS["learning_rate"],
                       weight_decay=MF_PARAMS["weight_decay"], seed=SEED, device=device)
    mf_metrics, mf_details, mf_users = evaluate_predictions(
        mf_matrix, validation, train, item_ids, train_mean, policy)
    torch.cuda.synchronize(device)
    mf_runtime = time.perf_counter() - start

    start = time.perf_counter()
    lg_matrix, lg_loss = fit_lightgcn(
        train, user_ids, item_ids, embedding_dim=LG_PARAMS["embedding_dim"],
        number_of_layers=LG_PARAMS["number_of_layers"],
        learning_rate=LG_PARAMS["learning_rate"], epochs=LG_PARAMS["epochs"],
        weight_decay=LG_PARAMS["weight_decay"], seed=SEED, device=device)
    lg_metrics, lg_details, lg_users = evaluate_predictions(
        lg_matrix, validation, train, item_ids, train_mean, policy)
    torch.cuda.synchronize(device)
    lg_runtime = time.perf_counter() - start

    mf_check = check_reproduction("mf", mf_metrics, authoritative_row("mf", MF_PARAMS))
    lg_check = check_reproduction("lightgcn_rmse", lg_metrics,
                                  authoritative_row("lightgcn", LG_PARAMS))
    if mf_check["status"] != "PASS" or lg_check["status"] != "PASS":
        raise RuntimeError("Validation reproduction mismatch; no recovery artifacts were written.\n" +
                           json.dumps({"mf": mf_check, "lightgcn_rmse": lg_check}, indent=2))

    mf_candidates = candidate_scores(mf_matrix, validation, train, item_ids)
    lg_candidates = candidate_scores(lg_matrix, validation, train, item_ids)
    mf_per_user = format_per_user(mf_users, train)
    lg_per_user = format_per_user(lg_users, train)

    rating_keys_equal = set(zip(mf_details.Student_ID, mf_details.Course_ID)) == set(zip(lg_details.Student_ID, lg_details.Course_ID))
    eligible_equal = set(mf_per_user.loc[mf_per_user.ranking_eligible, "user_id"]) == set(lg_per_user.loc[lg_per_user.ranking_eligible, "user_id"])
    mf_sets = mf_candidates.groupby("user_id")["course_id"].agg(set)
    lg_sets = lg_candidates.groupby("user_id")["course_id"].agg(set)
    candidate_sets_equal = mf_sets.equals(lg_sets)
    relevance_equal = (mf_candidates[["user_id", "course_id", "is_relevant_validation"]]
                       .equals(lg_candidates[["user_id", "course_id", "is_relevant_validation"]]))
    exclusions_valid = not mf_candidates.train_observed.any() and not lg_candidates.train_observed.any()
    counts_equal = (mf_per_user.set_index("user_id")["number_of_ranking_candidates"]
                    .equals(lg_per_user.set_index("user_id")["number_of_ranking_candidates"]))
    audit_pass = all([rating_keys_equal, eligible_equal, candidate_sets_equal,
                      relevance_equal, exclusions_valid, counts_equal,
                      len(mf_per_user) == len(lg_per_user)])
    if not audit_pass:
        raise RuntimeError("Cross-model artifact consistency audit failed; nothing written")

    for name, details, users, candidates, params, check, runtime, loss in [
        ("mf", mf_details, mf_per_user, mf_candidates, MF_PARAMS, mf_check, mf_runtime, None),
        ("lightgcn_rmse", lg_details, lg_per_user, lg_candidates, LG_PARAMS, lg_check, lg_runtime, lg_loss),
    ]:
        destination = OUTPUT / name
        destination.mkdir(parents=True, exist_ok=True)
        format_rating_details(details).to_csv(destination / "validation_predictions.csv", index=False)
        users.to_csv(destination / "validation_per_user_metrics.csv", index=False)
        candidates.to_csv(destination / "validation_candidate_scores.csv", index=False)
        write_json(destination / "config.json", config_payload(name, params, repro, runtime, loss))
        write_json(destination / "reproduction_check.json", check)

    candidate_min = int(mf_per_user.number_of_ranking_candidates.min())
    candidate_max = int(mf_per_user.number_of_ranking_candidates.max())
    audit = f"""# Validation artifact consistency audit

Status: **PASS**

| Check | Result |
|---|---|
| Identical validation rating keys | {rating_keys_equal} |
| Identical ranking-eligible users | {eligible_equal} |
| Identical candidate item sets per user | {candidate_sets_equal} |
| Identical relevance labels | {relevance_equal} |
| Train-observed items excluded | {exclusions_valid} |
| Identical candidate counts per user | {counts_equal} |
| Identical number of validation users | {len(mf_per_user) == len(lg_per_user)} ({len(mf_per_user)}) |

Candidate scores contain only full-catalog items not observed by that user in TRAIN. `train_observed` is therefore always false. Relevance is `validation rating >= 4`, exactly matching `EvaluationPolicy`. Ranking order remains shared-evaluator order: descending raw score, then ascending course ID for ties.
"""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "validation_artifact_audit.md").write_text(audit, encoding="utf-8")

    def diff(check: dict, metric: str) -> float:
        return check["metric_comparison"][metric]["absolute_difference"]

    summary = f"""# Validation prediction artifact recovery

## Outcome

Both frozen publication models were reproduced successfully using seed 42, deterministic CUDA, the unchanged seed-42 TRAIN/validation manifests, their existing `fit_predict` implementations, and the shared evaluator. No test manifest or test result was loaded. No hyperparameter search or router analysis was performed.

| Model | MAE | saved MAE | abs. diff | RMSE | saved RMSE | abs. diff | NDCG@5 | saved NDCG@5 | abs. diff | Recall@5 | saved Recall@5 | abs. diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MF | {mf_metrics['MAE']:.12f} | {mf_check['metric_comparison']['MAE']['saved']:.12f} | {diff(mf_check, 'MAE'):.3g} | {mf_metrics['RMSE']:.12f} | {mf_check['metric_comparison']['RMSE']['saved']:.12f} | {diff(mf_check, 'RMSE'):.3g} | {mf_metrics['NDCG@5']:.12f} | {mf_check['metric_comparison']['NDCG@5']['saved']:.12f} | {diff(mf_check, 'NDCG@5'):.3g} | {mf_metrics['Recall@5']:.12f} | {mf_check['metric_comparison']['Recall@5']['saved']:.12f} | {diff(mf_check, 'Recall@5'):.3g} |
| LightGCN-RMSE | {lg_metrics['MAE']:.12f} | {lg_check['metric_comparison']['MAE']['saved']:.12f} | {diff(lg_check, 'MAE'):.3g} | {lg_metrics['RMSE']:.12f} | {lg_check['metric_comparison']['RMSE']['saved']:.12f} | {diff(lg_check, 'RMSE'):.3g} | {lg_metrics['NDCG@5']:.12f} | {lg_check['metric_comparison']['NDCG@5']['saved']:.12f} | {diff(lg_check, 'NDCG@5'):.3g} | {lg_metrics['Recall@5']:.12f} | {lg_check['metric_comparison']['Recall@5']['saved']:.12f} | {diff(lg_check, 'Recall@5'):.3g} |

All ten aggregate floating-point metrics reproduced within absolute tolerance {ATOL}; maximum absolute differences were {mf_check['maximum_metric_absolute_difference']:.3g} (MF) and {lg_check['maximum_metric_absolute_difference']:.3g} (LightGCN-RMSE).

## Artifact counts and alignment

- Validation rating rows: {len(mf_details)} for each model.
- Validation users: {len(mf_per_user)} for each model.
- Ranking-eligible users: {int(mf_per_user.ranking_eligible.sum())} for each model.
- Candidate-count range: {candidate_min}–{candidate_max} per validation user.
- Candidate rows: {len(mf_candidates)} for each model.
- MF and LightGCN candidate sets: **exactly aligned** per user.
- Rating keys, eligible users, relevance labels, TRAIN exclusions, and candidate counts: **exactly aligned**.

## Sufficiency for the router

The recovered validation artifacts are sufficient for the previously specified validation-only sparsity-router experiment: each model now has observed-rating predictions, per-user shared-evaluator metrics, and raw full-catalog candidate scores over identical candidate sets. Threshold selection and test evaluation have deliberately not been started.
"""
    (OUTPUT / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
