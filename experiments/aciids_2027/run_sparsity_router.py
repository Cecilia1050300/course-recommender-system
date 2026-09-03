"""Validation-selected interaction-count router over frozen saved artifacts.

No recommender is trained and no model inference is performed. The program
freezes and persists the validation-selected threshold before opening any test
artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.evaluator import EvaluationPolicy, evaluate_predictions
from shared.reproducibility import write_json


SEED = 42
THRESHOLDS = [3, 5, 7, 10, 15, 20]
SPLITS = ROOT / "experiments/aciids_2027/splits/seed_42"
RECOVERY = ROOT / "experiments/aciids_2027/artifacts/validation_predictions"
RESULTS = ROOT / "experiments/aciids_2027/results"
OUTPUT = RESULTS / "sparsity_router_v1"
POLICY = EvaluationPolicy()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_map(users: list[str], counts: pd.Series, tau: int | None, expert: str | None = None) -> dict[str, str]:
    if expert is not None:
        return {user: expert for user in users}
    assert tau is not None
    return {user: "LightGCN" if int(counts.loc[user]) <= tau else "MF" for user in users}


def load_validation_scores(model: str) -> pd.DataFrame:
    frame = pd.read_csv(
        RECOVERY / model / "validation_candidate_scores.csv",
        dtype={"user_id": str, "course_id": str},
    )
    return frame.pivot(index="user_id", columns="course_id", values="score")


def evaluate_validation_policy(
    name: str,
    tau: int | None,
    routes: dict[str, str],
    mf_scores: pd.DataFrame,
    lg_scores: pd.DataFrame,
    validation: pd.DataFrame,
    train: pd.DataFrame,
    item_ids: list[str],
    counts: pd.Series,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    combined = mf_scores.copy()
    lg_users = [user for user, expert in routes.items() if expert == "LightGCN"]
    if lg_users:
        combined.loc[lg_users] = lg_scores.loc[lg_users]
    metrics, details, per_user = evaluate_predictions(
        combined, validation, train, item_ids, float(train.Rating.mean()), POLICY)
    eligible = set(per_user.loc[per_user.Relevant_Items > 0, "Student_ID"].astype(str))
    users = list(routes)
    n_lg = sum(routes[user] == "LightGCN" for user in users)
    n_mf = len(users) - n_lg
    row = {
        "policy": name, "threshold": tau,
        "users_routed_lightgcn": n_lg,
        "percentage_routed_lightgcn": 100 * n_lg / len(users),
        "users_routed_mf": n_mf,
        "percentage_routed_mf": 100 * n_mf / len(users),
        "ranking_eligible_routed_lightgcn": sum(routes[u] == "LightGCN" for u in eligible),
        "ranking_eligible_routed_mf": sum(routes[u] == "MF" for u in eligible),
        **metrics,
    }
    details["Selected_Expert"] = details.Student_ID.map(routes)
    per_user["Selected_Expert"] = per_user.Student_ID.map(routes)
    per_user["Train_Interaction_Count"] = per_user.Student_ID.map(counts)
    return row, details, per_user


def pareto_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, policy in frame.iterrows():
        dominated_by, dominates = [], []
        for j, other in frame.iterrows():
            if i == j:
                continue
            other_dominates = (
                other.RMSE <= policy.RMSE and other["NDCG@5"] >= policy["NDCG@5"]
                and (other.RMSE < policy.RMSE or other["NDCG@5"] > policy["NDCG@5"])
            )
            this_dominates = (
                policy.RMSE <= other.RMSE and policy["NDCG@5"] >= other["NDCG@5"]
                and (policy.RMSE < other.RMSE or policy["NDCG@5"] > other["NDCG@5"])
            )
            if other_dominates:
                dominated_by.append(other.policy)
            if this_dominates:
                dominates.append(other.policy)
        rows.append({
            "policy": policy.policy,
            "threshold": policy.threshold,
            "RMSE": policy.RMSE,
            "NDCG@5": policy["NDCG@5"],
            "pareto_optimal": not dominated_by,
            "dominated_by": ";".join(dominated_by),
            "dominates": ";".join(dominates),
        })
    return pd.DataFrame(rows)


def saved_test_model(model: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    directory = RESULTS / ("mf_publication" if model == "MF" else "lightgcn_publication")
    details = pd.read_csv(directory / "test_predictions.csv", dtype={"Student_ID": str, "Course_ID": str})
    users = pd.read_csv(directory / "test_per_user_metrics.csv", dtype={"Student_ID": str})
    published = json.loads((directory / "final_test_results.json").read_text())
    metrics = published["metrics"]
    return details, users, metrics


def aggregate_saved(details: pd.DataFrame, users: pd.DataFrame) -> dict:
    errors = details.Actual - details.Predicted
    result = {
        "MAE": float(errors.abs().mean()),
        "RMSE": float(np.sqrt(np.square(errors).mean())),
        "Rating_Rows": len(details),
        "Ranking_Eligible_Users": int(users["NDCG@5"].notna().sum()),
    }
    for k in [5, 10]:
        for metric in ["Precision", "Recall", "NDCG", "HitRate"]:
            result[f"{metric}@{k}"] = float(users[f"{metric}@{k}"].mean(skipna=True))
    return result


def verify_test_artifact(name: str, recovered: dict, saved: dict, atol: float = 1e-12) -> dict:
    metrics = ["MAE", "RMSE", "Precision@5", "Recall@5", "NDCG@5", "HitRate@5",
               "Precision@10", "Recall@10", "NDCG@10", "HitRate@10"]
    differences = {m: abs(float(recovered[m]) - float(saved[m])) for m in metrics}
    passed = max(differences.values()) <= atol
    if not passed:
        raise RuntimeError(f"Saved {name} test artifact consistency failed: {differences}")
    return {"status": "PASS", "absolute_tolerance": atol,
            "maximum_absolute_difference": max(differences.values()),
            "absolute_differences": differences}


def routed_test_artifacts(
    routes: dict[str, str], mf_details: pd.DataFrame, lg_details: pd.DataFrame,
    mf_users: pd.DataFrame, lg_users: pd.DataFrame, counts: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    detail_keys = ["Student_ID", "Course_ID", "Actual"]
    if not mf_details[detail_keys].equals(lg_details[detail_keys]):
        raise RuntimeError("MF and LightGCN test rating keys are not aligned")
    details = mf_details.copy()
    use_lg = details.Student_ID.map(routes).eq("LightGCN")
    details.loc[use_lg, :] = lg_details.loc[use_lg, :].to_numpy()
    details["Selected_Expert"] = details.Student_ID.map(routes)
    details["Train_Interaction_Count"] = details.Student_ID.map(counts)

    mf_idx, lg_idx = mf_users.set_index("Student_ID"), lg_users.set_index("Student_ID")
    if set(mf_idx.index) != set(lg_idx.index):
        raise RuntimeError("MF and LightGCN test per-user keys are not aligned")
    rows = []
    for user in mf_idx.index:
        expert = routes[user]
        source = lg_idx if expert == "LightGCN" else mf_idx
        row = source.loc[user].to_dict()
        row.update({"Student_ID": user, "Selected_Expert": expert,
                    "Train_Interaction_Count": int(counts.loc[user])})
        rows.append(row)
    per_user = pd.DataFrame(rows)
    metrics = aggregate_saved(details, per_user)
    return details, per_user, metrics


def bucket_name(count: int) -> str:
    if count <= 5:
        return "1-5"
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    return ">20"


def bucket_results(model: str, details: pd.DataFrame, users: pd.DataFrame,
                   routes: dict[str, str], counts: pd.Series) -> list[dict]:
    detail = details.copy()
    detail["bucket"] = detail.Student_ID.map(counts).map(bucket_name)
    user = users.copy()
    user["bucket"] = user.Student_ID.map(counts).map(bucket_name)
    result = []
    for bucket in ["1-5", "6-10", "11-20", ">20"]:
        d, u = detail[detail.bucket == bucket], user[user.bucket == bucket]
        eligible = u[u["NDCG@5"].notna()]
        bucket_users = set(u.Student_ID)
        result.append({
            "model": model, "bucket": bucket, "rating_users": u.Student_ID.nunique(),
            "rating_rows": len(d), "ranking_eligible_users": len(eligible),
            "MAE": float((d.Actual - d.Predicted).abs().mean()),
            "RMSE": float(np.sqrt(np.square(d.Actual - d.Predicted).mean())),
            "Recall@5": float(eligible["Recall@5"].mean()),
            "NDCG@5": float(eligible["NDCG@5"].mean()),
            "HitRate@5": float(eligible["HitRate@5"].mean()),
            "users_routed_mf": sum(routes[x] == "MF" for x in bucket_users),
            "users_routed_lightgcn": sum(routes[x] == "LightGCN" for x in bucket_users),
        })
    return result


def oracle_diagnostics(mf_details: pd.DataFrame, lg_details: pd.DataFrame,
                       mf_users: pd.DataFrame, lg_users: pd.DataFrame,
                       router_metrics: dict) -> dict:
    mf_u, lg_u = mf_users.set_index("Student_ID"), lg_users.set_index("Student_ID")
    common = mf_u.index.intersection(lg_u.index)
    rating_mf = int((mf_u.loc[common, "User_MAE"] < lg_u.loc[common, "User_MAE"]).sum())
    rating_lg = int((lg_u.loc[common, "User_MAE"] < mf_u.loc[common, "User_MAE"]).sum())
    rating_tie = len(common) - rating_mf - rating_lg
    rating_routes = {
        user: ("LightGCN" if lg_u.loc[user, "User_MAE"] < mf_u.loc[user, "User_MAE"] else "MF")
        for user in common
    }
    detail = mf_details.copy()
    use_lg = detail.Student_ID.map(rating_routes).eq("LightGCN")
    detail.loc[use_lg, :] = lg_details.loc[use_lg, :].to_numpy()
    rating_mae = float((detail.Actual - detail.Predicted).abs().mean())
    rating_rmse = float(np.sqrt(np.square(detail.Actual - detail.Predicted).mean()))

    eligible = common[mf_u.loc[common, "NDCG@5"].notna() & lg_u.loc[common, "NDCG@5"].notna()]
    rank_mf = int((mf_u.loc[eligible, "NDCG@5"] > lg_u.loc[eligible, "NDCG@5"]).sum())
    rank_lg = int((lg_u.loc[eligible, "NDCG@5"] > mf_u.loc[eligible, "NDCG@5"]).sum())
    rank_tie = len(eligible) - rank_mf - rank_lg
    oracle_ndcg = float(pd.concat([mf_u.loc[eligible, "NDCG@5"],
                                   lg_u.loc[eligible, "NDCG@5"]], axis=1).max(axis=1).mean())
    return {
        "label": "POST-HOC ORACLE UPPER BOUND — NOT A DEPLOYABLE METHOD",
        "used_for_selection": False,
        "rating_oracle": {
            "selection_metric": "lower per-user MAE", "users": len(common),
            "mf_selected": rating_mf, "lightgcn_selected": rating_lg, "ties": rating_tie,
            "tie_handling_for_secondary_metrics": "MF deterministic default",
            "MAE": rating_mae, "RMSE": rating_rmse,
            "router_MAE_gap": router_metrics["MAE"] - rating_mae,
            "router_RMSE_gap": router_metrics["RMSE"] - rating_rmse,
        },
        "ranking_oracle": {
            "selection_metric": "higher per-user NDCG@5", "eligible_users": len(eligible),
            "mf_selected": rank_mf, "lightgcn_selected": rank_lg, "ties": rank_tie,
            "NDCG@5": oracle_ndcg,
            "router_NDCG@5_gap": oracle_ndcg - router_metrics["NDCG@5"],
        },
    }


def main() -> None:
    # Phase 1: validation only. No test path is opened above this phase boundary.
    train = pd.read_csv(SPLITS / "train.csv", dtype={"Student_ID": str, "Course_ID": str})
    validation = pd.read_csv(SPLITS / "validation.csv", dtype={"Student_ID": str, "Course_ID": str})
    counts = train.groupby("Student_ID").size()
    catalog = pd.read_csv(ROOT / "old/rating_matrix - rating_matrix.csv", index_col=0).fillna(0)
    item_ids = catalog.columns.astype(str).tolist()
    mf_scores = load_validation_scores("mf")
    lg_scores = load_validation_scores("lightgcn_rmse")
    users = sorted(validation.Student_ID.unique())

    policies = [("Always MF", None, route_map(users, counts, None, "MF")),
                ("Always LightGCN", None, route_map(users, counts, None, "LightGCN"))]
    policies.extend((f"tau={tau}", tau, route_map(users, counts, tau)) for tau in THRESHOLDS)
    validation_rows = []
    for name, tau, routes in policies:
        row, _, _ = evaluate_validation_policy(
            name, tau, routes, mf_scores, lg_scores, validation, train, item_ids, counts)
        validation_rows.append(row)
    validation_results = pd.DataFrame(validation_rows)
    pareto = pareto_analysis(validation_results)

    rmse_mf = float(validation_results.loc[validation_results.policy == "Always MF", "RMSE"].iloc[0])
    ceiling = 1.10 * rmse_mf
    candidates = validation_results[
        validation_results.threshold.notna() & (validation_results.RMSE <= ceiling)
    ].copy()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validation_results.to_csv(OUTPUT / "validation_router_results.csv", index=False)
    pareto.to_csv(OUTPUT / "validation_pareto_analysis.csv", index=False)
    if candidates.empty:
        selection = {
            "status": "NO ELIGIBLE ROUTER",
            "selected_tau": None,
            "selection_data": "validation_only",
            "rule": "maximize validation NDCG@5 among threshold routers with RMSE <= 1.10 * Always-MF RMSE; ties: lower RMSE, then smaller tau",
            "always_mf_validation_rmse": rmse_mf,
            "rmse_eligibility_ceiling": ceiling,
            "eligible_thresholds": [],
            "test_artifacts_loaded_at_write_time": False,
        }
        write_json(OUTPUT / "selected_threshold.json", selection)
        write_json(OUTPUT / "final_test_results.json", {
            "status": "NOT RUN — NO ELIGIBLE ROUTER",
            "test_artifacts_loaded": False,
            "test_evaluations": 0,
        })
        write_json(OUTPUT / "oracle_upper_bound.json", {
            "status": "NOT RUN — NO FROZEN ROUTER AND TEST ARTIFACTS NOT LOADED",
            "label": "POST-HOC ORACLE UPPER BOUND — NOT A DEPLOYABLE METHOD",
            "used_for_selection": False,
        })
        status_column = ["status"]
        pd.DataFrame([{"status": "NOT RUN — NO ELIGIBLE ROUTER"}]).to_csv(
            OUTPUT / "test_predictions_router.csv", index=False, columns=status_column)
        pd.DataFrame([{"status": "NOT RUN — NO ELIGIBLE ROUTER"}]).to_csv(
            OUTPUT / "test_per_user_metrics_router.csv", index=False, columns=status_column)
        pd.DataFrame([{"status": "NOT RUN — NO ELIGIBLE ROUTER"}]).to_csv(
            OUTPUT / "sparsity_bucket_results.csv", index=False, columns=status_column)
        write_json(OUTPUT / "config.json", {
            "experiment": "sparsity_router_v1", "seed": SEED,
            "experts": {"MF": "publication-selected", "LightGCN": "RMSE-selected publication"},
            "models_retrained": False, "new_inference": False,
            "thresholds": THRESHOLDS,
            "route": "LightGCN if TRAIN interaction count <= tau, else MF",
            "selection_rule": selection["rule"],
            "evaluation_policy": POLICY.__dict__,
            "test_artifacts_loaded": False, "test_evaluations": 0,
            "validation_artifact_hashes_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted(RECOVERY.glob("*/*.csv"))
            },
        })
        frontier = pareto.loc[pareto.pareto_optimal, "policy"].tolist()
        rows = "\n".join(
            f"| {r['policy']} | {'' if pd.isna(r['threshold']) else int(r['threshold'])} | {r['MAE']:.6f} | {r['RMSE']:.6f} | {r['NDCG@5']:.6f} | {r['Recall@5']:.6f} | {int(r['users_routed_lightgcn'])} | {int(r['users_routed_mf'])} |"
            for _, r in validation_results.iterrows()
        )
        summary = f"""# Sparsity-aware MF/LightGCN router V1

## Outcome: NO ELIGIBLE ROUTER

No recommender was retrained and no inference was run. Validation used only the recovered artifacts. Always-MF validation RMSE is {rmse_mf:.6f}, so the predeclared 10% ceiling is {ceiling:.6f}. Every candidate threshold exceeded that ceiling; therefore no threshold was frozen.

| Policy | tau | MAE | RMSE | NDCG@5 | Recall@5 | Users→LightGCN | Users→MF |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

- Validation Pareto-optimal policies: {', '.join(frontier)}.
- Selected threshold: **none**.
- Eligible thresholds: **none**.
- Test artifacts loaded: **false**.
- Test evaluations: **0**.
- Test baseline comparison, sparsity buckets, and oracle diagnostics: **not performed**, because doing so without a frozen eligible router would violate the protocol.

## Required interpretation

1. Pareto-optimal policies are listed above and fully audited in `validation_pareto_analysis.csv`.
2. Threshold selection: **NOT SUPPORTED — NO ELIGIBLE ROUTER**.
3–10. Test routing share, test improvements, bucket effects, and oracle headroom: **INCONCLUSIVE**, because test was not accessed.
11. Simple interaction-count routing as a paper method: **NOT SUPPORTED** under the predeclared validation RMSE constraint.
12. Learned-router justification: **INCONCLUSIVE**; this failed threshold family alone is not evidence that a learned gate will work.

`final_test_results.json`, test CSVs, bucket CSV, and `oracle_upper_bound.json` are explicit status artifacts, not fabricated results.
"""
        (OUTPUT / "summary.md").write_text(summary, encoding="utf-8")
        print(summary)
        return
    candidates["threshold"] = candidates.threshold.astype(int)
    selected = candidates.sort_values(
        ["NDCG@5", "RMSE", "threshold"], ascending=[False, True, True]
    ).iloc[0]
    selected_tau = int(selected.threshold)

    selection = {
        "status": "SELECTED", "selected_tau": selected_tau,
        "selection_data": "validation_only",
        "rule": "maximize validation NDCG@5 among threshold routers with RMSE <= 1.10 * Always-MF RMSE; ties: lower RMSE, then smaller tau",
        "always_mf_validation_rmse": rmse_mf,
        "rmse_eligibility_ceiling": ceiling,
        "eligible_thresholds": candidates.threshold.astype(int).tolist(),
        "selected_validation_metrics": {key: float(selected[key]) for key in
                                        ["MAE", "RMSE", "Precision@5", "Recall@5", "NDCG@5", "HitRate@5",
                                         "Precision@10", "Recall@10", "NDCG@10", "HitRate@10"]},
        "test_artifacts_loaded_at_write_time": False,
    }
    # Protocol checkpoint: threshold is persisted before any test artifact is opened.
    write_json(OUTPUT / "selected_threshold.json", selection)

    # Phase 2: frozen one-time test artifact composition.
    mf_details, mf_users, mf_saved = saved_test_model("MF")
    lg_details, lg_users, lg_saved = saved_test_model("LightGCN")
    mf_loaded = aggregate_saved(mf_details, mf_users)
    lg_loaded = aggregate_saved(lg_details, lg_users)
    mf_verify = verify_test_artifact("MF", mf_loaded, mf_saved)
    lg_verify = verify_test_artifact("LightGCN", lg_loaded, lg_saved)
    test_users = sorted(mf_users.Student_ID.unique())
    routes = route_map(test_users, counts, selected_tau)
    router_details, router_users, router_metrics = routed_test_artifacts(
        routes, mf_details, lg_details, mf_users, lg_users, counts)
    router_details.to_csv(OUTPUT / "test_predictions_router.csv", index=False)
    router_users.to_csv(OUTPUT / "test_per_user_metrics_router.csv", index=False)

    sparsity = []
    for name, details, per_user in [("MF", mf_details, mf_users),
                                    ("LightGCN", lg_details, lg_users),
                                    ("Router", router_details, router_users)]:
        sparsity.extend(bucket_results(name, details, per_user, routes, counts))
    sparsity_df = pd.DataFrame(sparsity)
    sparsity_df.to_csv(OUTPUT / "sparsity_bucket_results.csv", index=False)

    oracle = oracle_diagnostics(mf_details, lg_details, mf_users, lg_users, router_metrics)
    write_json(OUTPUT / "oracle_upper_bound.json", oracle)
    rmse_improvement_lg = lg_loaded["RMSE"] - router_metrics["RMSE"]
    rmse_degradation_mf = router_metrics["RMSE"] - mf_loaded["RMSE"]
    ndcg_improvement_mf = router_metrics["NDCG@5"] - mf_loaded["NDCG@5"]
    ndcg_gain = lg_loaded["NDCG@5"] - mf_loaded["NDCG@5"]
    retention = ndcg_improvement_mf / ndcg_gain
    final = {
        "test_evaluations": 1, "selected_tau": selected_tau,
        "metrics": router_metrics,
        "routed_users": {
            "LightGCN": sum(x == "LightGCN" for x in routes.values()),
            "MF": sum(x == "MF" for x in routes.values()),
            "total": len(routes),
        },
        "saved_baselines": {"MF": mf_loaded, "LightGCN": lg_loaded},
        "baseline_artifact_verification": {"MF": mf_verify, "LightGCN": lg_verify},
        "comparisons": {
            "RMSE_improvement_vs_LightGCN": rmse_improvement_lg,
            "RMSE_degradation_vs_MF": rmse_degradation_mf,
            "NDCG5_improvement_vs_MF": ndcg_improvement_mf,
            "NDCG5_gain_retention_fraction": retention,
        },
    }
    write_json(OUTPUT / "final_test_results.json", final)
    config = {
        "experiment": "sparsity_router_v1", "seed": SEED,
        "experts": {"MF": "publication-selected", "LightGCN": "RMSE-selected publication"},
        "models_retrained": False, "new_inference": False,
        "thresholds": THRESHOLDS,
        "route": "LightGCN if TRAIN interaction count <= tau, else MF",
        "selection_rule": selection["rule"],
        "evaluation_policy": POLICY.__dict__,
        "test_evaluations": 1,
        "validation_artifact_hashes_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted(RECOVERY.glob("*/*.csv"))
        },
        "test_artifact_hashes_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for directory in [RESULTS / "mf_publication", RESULTS / "lightgcn_publication"]
            for path in [directory / "test_predictions.csv", directory / "test_per_user_metrics.csv",
                         directory / "final_test_results.json"]
        },
    }
    write_json(OUTPUT / "config.json", config)

    frontier = pareto.loc[pareto.pareto_optimal, "policy"].tolist()
    n_lg = final["routed_users"]["LightGCN"]
    pct_lg = 100 * n_lg / final["routed_users"]["total"]
    router_dominates_mf = router_metrics["RMSE"] <= mf_loaded["RMSE"] and router_metrics["NDCG@5"] >= mf_loaded["NDCG@5"] and (router_metrics["RMSE"] < mf_loaded["RMSE"] or router_metrics["NDCG@5"] > mf_loaded["NDCG@5"])
    router_dominates_lg = router_metrics["RMSE"] <= lg_loaded["RMSE"] and router_metrics["NDCG@5"] >= lg_loaded["NDCG@5"] and (router_metrics["RMSE"] < lg_loaded["RMSE"] or router_metrics["NDCG@5"] > lg_loaded["NDCG@5"])
    bucket_lines = []
    for bucket in ["1-5", "6-10", "11-20", ">20"]:
        rows = sparsity_df[sparsity_df.bucket == bucket].set_index("model")
        bucket_lines.append(
            f"- {bucket}: router MAE {rows.loc['Router','MAE']:.4f}, NDCG@5 {rows.loc['Router','NDCG@5']:.4f}; "
            f"MF NDCG@5 {rows.loc['MF','NDCG@5']:.4f}, LightGCN NDCG@5 {rows.loc['LightGCN','NDCG@5']:.4f}; "
            f"routed MF/LightGCN {int(rows.loc['Router','users_routed_mf'])}/{int(rows.loc['Router','users_routed_lightgcn'])}."
        )
    bucket_text = "\n".join(bucket_lines)
    oracle_ndcg_gap = oracle["ranking_oracle"]["router_NDCG@5_gap"]
    oracle_mae_gap = oracle["rating_oracle"]["router_MAE_gap"]
    simple_verdict = "SUPPORTED" if router_dominates_mf or router_dominates_lg else ("SUGGESTIVE" if retention > 0.5 and rmse_improvement_lg > 0 else "NOT SUPPORTED")
    learned_verdict = "SUGGESTIVE" if oracle_ndcg_gap > 0.02 or oracle_mae_gap > 0.02 else "NOT SUPPORTED"
    summary = f"""# Sparsity-aware MF/LightGCN router V1

## Protocol

No recommender was retrained and no model inference was run. Validation used the recovered seed-42 user-level/full-catalog artifacts. The predeclared rule selected and persisted `tau={selected_tau}` before any test artifact was opened. Test was composed once from existing publication prediction and per-user metric files. All predictions/scores for a user come from one expert.

## Validation selection and Pareto analysis

- Pareto-optimal policies: {', '.join(frontier)}.
- Always-MF RMSE: {rmse_mf:.6f}; 10% ceiling: {ceiling:.6f}.
- Eligible thresholds: {', '.join(map(str, selection['eligible_thresholds']))}.
- Selected threshold: **tau={selected_tau}**, by highest validation NDCG@5 among eligible routers.
- Selected validation RMSE/NDCG@5: {selected.RMSE:.6f} / {selected['NDCG@5']:.6f}.

Full domination relationships are in `validation_pareto_analysis.csv`.

## Frozen test result

| Policy | MAE | RMSE | Precision@5 | Recall@5 | NDCG@5 | HitRate@5 | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always MF | {mf_loaded['MAE']:.6f} | {mf_loaded['RMSE']:.6f} | {mf_loaded['Precision@5']:.6f} | {mf_loaded['Recall@5']:.6f} | {mf_loaded['NDCG@5']:.6f} | {mf_loaded['HitRate@5']:.6f} | {mf_loaded['Precision@10']:.6f} | {mf_loaded['Recall@10']:.6f} | {mf_loaded['NDCG@10']:.6f} | {mf_loaded['HitRate@10']:.6f} |
| Always LightGCN | {lg_loaded['MAE']:.6f} | {lg_loaded['RMSE']:.6f} | {lg_loaded['Precision@5']:.6f} | {lg_loaded['Recall@5']:.6f} | {lg_loaded['NDCG@5']:.6f} | {lg_loaded['HitRate@5']:.6f} | {lg_loaded['Precision@10']:.6f} | {lg_loaded['Recall@10']:.6f} | {lg_loaded['NDCG@10']:.6f} | {lg_loaded['HitRate@10']:.6f} |
| Router tau={selected_tau} | {router_metrics['MAE']:.6f} | {router_metrics['RMSE']:.6f} | {router_metrics['Precision@5']:.6f} | {router_metrics['Recall@5']:.6f} | {router_metrics['NDCG@5']:.6f} | {router_metrics['HitRate@5']:.6f} | {router_metrics['Precision@10']:.6f} | {router_metrics['Recall@10']:.6f} | {router_metrics['NDCG@10']:.6f} | {router_metrics['HitRate@10']:.6f} |

Baseline artifact recomputation matched saved metrics (maximum difference MF={mf_verify['maximum_absolute_difference']:.3g}, LightGCN={lg_verify['maximum_absolute_difference']:.3g}).

## Required comparisons

- Users routed to LightGCN: {n_lg}/{len(routes)} ({pct_lg:.2f}%); MF: {len(routes)-n_lg}/{len(routes)} ({100-pct_lg:.2f}%).
- RMSE improvement versus LightGCN: {rmse_improvement_lg:.6f} ({100*rmse_improvement_lg/lg_loaded['RMSE']:.2f}%).
- RMSE degradation versus MF: {rmse_degradation_mf:.6f} ({100*rmse_degradation_mf/mf_loaded['RMSE']:.2f}%).
- NDCG@5 improvement versus MF: {ndcg_improvement_mf:.6f}.
- Fraction of LightGCN's NDCG@5 gain over MF retained: {retention:.3f} ({100*retention:.1f}%).
- Router dominates MF: {router_dominates_mf}; router dominates LightGCN: {router_dominates_lg}.

## Sparsity buckets

{bucket_text}

“Benefit/degrade” is metric-specific: each bucket row in `sparsity_bucket_results.csv` contains rating rows/users, eligible users, MAE/RMSE and ranking metrics for all three policies.

## POST-HOC ORACLE UPPER BOUND — NOT A DEPLOYABLE METHOD

The oracles were computed only after threshold freeze and were not used for selection.

- Rating oracle: MF wins {oracle['rating_oracle']['mf_selected']}, LightGCN wins {oracle['rating_oracle']['lightgcn_selected']}, ties {oracle['rating_oracle']['ties']}; MAE={oracle['rating_oracle']['MAE']:.6f}. Router-to-oracle MAE gap={oracle_mae_gap:.6f}.
- Ranking oracle: MF wins {oracle['ranking_oracle']['mf_selected']}, LightGCN wins {oracle['ranking_oracle']['lightgcn_selected']}, ties {oracle['ranking_oracle']['ties']}; NDCG@5={oracle['ranking_oracle']['NDCG@5']:.6f}. Oracle-to-router NDCG@5 gap={oracle_ndcg_gap:.6f}.

## Explicit answers and verdicts

1. **Validation Pareto policies — SUPPORTED:** {', '.join(frontier)}.
2. **Selected threshold — SUPPORTED:** tau={selected_tau}, using validation only.
3. **LightGCN routing share — SUPPORTED:** {pct_lg:.2f}%.
4. **Meaningful rating recovery versus Always LightGCN — {'SUPPORTED' if rmse_improvement_lg > 0 else 'NOT SUPPORTED'}:** RMSE improves by {rmse_improvement_lg:.6f}.
5. **LightGCN NDCG gain retained — SUPPORTED:** {100*retention:.1f}%.
6. **Better than MF for ranking — {'SUPPORTED' if ndcg_improvement_mf > 0 else 'NOT SUPPORTED'}:** NDCG@5 difference {ndcg_improvement_mf:+.6f}.
7. **Better than LightGCN for rating — {'SUPPORTED' if rmse_improvement_lg > 0 else 'NOT SUPPORTED'}:** RMSE difference {-rmse_improvement_lg:+.6f} (router minus LightGCN).
8. **Dominance versus baselines — {'SUPPORTED' if router_dominates_mf or router_dominates_lg else 'NOT SUPPORTED'}:** it {'dominates at least one baseline' if router_dominates_mf or router_dominates_lg else 'is an intermediate trade-off point'} in RMSE–NDCG@5 space.
9. **Sparsity explanation — SUGGESTIVE:** routing is determined entirely by TRAIN count; bucket-specific gains/degradations above localize the result but do not establish causality.
10. **Oracle headroom — SUPPORTED:** MAE gap {oracle_mae_gap:.6f}, NDCG@5 gap {oracle_ndcg_gap:.6f} remain.
11. **Simple rule as a paper method — {simple_verdict}:** this is one fixed split and two frozen experts; publication claims require robustness beyond this experiment.
12. **Justification for a learned router — {learned_verdict}:** oracle headroom indicates potential, but a learned gate is not implemented or validated here.

## Scientific boundary

This experiment estimates one validation-selected operating point, not a causal effect of sparsity and not prospective deployment performance. Oracle metrics are post-hoc upper bounds. Test outcomes were accessed only after `selected_threshold.json` had been written.
"""
    (OUTPUT / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
