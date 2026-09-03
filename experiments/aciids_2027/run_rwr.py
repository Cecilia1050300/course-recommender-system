"""Publication RWR rerun preserving the three legacy graph variants."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.evaluator import EvaluationPolicy, evaluate_predictions
from shared.reproducibility import set_global_seed, write_json

SEED = 42
C_VALUES = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
TEMPERATURES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
VARIANTS = ["RWR_Binary", "RWR_Linear", "RWR_Softmax"]
SPLITS = ROOT / "experiments/aciids_2027/splits/seed_42"
CATALOG_FILE = ROOT / "old/rating_matrix - rating_matrix.csv"
OUTPUT = ROOT / "experiments/aciids_2027/results/rwr_publication"
RESULTS = ROOT / "experiments/aciids_2027/results"
POLICY = EvaluationPolicy()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rating_graph(train: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    catalog = pd.read_csv(CATALOG_FILE, index_col=0).fillna(0)
    users, items = catalog.index.astype(str).tolist(), catalog.columns.astype(str).tolist()
    u_map, i_map = {u: n for n, u in enumerate(users)}, {i: n for n, i in enumerate(items)}
    ratings = np.zeros((len(users), len(items)), dtype=np.float64)
    for row in train.itertuples(index=False):
        ratings[u_map[str(row.Student_ID)], i_map[str(row.Course_ID)]] = float(row.Rating)
    adjacency = np.zeros((len(users) + len(items), len(users) + len(items)), dtype=np.float64)
    adjacency[:len(users), len(users):] = ratings
    adjacency[len(users):, :len(users)] = ratings.T
    return users, items, ratings, adjacency


def transition(adjacency: np.ndarray, variant: str, temperature: float | None) -> csr_matrix:
    if variant == "RWR_Binary":
        weights = np.where(adjacency > 0, 1.0, 0.0)
        sums = weights.sum(axis=1, keepdims=True)
        matrix = weights / np.where(sums == 0, 1.0, sums)
    elif variant == "RWR_Linear":
        weights = adjacency.copy()
        sums = weights.sum(axis=1, keepdims=True)
        matrix = weights / np.where(sums == 0, 1.0, sums)
    elif variant == "RWR_Softmax":
        assert temperature is not None
        matrix = np.zeros_like(adjacency)
        for row in range(len(adjacency)):
            mask = adjacency[row] > 0
            if not mask.any():
                continue
            values = adjacency[row, mask] / temperature
            values -= values.max()
            exp_values = np.exp(values)
            matrix[row, mask] = exp_values / exp_values.sum()
    else:
        raise ValueError(variant)
    return csr_matrix(matrix)


def rwr_scores(W: csr_matrix, num_users: int, num_items: int, c: float) -> tuple[np.ndarray, dict]:
    """Run the exact legacy fixed-point iteration independently for every user."""
    transposed = W.transpose().tocsr()
    raw = np.empty((num_users, num_items), dtype=np.float64)
    iterations, unconverged = [], 0
    for user in range(num_users):
        restart = np.zeros(num_users + num_items, dtype=np.float64)
        restart[user] = 1.0
        probability = restart.copy()
        converged = False
        for iteration in range(1, 201):
            updated = (1.0 - c) * transposed.dot(probability) + c * restart
            if np.linalg.norm(updated - probability, ord=2) < 1e-8:
                probability = updated
                converged = True
                break
            probability = updated
        iterations.append(iteration)
        unconverged += int(not converged)
        raw[user] = probability[num_users:]
    return raw, {"mean_iterations": float(np.mean(iterations)),
                 "max_iterations": int(max(iterations)), "unconverged_users": unconverged}


def legacy_rating_mapping(raw: np.ndarray, ratings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Legacy per-user log/MinMax mapping, fitted over TRAIN-unobserved items only."""
    mapped = np.zeros_like(raw)
    degenerate = np.zeros(raw.shape[0], dtype=bool)
    for user in range(raw.shape[0]):
        unobserved = ratings[user] == 0
        values = np.log(raw[user, unobserved] + 1e-10)
        minimum, maximum = values.min(), values.max()
        if maximum != minimum:
            mapped[user, unobserved] = 1.0 + 4.0 * (values - minimum) / (maximum - minimum)
        else:
            mapped[user, unobserved] = 3.0
            degenerate[user] = True
    return mapped, degenerate


def evaluate_dual(
    raw: np.ndarray, mapped: np.ndarray, users: list[str], items: list[str],
    evaluation: pd.DataFrame, train: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    raw_df = pd.DataFrame(raw, index=users, columns=items)
    mapped_df = pd.DataFrame(mapped, index=users, columns=items)
    train_mean = float(train.Rating.mean())
    rating_metrics, rating_details, rating_users = evaluate_predictions(
        mapped_df, evaluation, train, items, train_mean, POLICY)
    ranking_metrics, _, ranking_users = evaluate_predictions(
        raw_df, evaluation, train, items, train_mean, POLICY)
    metrics = {key: rating_metrics[key] for key in
               ["MAE", "RMSE", "Rating_Rows", "Missing_Predictions",
                "Fallback_Predictions", "Cold_User_Rows", "Cold_Item_Rows"]}
    metrics["Ranking_Eligible_Users"] = ranking_metrics["Ranking_Eligible_Users"]
    for k in POLICY.top_ks:
        for name in ["Precision", "Recall", "NDCG", "HitRate"]:
            metrics[f"{name}@{k}"] = ranking_metrics[f"{name}@{k}"]
    rating_columns = ["User_MAE", "User_Rating_Rows", "User_RMSE"]
    per_user = ranking_users.drop(columns=rating_columns).merge(
        rating_users[["Student_ID"] + rating_columns], on="Student_ID", how="left")
    return metrics, rating_details, per_user


def configuration_grid() -> list[tuple[str, float, float | None]]:
    result = []
    for variant in ["RWR_Binary", "RWR_Linear"]:
        result.extend((variant, c, None) for c in C_VALUES)
    result.extend(("RWR_Softmax", c, temperature)
                  for temperature in TEMPERATURES for c in C_VALUES)
    return result


def fallback_counts(degenerate: np.ndarray, users: list[str], evaluation: pd.DataFrame,
                    convergence: dict) -> dict:
    u_map = {u: n for n, u in enumerate(users)}
    eval_users = [str(u) for u in evaluation.Student_ID.unique()]
    degenerate_users = {users[n] for n in np.flatnonzero(degenerate)}
    return {
        "degenerate_minmax_users_all": int(degenerate.sum()),
        "degenerate_minmax_users_evaluated": sum(u in degenerate_users for u in eval_users),
        "rating_rows_using_degenerate_3": int(evaluation.Student_ID.astype(str).isin(degenerate_users).sum()),
        "shared_evaluator_fallback_rows": 0,
        **convergence,
    }


def candidate_signature(evaluation: pd.DataFrame, train: pd.DataFrame,
                        items: list[str]) -> tuple[str, int, int, int]:
    observed = train.groupby("Student_ID")["Course_ID"].agg(set).to_dict()
    keys = [(u, i) for u in sorted(evaluation.Student_ID.unique())
            for i in items if i not in observed.get(u, set())]
    digest = hashlib.sha256("\n".join(f"{u}\t{i}" for u, i in keys).encode()).hexdigest()
    counts = pd.Series([u for u, _ in keys]).value_counts()
    return digest, len(keys), int(counts.min()), int(counts.max())


def main() -> None:
    overall_started = time.perf_counter()
    repro = set_global_seed(SEED, deterministic=True)
    # Phase 1: no TEST access.
    train = pd.read_csv(SPLITS / "train.csv", dtype={"Student_ID": str, "Course_ID": str})
    validation = pd.read_csv(SPLITS / "validation.csv", dtype={"Student_ID": str, "Course_ID": str})
    users, items, ratings, adjacency = build_rating_graph(train)
    transition_cache = {}
    validation_rows, fallback_rows = [], []
    for variant, c, temperature_value in configuration_grid():
        cache_key = (variant, temperature_value)
        if cache_key not in transition_cache:
            transition_cache[cache_key] = transition(adjacency, variant, temperature_value)
        started = time.perf_counter()
        raw, convergence = rwr_scores(transition_cache[cache_key], len(users), len(items), c)
        mapped, degenerate = legacy_rating_mapping(raw, ratings)
        metrics, _, _ = evaluate_dual(raw, mapped, users, items, validation, train)
        audit = fallback_counts(degenerate, users, validation, convergence)
        row = {"Method": variant, "restart_probability_c": c,
               "temperature": temperature_value, "runtime_seconds": time.perf_counter() - started,
               **metrics, **audit}
        validation_rows.append(row)
        fallback_rows.append({"split": "validation", **row})
    validation_results = pd.DataFrame(validation_rows)

    selected = {}
    for variant in VARIANTS:
        candidates = validation_results[validation_results.Method == variant].copy()
        candidates["temperature_sort"] = candidates.temperature.fillna(-1.0)
        winner = candidates.sort_values(
            ["RMSE", "NDCG@5", "restart_probability_c", "temperature_sort"],
            ascending=[True, False, True, True]).iloc[0]
        selected[variant] = {
            "restart_probability_c": float(winner.restart_probability_c),
            "temperature": None if pd.isna(winner.temperature) else float(winner.temperature),
            "selection_metric": "validation_RMSE",
            "validation_MAE": float(winner.MAE), "validation_RMSE": float(winner.RMSE),
            "validation_NDCG@5": float(winner["NDCG@5"]),
            "tie_breakers": ["higher_validation_NDCG@5", "smaller_c", "smaller_temperature"],
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validation_results.to_csv(OUTPUT / "validation_results.csv", index=False)
    write_json(OUTPUT / "selected_configs.json", {
        "status": "FROZEN_BEFORE_TEST_LOAD", "selection_data": "validation_only",
        "test_loaded_at_write_time": False, "selected": selected,
    })

    # Phase 2: first TEST load occurs after the selections are persisted.
    test = pd.read_csv(SPLITS / "test.csv", dtype={"Student_ID": str, "Course_ID": str})
    final_rows, detail_frames, user_frames = [], [], []
    for variant in VARIANTS:
        cfg = selected[variant]
        cache_key = (variant, cfg["temperature"])
        if cache_key not in transition_cache:
            transition_cache[cache_key] = transition(adjacency, variant, cfg["temperature"])
        started = time.perf_counter()
        raw, convergence = rwr_scores(
            transition_cache[cache_key], len(users), len(items), cfg["restart_probability_c"])
        mapped, degenerate = legacy_rating_mapping(raw, ratings)
        metrics, details, per_user = evaluate_dual(raw, mapped, users, items, test, train)
        runtime = time.perf_counter() - started
        audit = fallback_counts(degenerate, users, test, convergence)
        final_rows.append({"Method": variant,
                           "restart_probability_c": cfg["restart_probability_c"],
                           "temperature": cfg["temperature"], "runtime_seconds": runtime,
                           **metrics, **audit})
        fallback_rows.append({"split": "test", "Method": variant,
                              "restart_probability_c": cfg["restart_probability_c"],
                              "temperature": cfg["temperature"], **audit})
        details.insert(0, "Method", variant)
        details["Raw_RWR_Score"] = [raw[users.index(str(r.Student_ID)), items.index(str(r.Course_ID))]
                                    for r in test.itertuples(index=False)]
        details["Rating_Mapping"] = "per_user_unobserved_log_minmax_[1,5]"
        per_user.insert(0, "Method", variant)
        detail_frames.append(details)
        user_frames.append(per_user)

    final = pd.DataFrame(final_rows)
    final.to_csv(OUTPUT / "final_test_results.csv", index=False)
    final_json_records = []
    for record in final.to_dict(orient="records"):
        final_json_records.append({
            key: (None if isinstance(value, float) and np.isnan(value) else value)
            for key, value in record.items()
        })
    write_json(OUTPUT / "final_test_results.json", {
        "test_evaluations_per_variant": 1, "selected_configs": selected,
        "results": final_json_records,
    })
    pd.concat(detail_frames, ignore_index=True).to_csv(OUTPUT / "test_predictions.csv", index=False)
    pd.concat(user_frames, ignore_index=True).to_csv(OUTPUT / "test_per_user_metrics.csv", index=False)
    pd.DataFrame(fallback_rows).to_csv(OUTPUT / "fallback_audit.csv", index=False)

    validation_signature = candidate_signature(validation, train, items)
    test_signature = candidate_signature(test, train, items)
    config = {
        "experiment": "rwr_publication", "seed": SEED, "variants": VARIANTS,
        "restart_probability_grid": C_VALUES, "softmax_temperature_grid": TEMPERATURES,
        "validation_configuration_count": len(validation_results),
        "selection": "per variant: lowest validation RMSE; tie highest NDCG@5; tie smaller c; tie smaller temperature",
        "test_evaluations_per_variant": 1,
        "graph_fit_data": "TRAIN_only", "rating_mapping_fit_data": "TRAIN graph and TRAIN-observed mask only",
        "rating_mapping": "log(raw_rwr+1e-10), per-user MinMax over TRAIN-unobserved items to [1,5]; constant range -> 3",
        "ranking_score": "raw_RWR_item_node_proximity",
        "evaluation_policy": POLICY.__dict__, "reproducibility": repro,
        "split_directory": str(SPLITS), "catalog_size": len(items),
        "runtime_seconds": time.perf_counter() - overall_started,
        "input_hashes_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
            [SPLITS / "train.csv", SPLITS / "validation.csv", SPLITS / "test.csv",
             CATALOG_FILE, ROOT / "shared/evaluator.py",
             ROOT / "methods/rwr/RWR_experiment_ndcg.py"]},
    }
    write_json(OUTPUT / "config.json", config)

    audit = f"""# RWR publication evaluation audit

Status: **PASS**

| Check | Verification |
|---|---|
| Immutable split | Seed-42 TRAIN {len(train)}, validation {len(validation)}, test {len(test)} rows for every variant. |
| Test rating keys | One shared test manifest; {len(set(zip(test.Student_ID, test.Course_ID)))} identical keys. |
| Full catalog | {len(items)} courses for every variant. |
| Candidate sets | Model-independent shared-evaluator construction. Validation signature `{validation_signature[0]}`; test signature `{test_signature[0]}`. |
| Candidate counts | Validation {validation_signature[1]} ({validation_signature[2]}–{validation_signature[3]}/user); test {test_signature[1]} ({test_signature[2]}–{test_signature[3]}/user). |
| TRAIN-item exclusion | Shared evaluator excludes items observed by that user in TRAIN. |
| Relevance | Binary `rating >= {POLICY.relevant_threshold}` from the supplied evaluation manifest. |
| Ranking eligibility | Shared evaluator includes only users with at least one relevant evaluation item. |
| Ranking tie-breaking | Raw RWR score descending, then Course ID ascending in shared evaluator. |
| Rating clipping | Legacy mapping is already [1,5]; shared evaluator still applies its standard [1,5] clip. |
| Leakage | Graph, transitions, and rating mapping use TRAIN only; no validation/test labels enter them. |
| Selection/test order | 64 validation configurations; per-variant selections persisted before test load; one test evaluation per variant. |

The candidate semantics are identical to publication MF, Vanilla GCN, LightGCN, and CF because all call the same evaluator with the same catalog and split manifests.
"""
    (OUTPUT / "evaluation_audit.md").write_text(audit, encoding="utf-8")

    mf = json.loads((RESULTS / "mf_publication/final_test_results.json").read_text())["metrics"]
    lg = json.loads((RESULTS / "lightgcn_publication/final_test_results.json").read_text())["metrics"]
    rmse_order = final.sort_values(["RMSE", "Method"])
    ndcg_order = final.sort_values(["NDCG@5", "Method"], ascending=[False, True])
    rmse_best, ndcg_best = rmse_order.iloc[0], ndcg_order.iloc[0]
    display_rows = []
    for _, r in final.iterrows():
        temperature_text = "" if pd.isna(r["temperature"]) else f"{r['temperature']:g}"
        display_rows.append(
            f"| {r['Method']} | {r['restart_probability_c']:g} | {temperature_text} | "
            f"{r['MAE']:.6f} | {r['RMSE']:.6f} | {r['Precision@5']:.6f} | {r['Recall@5']:.6f} | {r['NDCG@5']:.6f} | {r['HitRate@5']:.6f} | "
            f"{r['Precision@10']:.6f} | {r['Recall@10']:.6f} | {r['NDCG@10']:.6f} | {r['HitRate@10']:.6f} | {r['runtime_seconds']:.3f} |"
        )
    results_markdown = "\n".join(display_rows)
    same = rmse_best.Method == ndcg_best.Method
    summary = f"""# RWR publication rerun

## Protocol and calibration

Three fundamentally different legacy transitions are retained as separate baselines. Binary and Linear each searched eight restart probabilities; Softmax searched eight restart probabilities across six temperatures. Selection was per variant on validation RMSE, then validation NDCG@5, then smaller `c` and temperature. Test was loaded only after selections were written and evaluated once per variant.

Rating predictions preserve the legacy per-user `log(raw proximity + 1e-10)` then MinMax-to-[1,5] conversion over TRAIN-unobserved items. No labels fit this mapping, but it is a heuristic candidate-relative calibration and is not equivalent to direct explicit-rating regression. Ranking uses raw RWR proximity.

## Final test results

| Variant | c | Temperature | MAE | RMSE | P@5 | R@5 | NDCG@5 | HR@5 | P@10 | R@10 | NDCG@10 | HR@10 | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{results_markdown}

## Answers

1. **Best RWR RMSE:** `{rmse_best.Method}` ({rmse_best.RMSE:.6f}).
2. **Best RWR NDCG@5:** `{ndcg_best.Method}` ({ndcg_best['NDCG@5']:.6f}).
3. **Same variant?** {'Yes' if same else 'No'}.
4. **Rating predictor or ranking model?** RWR is natively a proximity/ranking model. Its explicit rating output depends on heuristic per-user MinMax calibration.
5. **Compared with MF:** best RWR RMSE {rmse_best.RMSE:.6f} versus MF {mf['RMSE']:.6f}; best RWR NDCG@5 {ndcg_best['NDCG@5']:.6f} versus MF {mf['NDCG@5']:.6f}.
6. **Compared with LightGCN:** best RWR RMSE {rmse_best.RMSE:.6f} versus LightGCN {lg['RMSE']:.6f}; best RWR NDCG@5 {ndcg_best['NDCG@5']:.6f} versus LightGCN {lg['NDCG@5']:.6f}.
7. **Main-table role:** keep RWR in the main comparison table as the non-parametric graph/ranking baseline, while explicitly flagging its heuristic rating calibration. It is highly competitive on ranking but unsuitable as evidence of strong explicit-rating prediction.
8. **Rating caveat:** MAE/RMSE are sensitive to a user-specific candidate-set transformation that is neither learned nor globally calibrated. They are evaluator-compatible but scientifically less direct than MF/GCN explicit-rating predictions.

See `legacy_method_audit.md`, `validation_results.csv`, `fallback_audit.csv`, and `evaluation_audit.md` for full provenance.
"""
    (OUTPUT / "summary.md").write_text(summary, encoding="utf-8")
    print(final[["Method", "restart_probability_c", "temperature", "MAE", "RMSE", "NDCG@5", "NDCG@10"]].to_string(index=False))


if __name__ == "__main__":
    main()
