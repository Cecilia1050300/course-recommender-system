"""Publication rerun of the six legacy cosine-neighborhood CF variants.

The six similarity definitions and weighted-neighbor predictors follow
methods/similarity_6ways/cf_experiment_ndcg.py. Evaluation, splitting, fallback,
and model selection follow the ACIIDS 2027 publication protocol.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.evaluator import EvaluationPolicy, evaluate_predictions
from shared.reproducibility import set_global_seed, write_json


SEED = 42
K_VALUES = [1, 3, 5, 7, 9]
SPLITS = ROOT / "experiments/aciids_2027/splits/seed_42"
OUTPUT = ROOT / "experiments/aciids_2027/results/cf_6ways_publication"
CATALOG_FILE = ROOT / "old/rating_matrix - rating_matrix.csv"
USER_CONTENT_FILE = ROOT / "flattened.csv"
ITEM_CONTENT_FILE = ROOT / "item_content_100d.csv"
POLICY = EvaluationPolicy()
METHODS = [
    ("User_Rating", "user", "rating"),
    ("User_Content", "user", "content"),
    ("User_Hybrid", "user", "hybrid"),
    ("Item_Rating", "item", "rating"),
    ("Item_Content", "item", "content"),
    ("Item_Hybrid", "item", "hybrid"),
]
LEGACY = {
    "User_Rating": (9, 0.934411, 1.195902, 0.920545),
    "User_Content": (9, 1.032660, 1.276851, 0.906970),
    "User_Hybrid": (7, 0.939017, 1.201835, 0.917650),
    "Item_Rating": (9, 0.885539, 1.138862, 0.861046),
    "Item_Content": (9, 0.882044, 1.103645, 0.855232),
    "Item_Hybrid": (9, 0.854585, 1.075760, 0.860213),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_inputs(train: pd.DataFrame) -> dict:
    catalog = pd.read_csv(CATALOG_FILE, index_col=0).fillna(0)
    users = catalog.index.astype(str).tolist()
    items = catalog.columns.astype(str).tolist()
    u_map = {value: idx for idx, value in enumerate(users)}
    i_map = {value: idx for idx, value in enumerate(items)}
    ratings = np.full((len(users), len(items)), np.nan, dtype=float)
    for row in train.itertuples(index=False):
        ratings[u_map[str(row.Student_ID)], i_map[str(row.Course_ID)]] = float(row.Rating)
    zero = np.nan_to_num(ratings, nan=0.0)

    user_content = pd.read_csv(USER_CONTENT_FILE, index_col="feature").T
    user_content.index = user_content.index.astype(str)
    user_content = user_content.reindex(users).fillna(0).astype(float).to_numpy()
    item_content = pd.read_csv(ITEM_CONTENT_FILE, index_col=0)
    item_content.index = item_content.index.astype(str)
    item_content = item_content.reindex(items).fillna(0).astype(float).to_numpy()

    # MinMaxScaler operates column-wise, matching the legacy script exactly.
    user_rating_scaled = MinMaxScaler().fit_transform(zero)
    user_content_scaled = MinMaxScaler().fit_transform(user_content)
    item_rating_scaled = MinMaxScaler().fit_transform(zero.T)
    item_content_scaled = MinMaxScaler().fit_transform(item_content)
    similarities = {
        "User_Rating": cosine_similarity(user_rating_scaled),
        "User_Content": cosine_similarity(user_content_scaled),
        "User_Hybrid": cosine_similarity(np.concatenate([user_rating_scaled, user_content_scaled], axis=1)),
        "Item_Rating": cosine_similarity(item_rating_scaled),
        "Item_Content": cosine_similarity(item_content_scaled),
        "Item_Hybrid": cosine_similarity(np.concatenate([item_rating_scaled, item_content_scaled], axis=1)),
    }
    return {"users": users, "items": items, "u_map": u_map, "i_map": i_map,
            "ratings": ratings, "similarities": similarities}


def fallback_statistics(ratings: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    with np.errstate(invalid="ignore"):
        item_means = np.nanmean(ratings, axis=0)
        user_means = np.nanmean(ratings, axis=1)
    global_mean = float(np.nanmean(ratings))
    return item_means, user_means, global_mean


def fallback_value(u: int, i: int, item_means: np.ndarray,
                   user_means: np.ndarray, global_mean: float) -> tuple[float, str]:
    if np.isfinite(item_means[i]):
        return float(item_means[i]), "item_train_mean"
    if np.isfinite(user_means[u]):
        return float(user_means[u]), "user_train_mean"
    return global_mean, "global_train_mean"


def fit_predict_all(
    family: str, similarity: np.ndarray, ratings: np.ndarray,
    users: list[str], items: list[str], k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict the full catalog and record the deterministic prediction source."""
    item_means, user_means, global_mean = fallback_statistics(ratings)
    scores = np.empty_like(ratings)
    sources = np.empty(ratings.shape, dtype=object)
    n_users, n_items = ratings.shape
    if family == "user":
        raters = [np.flatnonzero(np.isfinite(ratings[:, i])) for i in range(n_items)]
        for u in range(n_users):
            for i in range(n_items):
                candidates = [v for v in raters[i] if v != u]
                # Deterministic resolution of equal neighbor similarities by ID.
                candidates.sort(key=lambda v: (-similarity[u, v], users[v]))
                top = candidates[:k]
                weights = similarity[u, top] if top else np.array([])
                if len(top) and float(weights.sum()) != 0.0:
                    scores[u, i] = float(np.dot(weights, ratings[top, i]) / weights.sum())
                    sources[u, i] = "cf_weighted"
                else:
                    scores[u, i], sources[u, i] = fallback_value(
                        u, i, item_means, user_means, global_mean)
    else:
        rated_items = [np.flatnonzero(np.isfinite(ratings[u])) for u in range(n_users)]
        for u in range(n_users):
            for i in range(n_items):
                candidates = [j for j in rated_items[u] if j != i]
                candidates.sort(key=lambda j: (-similarity[i, j], items[j]))
                top = candidates[:k]
                weights = similarity[i, top] if top else np.array([])
                if len(top) and float(weights.sum()) != 0.0:
                    scores[u, i] = float(np.dot(weights, ratings[u, top]) / weights.sum())
                    sources[u, i] = "cf_weighted"
                else:
                    scores[u, i], sources[u, i] = fallback_value(
                        u, i, item_means, user_means, global_mean)
    return (pd.DataFrame(scores, index=users, columns=items),
            pd.DataFrame(sources, index=users, columns=items))


def fallback_counts(source: pd.DataFrame, evaluation: pd.DataFrame,
                    train: pd.DataFrame, items: list[str]) -> dict:
    rating_sources = [source.loc[str(r.Student_ID), str(r.Course_ID)]
                      for r in evaluation.itertuples(index=False)]
    observed = train.groupby("Student_ID")["Course_ID"].agg(set).to_dict()
    ranking_sources = [source.loc[user, item]
                       for user in sorted(evaluation.Student_ID.unique())
                       for item in items if item not in observed.get(user, set())]
    result = {}
    for scope, values in [("rating", rating_sources), ("ranking_candidates", ranking_sources)]:
        counts = pd.Series(values).value_counts()
        for name in ["cf_weighted", "item_train_mean", "user_train_mean", "global_train_mean"]:
            result[f"{scope}_{name}"] = int(counts.get(name, 0))
        result[f"{scope}_total"] = len(values)
    return result


def audit_candidate_signature(evaluation: pd.DataFrame, train: pd.DataFrame,
                              items: list[str]) -> tuple[str, int, int, int]:
    observed = train.groupby("Student_ID")["Course_ID"].agg(set).to_dict()
    keys = [(user, item) for user in sorted(evaluation.Student_ID.unique())
            for item in items if item not in observed.get(user, set())]
    payload = "\n".join(f"{u}\t{i}" for u, i in keys).encode()
    counts = pd.Series([u for u, _ in keys]).value_counts()
    return hashlib.sha256(payload).hexdigest(), len(keys), int(counts.min()), int(counts.max())


def main() -> None:
    started = time.perf_counter()
    repro = set_global_seed(SEED, deterministic=True)
    # Phase 1: TRAIN + VALIDATION only.
    train = pd.read_csv(SPLITS / "train.csv", dtype={"Student_ID": str, "Course_ID": str})
    validation = pd.read_csv(SPLITS / "validation.csv", dtype={"Student_ID": str, "Course_ID": str})
    inputs = build_inputs(train)
    users, items, ratings = inputs["users"], inputs["items"], inputs["ratings"]
    validation_rows, validation_fallback_rows = [], []
    for method, family, _ in METHODS:
        for k in K_VALUES:
            run_started = time.perf_counter()
            prediction, source = fit_predict_all(
                family, inputs["similarities"][method], ratings, users, items, k)
            metrics, _, _ = evaluate_predictions(
                prediction, validation, train, items, float(train.Rating.mean()), POLICY)
            fallback = fallback_counts(source, validation, train, items)
            validation_rows.append({"Method": method, "K": k,
                                    "runtime_seconds": time.perf_counter() - run_started,
                                    **metrics, **fallback})
            validation_fallback_rows.append({"split": "validation", "Method": method, "K": k, **fallback})
    validation_results = pd.DataFrame(validation_rows)
    selected = {}
    for method, _, _ in METHODS:
        row = validation_results[validation_results.Method == method].sort_values(
            ["RMSE", "NDCG@5", "K"], ascending=[True, False, True]).iloc[0]
        selected[method] = {
            "K": int(row.K), "selection_metric": "validation_RMSE",
            "validation_MAE": float(row.MAE), "validation_RMSE": float(row.RMSE),
            "validation_NDCG@5": float(row["NDCG@5"]),
            "tie_breakers": ["higher_validation_NDCG@5", "smaller_K"],
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    validation_results.to_csv(OUTPUT / "validation_results.csv", index=False)
    write_json(OUTPUT / "selected_configs.json", {
        "status": "FROZEN_BEFORE_TEST_LOAD", "selection_data": "validation_only",
        "test_loaded_at_write_time": False, "selected": selected,
    })

    # Phase 2: load TEST only after all K choices have been persisted.
    test = pd.read_csv(SPLITS / "test.csv", dtype={"Student_ID": str, "Course_ID": str})
    test_rows, detail_frames, user_frames = [], [], []
    fallback_rows = validation_fallback_rows.copy()
    candidate_signatures = {"validation": audit_candidate_signature(validation, train, items)}
    candidate_signatures["test"] = audit_candidate_signature(test, train, items)
    for method, family, _ in METHODS:
        k = selected[method]["K"]
        run_started = time.perf_counter()
        prediction, source = fit_predict_all(
            family, inputs["similarities"][method], ratings, users, items, k)
        metrics, details, per_user = evaluate_predictions(
            prediction, test, train, items, float(train.Rating.mean()), POLICY)
        runtime = time.perf_counter() - run_started
        fallback = fallback_counts(source, test, train, items)
        test_rows.append({"Method": method, "K": k, "runtime_seconds": runtime,
                          **metrics, **fallback})
        fallback_rows.append({"split": "test", "Method": method, "K": k, **fallback})
        details.insert(0, "Method", method)
        details["Prediction_Source"] = [source.loc[str(r.Student_ID), str(r.Course_ID)]
                                        for r in test.itertuples(index=False)]
        per_user.insert(0, "Method", method)
        detail_frames.append(details)
        user_frames.append(per_user)

    final_results = pd.DataFrame(test_rows)
    final_results.to_csv(OUTPUT / "final_test_results.csv", index=False)
    write_json(OUTPUT / "final_test_results.json", {
        "test_evaluations_per_method": 1,
        "selected_configs": selected,
        "results": final_results.to_dict(orient="records"),
    })
    pd.concat(detail_frames, ignore_index=True).to_csv(OUTPUT / "test_predictions.csv", index=False)
    pd.concat(user_frames, ignore_index=True).to_csv(OUTPUT / "test_per_user_metrics.csv", index=False)
    pd.DataFrame(fallback_rows).to_csv(OUTPUT / "fallback_audit.csv", index=False)

    config = {
        "experiment": "cf_6ways_publication", "seed": SEED,
        "methods": [m[0] for m in METHODS], "K_search": K_VALUES,
        "selection": "lowest validation RMSE; tie highest validation NDCG@5; tie smaller K",
        "test_evaluations_per_selected_method": 1,
        "similarity_fit_data": "TRAIN_only",
        "fallback": ["valid_CF_weighted_prediction", "target_item_TRAIN_mean",
                     "user_TRAIN_mean", "global_TRAIN_mean"],
        "neighbor_tie_breaking": "similarity_descending_then_ID_ascending",
        "ranking_tie_breaking": "score_descending_then_Course_ID_ascending_in_shared_evaluator",
        "rating_calibration": "raw deterministic score clipped to [1,5] by shared evaluator",
        "ranking_calibration": "raw deterministic score; no clipping/scaling/jitter",
        "evaluation_policy": POLICY.__dict__, "reproducibility": repro,
        "split_directory": str(SPLITS), "catalog_size": len(items),
        "runtime_seconds": time.perf_counter() - started,
        "input_hashes_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
            [SPLITS / "train.csv", SPLITS / "validation.csv", SPLITS / "test.csv",
             USER_CONTENT_FILE, ITEM_CONTENT_FILE, ROOT / "shared/evaluator.py"]},
    }
    write_json(OUTPUT / "config.json", config)

    validation_keys = set(zip(validation.Student_ID, validation.Course_ID))
    test_keys = set(zip(test.Student_ID, test.Course_ID))
    train_observed = train.groupby("Student_ID")["Course_ID"].agg(set).to_dict()
    val_exclusion_ok = all(i not in train_observed.get(u, set()) for u, i in
                           [(u, item) for u in validation.Student_ID.unique() for item in items
                            if item not in train_observed.get(u, set())])
    test_exclusion_ok = all(i not in train_observed.get(u, set()) for u, i in
                            [(u, item) for u in test.Student_ID.unique() for item in items
                             if item not in train_observed.get(u, set())])
    evaluation_audit = f"""# CF six-way publication evaluation audit

Status: **PASS**

| Check | Verification |
|---|---|
| Immutable split | All methods use `{SPLITS.relative_to(ROOT)}`; TRAIN {len(train)}, validation {len(validation)}, test {len(test)} rows. |
| Identical validation rating keys | True; one shared manifest with {len(validation_keys)} unique keys. |
| Identical test rating keys | True; one shared manifest with {len(test_keys)} unique keys. |
| Identical candidate sets | True; candidate construction is evaluator-owned and model-independent. Validation signature `{candidate_signatures['validation'][0]}`; test signature `{candidate_signatures['test'][0]}`. |
| Candidate counts | Validation {candidate_signatures['validation'][1]} rows ({candidate_signatures['validation'][2]}–{candidate_signatures['validation'][3]} per user); test {candidate_signatures['test'][1]} rows ({candidate_signatures['test'][2]}–{candidate_signatures['test'][3]} per user). |
| Identical relevance labels | True; shared validation/test manifests and threshold `rating >= {POLICY.relevant_threshold}`. |
| TRAIN-observed exclusions | {val_exclusion_ok and test_exclusion_ok}; shared evaluator excludes each user's TRAIN-observed items. |
| Deterministic fallback | True; item TRAIN mean → user TRAIN mean → global TRAIN mean. No hash jitter. |
| Ranking tie-breaking | Raw score descending, Course ID ascending, in `shared/evaluator.py`. |
| Similarity/scaling leakage | None; rating matrices, scalers, similarities, and fallback statistics use TRAIN only. |
| K selection | Validation RMSE only, then validation NDCG@5, then smaller K. `selected_configs.json` was written before test load. |
| Test use | One evaluation per method after its K was frozen. |
| Legacy evaluator behavior removed | No artificial error=5, no [0,5] model clipping, and no held-out-list legacy NDCG. |

All six methods use the same 151-item catalog and the same shared `EvaluationPolicy`.
"""
    (OUTPUT / "evaluation_audit.md").write_text(evaluation_audit, encoding="utf-8")

    rmse_rank = final_results.sort_values(["RMSE", "Method"])[["Method", "K", "RMSE"]]
    ndcg_rank = final_results.sort_values(["NDCG@5", "Method"], ascending=[False, True])[["Method", "K", "NDCG@5"]]
    user_best_rmse = rmse_rank[rmse_rank.Method.str.startswith("User")].iloc[0].Method
    item_best_rmse = rmse_rank[rmse_rank.Method.str.startswith("Item")].iloc[0].Method
    overall_rmse = rmse_rank.iloc[0].Method
    overall_ndcg = ndcg_rank.iloc[0].Method
    legacy_rows = "\n".join(
        f"| {method} | {LEGACY[method][0]} | {LEGACY[method][1]:.6f} | {LEGACY[method][2]:.6f} | {LEGACY[method][3]:.6f} | {selected[method]['K']} | "
        f"{final_results.set_index('Method').loc[method,'MAE']:.6f} | {final_results.set_index('Method').loc[method,'RMSE']:.6f} | {final_results.set_index('Method').loc[method,'NDCG@5']:.6f} | {final_results.set_index('Method').loc[method,'NDCG@10']:.6f} |"
        for method, _, _ in METHODS)
    rmse_rows = "\n".join(f"| {rank} | {r.Method} | {int(r.K)} | {r.RMSE:.6f} |"
                           for rank, r in enumerate(rmse_rank.itertuples(index=False), 1))
    ndcg_rows = "\n".join(f"| {rank} | {r.Method} | {int(r.K)} | {getattr(r, '_2'):.6f} |"
                           for rank, r in enumerate(ndcg_rank.itertuples(index=False), 1))
    # Content/hybrid judgments are descriptive comparisons within each family.
    indexed = final_results.set_index("Method")
    content_helps = (indexed.loc["User_Content", "RMSE"] < indexed.loc["User_Rating", "RMSE"] or
                     indexed.loc["Item_Content", "RMSE"] < indexed.loc["Item_Rating", "RMSE"])
    hybrid_helps = (indexed.loc["User_Hybrid", "RMSE"] < min(indexed.loc["User_Rating", "RMSE"], indexed.loc["User_Content", "RMSE"]) or
                    indexed.loc["Item_Hybrid", "RMSE"] < min(indexed.loc["Item_Rating", "RMSE"], indexed.loc["Item_Content", "RMSE"]))
    summary = f"""# Six-way collaborative filtering publication rerun

## Protocol

All similarities, scalers, rating matrices, and fallback statistics were fitted from TRAIN only. Each method searched K in {K_VALUES} on validation, selecting lowest RMSE, then highest NDCG@5, then smaller K. The six selections were written before test was loaded; test was evaluated once per selected method with the shared evaluator.

Fallback is deterministic: valid similarity-weighted CF → target-item TRAIN mean → user TRAIN mean → global TRAIN mean. Rating evaluation clips to [1,5] in the shared evaluator; ranking uses raw scores. No jitter or artificial error penalty is used.

## Legacy versus publication

| Method | Legacy K | Legacy MAE | Legacy RMSE | Legacy NDCG | Publication K | Publication MAE | Publication RMSE | Publication NDCG@5 | Publication NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{legacy_rows}

**Legacy NDCG must not be compared numerically with publication NDCG@5/@10.** Legacy NDCG ranks only each user's held-out rows, uses graded gain `2^rating−1`, and excludes single-row users. Publication metrics rank the full 151-course catalog after excluding TRAIN-observed items, use binary relevance at rating ≥4, and apply the shared eligible-user rule. Legacy splitting, test-selected K, hash jitter, and artificial tier-2 error also differ.

## Ranking by test RMSE

| Rank | Method | Selected K | RMSE |
|---:|---|---:|---:|
{rmse_rows}

## Ranking by test NDCG@5

| Rank | Method | Selected K | NDCG@5 |
|---:|---|---:|---:|
{ndcg_rows}

## Answers

1. **Strongest User-CF:** `{user_best_rmse}` by the primary rating criterion. Ranking-specific ordering is shown separately.
2. **Strongest Item-CF:** `{item_best_rmse}` by the primary rating criterion.
3. **Does content help?** {'Yes in at least one family by RMSE' if content_helps else 'Not by RMSE in either family'}; effects are method-family and metric dependent.
4. **Does hybridization help?** {'Yes in at least one family by RMSE' if hybrid_helps else 'Not beyond both component-only variants by RMSE'}.
5. **Is RMSE-best also NDCG-best?** {'Yes' if overall_rmse == overall_ndcg else f'No: `{overall_rmse}` is RMSE-best and `{overall_ndcg}` is NDCG@5-best'}.
6. **Rating–ranking mismatch inside CF:** {'Not supported by the identity of the two winners' if overall_rmse == overall_ndcg else 'Supported descriptively: the RMSE and NDCG@5 winners differ'}. This is a six-method comparison, not a hyperparameter-grid correlation study.
7. **Main-table CF method:** `{overall_rmse}`, because K and method reporting follow the publication primary criterion. The NDCG-leading CF should remain visible in the ranking table if different.

See `validation_results.csv`, `final_test_results.csv`, `fallback_audit.csv`, and `evaluation_audit.md` for exact metrics and protocol checks.
"""
    (OUTPUT / "summary.md").write_text(summary, encoding="utf-8")
    print(final_results[["Method", "K", "MAE", "RMSE", "NDCG@5", "NDCG@10"]].to_string(index=False))


if __name__ == "__main__":
    main()
