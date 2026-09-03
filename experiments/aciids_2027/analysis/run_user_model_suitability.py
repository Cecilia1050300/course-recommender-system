"""Post-hoc user suitability analysis over frozen, saved publication artifacts.

This script never trains or runs a recommender. Explanatory features are derived
exclusively from the seed-42 training split. Saved test per-user metrics provide
diagnostic labels only.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/aciids_2027/analysis/user_model_suitability"
TRAIN = ROOT / "experiments/aciids_2027/splits/seed_42/train.csv"
MF_DIR = ROOT / "experiments/aciids_2027/results/mf_publication"
LG_DIR = ROOT / "experiments/aciids_2027/results/lightgcn_publication"
ROBUSTNESS_DIR = ROOT / "experiments/aciids_2027/robustness"

FEATURES = [
    "train_interaction_count", "fraction_of_catalog_observed", "mean_rating",
    "rating_std", "rating_min", "rating_max", "number_of_distinct_rating_levels",
    "fraction_rating_1", "fraction_rating_2", "fraction_rating_3",
    "fraction_rating_4", "fraction_rating_5", "fraction_rating_ge_4",
    "fraction_rating_lt_4", "rating_entropy", "rating_range",
    "number_of_observed_ordered_pairs", "normalized_pair_count_per_interaction",
    "mean_rating_gap_across_pairs", "max_rating_gap", "fraction_gap_1_pairs",
    "fraction_gap_2plus_pairs", "mean_degree_of_interacted_items",
    "median_degree_of_interacted_items", "min_degree_of_interacted_items",
    "max_degree_of_interacted_items", "fraction_of_interactions_with_head_items",
    "fraction_of_interactions_with_tail_items",
]


def bh_adjust(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    good = values.dropna().sort_values()
    m = len(good)
    if not m:
        return out
    raw = good.to_numpy() * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(raw[::-1])[::-1].clip(max=1.0)
    out.loc[good.index] = adj
    return out


def build_features(train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    item_degree = train.groupby("Course_ID").size().sort_index()
    ranked = sorted(item_degree.index, key=lambda item: (-int(item_degree[item]), str(item)))
    n_items = len(ranked)
    boundary = math.ceil(0.20 * n_items)
    head = set(ranked[:boundary])
    tail = set(ranked[-boundary:])
    rows = []
    for user, group in train.groupby("Student_ID", sort=True):
        ratings = group["Rating"].astype(float).to_numpy()
        n = len(ratings)
        counts = {r: int(np.sum(ratings == r)) for r in range(1, 6)}
        probs = np.array([counts[r] / n for r in range(1, 6)])
        nonzero = probs[probs > 0]
        gaps = []
        for high in range(2, 6):
            for low in range(1, high):
                gaps.extend([high - low] * (counts[high] * counts[low]))
        gaps_a = np.asarray(gaps, dtype=float)
        degrees = group["Course_ID"].map(item_degree).astype(float).to_numpy()
        rows.append({
            "Student_ID": user,
            "train_interaction_count": n,
            "fraction_of_catalog_observed": n / n_items,
            "mean_rating": ratings.mean(),
            "rating_std": ratings.std(ddof=0),
            "rating_min": ratings.min(),
            "rating_max": ratings.max(),
            "number_of_distinct_rating_levels": len(np.unique(ratings)),
            **{f"fraction_rating_{r}": counts[r] / n for r in range(1, 6)},
            "fraction_rating_ge_4": float(np.mean(ratings >= 4)),
            "fraction_rating_lt_4": float(np.mean(ratings < 4)),
            "rating_entropy": float(-(nonzero * np.log2(nonzero)).sum()),
            "rating_range": ratings.max() - ratings.min(),
            "number_of_observed_ordered_pairs": len(gaps),
            "normalized_pair_count_per_interaction": len(gaps) / n,
            "mean_rating_gap_across_pairs": gaps_a.mean() if len(gaps_a) else 0.0,
            "max_rating_gap": gaps_a.max() if len(gaps_a) else 0.0,
            "fraction_gap_1_pairs": float(np.mean(gaps_a == 1)) if len(gaps_a) else 0.0,
            "fraction_gap_2plus_pairs": float(np.mean(gaps_a >= 2)) if len(gaps_a) else 0.0,
            "mean_degree_of_interacted_items": degrees.mean(),
            "median_degree_of_interacted_items": np.median(degrees),
            "min_degree_of_interacted_items": degrees.min(),
            "max_degree_of_interacted_items": degrees.max(),
            "fraction_of_interactions_with_head_items": group["Course_ID"].isin(head).mean(),
            "fraction_of_interactions_with_tail_items": group["Course_ID"].isin(tail).mean(),
        })
    metadata = {
        "catalog_items": n_items,
        "head_items": len(head),
        "tail_items": len(tail),
        "head_min_degree": int(min(item_degree.loc[list(head)])),
        "tail_max_degree": int(max(item_degree.loc[list(tail)])),
    }
    return pd.DataFrame(rows), metadata


def load_per_user(path: Path, prefix: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    keep = ["Student_ID", "User_MAE", "User_RMSE", "User_Rating_Rows",
            "Relevant_Items", "Candidates", "Precision@5", "Recall@5",
            "NDCG@5", "HitRate@5"]
    data = data[keep].copy()
    return data.rename(columns={c: f"{prefix}_{c}" for c in keep if c != "Student_ID"})


def make_comparison(features: pd.DataFrame) -> pd.DataFrame:
    mf = load_per_user(MF_DIR / "test_per_user_metrics.csv", "MF")
    lg = load_per_user(LG_DIR / "test_per_user_metrics.csv", "LightGCN")
    out = features.merge(mf, on="Student_ID", how="inner").merge(lg, on="Student_ID", how="inner")
    out["MF_MAE_user"] = out["MF_User_MAE"]
    out["LightGCN_MAE_user"] = out["LightGCN_User_MAE"]
    out["MAE_difference"] = out["LightGCN_MAE_user"] - out["MF_MAE_user"]
    out["rating_winner"] = np.select(
        [out["MAE_difference"] > 0, out["MAE_difference"] < 0],
        ["MF", "LightGCN"], default="tie")
    out["MF_rating_better"] = (out["rating_winner"] == "MF").astype(int)
    out["LightGCN_rating_better"] = (out["rating_winner"] == "LightGCN").astype(int)
    out["ranking_eligible"] = out["MF_NDCG@5"].notna() & out["LightGCN_NDCG@5"].notna()
    out["MF_NDCG5_user"] = out["MF_NDCG@5"]
    out["LightGCN_NDCG5_user"] = out["LightGCN_NDCG@5"]
    out["NDCG5_difference"] = out["LightGCN_NDCG5_user"] - out["MF_NDCG5_user"]
    out["ranking_winner"] = np.select(
        [out["ranking_eligible"] & (out["NDCG5_difference"] > 0),
         out["ranking_eligible"] & (out["NDCG5_difference"] < 0),
         out["ranking_eligible"]],
        ["LightGCN", "MF", "tie"], default="ineligible")
    out["LightGCN_ranking_better"] = (out["ranking_winner"] == "LightGCN").astype(int)
    out["MF_ranking_better"] = (out["ranking_winner"] == "MF").astype(int)
    for metric in ["Recall@5", "HitRate@5"]:
        out[f"{metric}_difference"] = out[f"LightGCN_{metric}"] - out[f"MF_{metric}"]
        out[f"{metric}_winner"] = np.select(
            [out["ranking_eligible"] & (out[f"{metric}_difference"] > 0),
             out["ranking_eligible"] & (out[f"{metric}_difference"] < 0),
             out["ranking_eligible"]], ["LightGCN", "MF", "tie"], default="ineligible")
    desired = ["Student_ID"] + FEATURES + [
        "MF_MAE_user", "LightGCN_MAE_user", "MAE_difference", "rating_winner",
        "MF_rating_better", "LightGCN_rating_better", "ranking_eligible",
        "MF_NDCG5_user", "LightGCN_NDCG5_user", "NDCG5_difference",
        "ranking_winner", "MF_ranking_better", "LightGCN_ranking_better",
        "MF_Recall@5", "LightGCN_Recall@5", "Recall@5_difference", "Recall@5_winner",
        "MF_HitRate@5", "LightGCN_HitRate@5", "HitRate@5_difference", "HitRate@5_winner",
    ]
    return out[desired]


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) /
                       (len(a)+len(b)-2))
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0


def analyze_features(comp: pd.DataFrame) -> pd.DataFrame:
    records = []
    specs = [
        ("rating", "rating_winner", "MAE_difference", comp["rating_winner"].isin(["MF", "LightGCN"])),
        ("ranking_ndcg5", "ranking_winner", "NDCG5_difference", comp["ranking_eligible"] & comp["ranking_winner"].isin(["MF", "LightGCN"])),
    ]
    for analysis, winner_col, diff_col, mask in specs:
        data = comp.loc[mask]
        for feature in FEATURES:
            mf = data.loc[data[winner_col] == "MF", feature]
            lg = data.loc[data[winner_col] == "LightGCN", feature]
            try:
                u, p_u = mannwhitneyu(mf, lg, alternative="two-sided") if len(mf) and len(lg) else (np.nan, np.nan)
            except ValueError:
                u, p_u = np.nan, np.nan
            rho, p_s = spearmanr(data[feature], data[diff_col], nan_policy="omit")
            records.append({
                "analysis": analysis, "feature": feature,
                "mf_winner_n": len(mf), "mf_winner_mean": mf.mean(),
                "mf_winner_median": mf.median(), "mf_winner_std": mf.std(ddof=1),
                "lightgcn_winner_n": len(lg), "lightgcn_winner_mean": lg.mean(),
                "lightgcn_winner_median": lg.median(), "lightgcn_winner_std": lg.std(ddof=1),
                "cohens_d_lightgcn_minus_mf": cohens_d(mf, lg),
                "mann_whitney_u": u, "mann_whitney_p": p_u,
                "spearman_rho_with_difference": rho, "spearman_p": p_s,
            })
    result = pd.DataFrame(records)
    result["mann_whitney_p_bh"] = result.groupby("analysis")["mann_whitney_p"].transform(bh_adjust)
    result["spearman_p_bh"] = result.groupby("analysis")["spearman_p"].transform(bh_adjust)
    return result


def predictive_analysis(comp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = comp.loc[comp["ranking_eligible"] & comp["ranking_winner"].isin(["MF", "LightGCN"])].copy()
    y = (data["ranking_winner"] == "LightGCN").astype(int).to_numpy()
    min_class = int(pd.Series(y).value_counts().min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        return pd.DataFrame([{"record_type": "status", "model": "unavailable",
                              "detail": "Fewer than two users in one winner class."}]), {"n": len(data)}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_pre = ColumnTransformer([("numeric", Pipeline([
        ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), FEATURES)])
    count_pre = ColumnTransformer([("numeric", Pipeline([
        ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["train_interaction_count"])])
    models = {
        "majority_class": (Pipeline([("model", DummyClassifier(strategy="most_frequent"))]), []),
        "interaction_count_only": (Pipeline([("pre", count_pre), ("model", LogisticRegression(max_iter=2000, random_state=42))]), ["train_interaction_count"]),
        "logistic_all_features": (Pipeline([("pre", all_pre), ("model", LogisticRegression(max_iter=5000, random_state=42))]), FEATURES),
        "shallow_decision_tree": (Pipeline([("pre", ColumnTransformer([("numeric", SimpleImputer(strategy="median"), FEATURES)])),
                                             ("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=42))]), FEATURES),
    }
    rows = []
    X = data[FEATURES]
    for name, (model, names) in models.items():
        pred = cross_val_predict(model, X, y, cv=cv, method="predict")
        if name == "majority_class":
            prob = pred.astype(float)
        else:
            prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        rows.append({
            "record_type": "cv_performance", "model": name, "feature": "",
            "importance_type": "", "importance": np.nan, "n_users": len(y),
            "positive_lightgcn": int(y.sum()), "negative_mf": int((1-y).sum()),
            "cv_folds": n_splits, "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "roc_auc": roc_auc_score(y, prob),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "detail": "out-of-fold predictions",
        })
        if names:
            model.fit(X, y)
            fitted = model.named_steps["model"]
            vals = fitted.coef_[0] if hasattr(fitted, "coef_") else fitted.feature_importances_
            typ = "standardized_logistic_coefficient" if hasattr(fitted, "coef_") else "gini_importance"
            for feature, value in sorted(zip(names, vals), key=lambda z: abs(z[1]), reverse=True):
                rows.append({"record_type": "fitted_feature_importance", "model": name,
                             "feature": feature, "importance_type": typ,
                             "importance": value, "n_users": len(y), "detail": "fit on full diagnostic sample; descriptive only"})
    return pd.DataFrame(rows), {"n": len(data), "positive": int(y.sum()), "negative": int((1-y).sum()), "folds": n_splits}


def winner_counts(series: pd.Series, eligible_n: int) -> dict:
    counts = series.value_counts().to_dict()
    return {k: (int(counts.get(k, 0)), 100 * counts.get(k, 0) / eligible_n if eligible_n else np.nan)
            for k in ["MF", "LightGCN", "tie"]}


def fmt_counts(counts: dict) -> str:
    return ", ".join(f"{k} {n} ({p:.1f}%)" for k, (n, p) in counts.items())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN)
    features, pop_meta = build_features(train)
    features.to_csv(OUT / "user_level_features.csv", index=False)
    comp = make_comparison(features)
    comp.to_csv(OUT / "mf_vs_lightgcn_rmse_user_comparison.csv", index=False)

    ranking_cols = ["Student_ID", "MF_MAE_user", "LightGCN_MAE_user", "MAE_difference",
                    "MF_NDCG5_user", "LightGCN_NDCG5_user", "NDCG5_difference",
                    "ranking_eligible", "rating_winner", "ranking_winner",
                    "MF_Recall@5", "LightGCN_Recall@5", "Recall@5_difference",
                    "MF_HitRate@5", "LightGCN_HitRate@5", "HitRate@5_difference",
                    "availability_status"]
    unavailable = pd.DataFrame([{
        "availability_status": (
            "UNAVAILABLE: no saved per-user prediction or per-user metric artifact "
            "exists for the frozen LightGCN-ranking configuration; no inference rerun"
        )
    }]).reindex(columns=ranking_cols)
    unavailable.to_csv(OUT / "mf_vs_lightgcn_ranking_user_comparison.csv", index=False)

    fa = analyze_features(comp)
    fa.to_csv(OUT / "feature_analysis.csv", index=False)
    pred, pred_meta = predictive_analysis(comp)
    pred.to_csv(OUT / "suitability_prediction_results.csv", index=False)

    rating_counts = winner_counts(comp["rating_winner"], len(comp))
    eligible = comp.loc[comp["ranking_eligible"]]
    rank_counts = winner_counts(eligible["ranking_winner"], len(eligible))
    recall_counts = winner_counts(eligible["Recall@5_winner"], len(eligible))
    hit_counts = winner_counts(eligible["HitRate@5_winner"], len(eligible))
    perf = pred.loc[pred["record_type"] == "cv_performance"].set_index("model")
    rank_fa = fa[fa["analysis"] == "ranking_ndcg5"].copy()
    rank_fa["abs_rho"] = rank_fa["spearman_rho_with_difference"].abs()
    strongest = rank_fa.sort_values("abs_rho", ascending=False).head(5)
    count_rho = float(rank_fa.loc[rank_fa["feature"] == "train_interaction_count", "spearman_rho_with_difference"].iloc[0])
    majority_bacc = float(perf.loc["majority_class", "balanced_accuracy"])
    count_bacc = float(perf.loc["interaction_count_only", "balanced_accuracy"])
    all_bacc = float(perf.loc["logistic_all_features", "balanced_accuracy"])
    best_bacc = float(perf["balanced_accuracy"].max())
    signal = best_bacc - majority_bacc
    verdict_a = "SUPPORTED" if rank_counts["MF"][0] and rank_counts["LightGCN"][0] else "NOT SUPPORTED"
    verdict_b = "SUPPORTED" if count_bacc >= majority_bacc + 0.10 and count_bacc >= 0.65 else ("SUGGESTIVE" if count_bacc >= majority_bacc + 0.05 else "NOT SUPPORTED")
    verdict_c = "SUPPORTED" if all_bacc >= count_bacc + 0.05 and all_bacc >= majority_bacc + 0.05 else ("SUGGESTIVE" if all_bacc >= count_bacc + 0.025 else "NOT SUPPORTED")
    verdict_e = "SUPPORTED" if signal >= 0.10 else ("SUGGESTIVE" if signal >= 0.05 else "NOT SUPPORTED")
    verdict_f = "SUGGESTIVE" if all_bacc >= 0.65 and signal >= 0.10 else "NOT SUPPORTED"

    robust_files = list(ROBUSTNESS_DIR.glob("*per_user*")) + list(ROBUSTNESS_DIR.glob("*prediction*"))
    top_lines = "\n".join(
        f"- `{r.feature}`: Spearman ρ={r.spearman_rho_with_difference:.3f}, BH-adjusted p={r.spearman_p_bh:.4g}"
        for r in strongest.itertuples())
    model_lines = "\n".join(
        f"| {idx} | {row.accuracy:.3f} | {row.balanced_accuracy:.3f} | {row.roc_auc:.3f} | {row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} |"
        for idx, row in perf.iterrows())
    summary = f"""# User-level MF versus LightGCN suitability analysis

## Scope and artifact availability

This is a post-hoc, read-only diagnostic. It trained no recommender and ran no inference. Features use only `{TRAIN.relative_to(ROOT)}`. Outcomes use the frozen seed-42 `test_per_user_metrics.csv` files for publication MF and RMSE-selected LightGCN.

The frozen ranking-oriented LightGCN configuration has aggregate rows in the robustness outputs but **no saved per-user predictions or per-user metrics**. Consequently `mf_vs_lightgcn_ranking_user_comparison.csv` contains one status row with unavailable metric fields. No inference was rerun to fill it.

## TRAIN-only feature construction

- Users: {len(features)}; interactions: {len(train)}; TRAIN catalog: {pop_meta['catalog_items']} items.
- `rating_std` is population standard deviation (`ddof=0`). Entropy is Shannon entropy in bits over rating values 1–5.
- An observed ordered pair is an unordered pair of two courses from the same user whose ratings differ, oriented from the higher-rated course to the lower-rated course. Equal-rating pairs are excluded.
- Item degree is the number of TRAIN interactions. Head items are the first {pop_meta['head_items']} items (top 20%, rounded up) after sorting by descending TRAIN degree and then item ID; tail items are the last {pop_meta['tail_items']} items under that same deterministic ordering. The sets are disjoint. Head minimum degree={pop_meta['head_min_degree']}; tail maximum degree={pop_meta['tail_max_degree']}.
- `fraction_of_catalog_observed` uses the TRAIN catalog denominator only.

## Per-user outcomes: MF versus LightGCN-RMSE

Rating users (n={len(comp)}): {fmt_counts(rating_counts)}.

NDCG@5-eligible users (n={len(eligible)}): {fmt_counts(rank_counts)}. Ranking eligibility is exactly non-missing per-user metrics from both shared-evaluator artifacts; ineligible users are excluded.

Recall@5 winners: {fmt_counts(recall_counts)}. HitRate@5 winners: {fmt_counts(hit_counts)}.

Ties are retained descriptively and excluded from binary prediction. Rating differences are `LightGCN MAE − MF MAE` (positive favors MF). Ranking differences are `LightGCN NDCG@5 − MF NDCG@5` (positive favors LightGCN).

## Feature associations

`feature_analysis.csv` reports winner-group mean/median/sample SD, Cohen's d, Mann–Whitney U, and Spearman association with the continuous performance difference. Cohen's d is LightGCN-winner minus MF-winner, so its sign describes which winner group has the larger feature. Benjamini–Hochberg correction is applied separately within the rating and ranking analyses to the 28 Mann–Whitney tests and separately to the 28 Spearman tests.

The five largest absolute Spearman associations with LightGCN's NDCG@5 advantage are:

{top_lines}

Interaction count alone has ρ={count_rho:.3f} with NDCG@5 difference. Association is descriptive, not causal.

## Predictive feasibility

Binary diagnostic sample: n={pred_meta['n']} non-tied eligible users ({pred_meta['positive']} LightGCN winners, {pred_meta['negative']} MF winners); stratified {pred_meta['folds']}-fold CV, shuffled with seed 42. Each metric below comes from pooled out-of-fold predictions. The positive class is LightGCN winner.

| Model | Accuracy | Balanced accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
{model_lines}

The logistic coefficients are standardized; tree importance is Gini importance. Both are recorded in `suitability_prediction_results.csv` and fitted on the full diagnostic sample only for interpretation, not for the CV performance estimates.

## Leakage audit

- **Verified:** no explanatory feature uses validation/test ratings; all 28 features are computed from TRAIN rows only.
- **Verified:** no explanatory feature contains MF/LightGCN scores or predictions.
- **Verified:** no feature is constructed from ranking outcomes or evaluator metrics.
- Saved test outcomes are used only to define the post-hoc supervised winner label. This analyzes explainability/predictability; it does not select or tune a recommender.
- The simple classifier hyperparameters are fixed analytical defaults; no test-label-driven hyperparameter search was conducted.

## Multi-seed limitation

Multi-seed user-level analysis is **unavailable**. The robustness directory contains aggregate and sparsity-bucket results but {len(robust_files)} per-user prediction/metric files. Per the no-inference constraint, seed-level winner fractions and top-five user-feature associations were not reconstructed.

## Explicit answers

### A. Are there distinct user groups where MF and LightGCN are preferable? — {verdict_a}

Both MF and LightGCN win for nonzero subsets of eligible users. The group distribution and adjusted feature tests quantify whether these groups are distinguishable; this is heterogeneity evidence, not proof that a deployable gate will generalize.

### B. Is simple interaction sparsity alone enough to predict the winner? — {verdict_b}

Interaction-count-only balanced accuracy is {count_bacc:.3f}, versus {majority_bacc:.3f} for the majority baseline.

### C. Do richer TRAIN-only user features improve winner prediction? — {verdict_c}

All-feature logistic balanced accuracy is {all_bacc:.3f}, compared with {count_bacc:.3f} for count only. Performance is cross-validated within one seed's user set.

### D. Which features are most strongly associated with LightGCN ranking advantage? — SUGGESTIVE

The five associations listed above are the strongest in this dataset. BH-adjusted p-values and effect directions should be used together; correlated features and a single split prevent causal interpretation.

### E. Can we predict the ranking winner substantially above majority? — {verdict_e}

The best balanced-accuracy gain over majority is {signal:.3f}. Accuracy alone is not used for this verdict because winner classes may be imbalanced.

### F. Is there enough signal for a future user-specific objective/gate? — {verdict_f}

This verdict requires both meaningful out-of-fold improvement and usable absolute discrimination. A future gate is not justified as a publication conclusion without per-user multi-seed replication; ranking-oriented LightGCN also lacks a saved user-level artifact.

## Interpretation limits

This is a single-split, post-hoc association study. Users are the CV units but share an interaction graph, so folds are not fully independent in a graph-statistical sense. Test-derived labels are legitimate here only as diagnostic outcomes. Results must not be presented as prospective model-selection performance, causal effects, or evidence for the unavailable ranking-oriented comparison.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
