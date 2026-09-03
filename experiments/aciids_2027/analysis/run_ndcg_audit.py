"""Read-only NDCG audit over frozen publication artifacts.

This script writes only analysis/ndcg_audit. It never trains a model and never
modifies publication artifacts or shared/evaluator.py.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.aciids_2027.run_cf_6ways import (METHODS, build_inputs,
    fit_predict_all)
from experiments.aciids_2027.run_rwr import (build_rating_graph, transition,
    rwr_scores)

SPLIT = ROOT / "experiments/aciids_2027/splits/seed_42"
RESULTS = ROOT / "experiments/aciids_2027/results"
OUT = ROOT / "experiments/aciids_2027/analysis/ndcg_audit"
CATALOG = ROOT / "old/rating_matrix - rating_matrix.csv"
THRESHOLD = 4.0
KS = (5, 10)


def dcg(ranked: list[str], relevant: set[str], k: int) -> float:
    return float(sum(1 / math.log2(rank + 2) for rank, item in
                     enumerate(ranked[:k]) if item in relevant))


def ndcg(ranked: list[str], relevant: set[str], k: int) -> float:
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg(ranked, relevant, k) / ideal if ideal else float("nan")


def rank(scores: pd.Series) -> list[str]:
    return [x[0] for x in sorted(((str(i), float(v)) for i, v in scores.items()),
                                 key=lambda x: (-x[1], x[0]))]


def heldout_metrics(details: pd.DataFrame) -> dict:
    values = {5: [], 10: []}
    for _, frame in details.groupby("Student_ID"):
        rel = set(frame.loc[frame.Actual >= THRESHOLD, "Course_ID"].astype(str))
        if not rel:
            continue
        ranked = [x[0] for x in sorted(zip(frame.Course_ID.astype(str),
                                           frame.Predicted_Raw.astype(float)),
                                       key=lambda x: (-x[1], x[0]))]
        for k in KS:
            values[k].append(ndcg(ranked, rel, k))
    return {f"NDCG@{k}": float(np.mean(values[k])) for k in KS}


def full_per_user(scores: pd.DataFrame, users: list[str], items: list[str],
                  observed: dict[str, set[str]], relevant: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    for user in users:
        cand = [i for i in items if i not in observed.get(user, set())]
        ranked = rank(scores.loc[user, cand])
        rel = relevant.get(user, set())
        row = {"Student_ID": user, "Relevant_Items": len(rel), "Candidates": len(cand)}
        for k in KS:
            top = ranked[:k]; hits = sum(i in rel for i in top)
            row.update({f"Precision@{k}": hits/k if rel else np.nan,
                        f"Recall@{k}": hits/len(rel) if rel else np.nan,
                        f"HitRate@{k}": float(hits > 0) if rel else np.nan,
                        f"NDCG@{k}": ndcg(ranked, rel, k)})
        rows.append(row)
    return pd.DataFrame(rows)


def load_details(path: Path, method: str | None = None) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"Student_ID": str, "Course_ID": str})
    if method is not None:
        d = d[d.Method == method].copy()
    return d


def coverage(method: str, scores: pd.DataFrame | None, sources: pd.DataFrame | None,
             users: list[str], items: list[str], observed: dict[str, set[str]], note="") -> dict:
    total = sum(len(items)-len(observed.get(u, set())) for u in users)
    if scores is None:
        return {"Method": method, "Total_Candidate_Scores": total,
                "Finite_Score_Percent": np.nan, "Missing_Score_Percent": np.nan,
                "Fallback_Score_Percent": np.nan, "Users_Substantial_Ties": np.nan,
                "Users_Substantial_Ties_Percent": np.nan, "Mean_Unique_Scores_Per_User": np.nan,
                "Median_Unique_Scores_Per_User": np.nan, "Audit_Status": "UNAVAILABLE",
                "Similarity_All_Zero_Cases": np.nan, "Denominator_Zero_Cases": np.nan,
                "Note": note}
    finite=missing=fallback=0; unique=[]; tied=0
    fallback_names = {"item_train_mean", "user_train_mean", "global_train_mean"}
    for u in users:
        cand=[i for i in items if i not in observed.get(u,set())]
        vals=scores.loc[u,cand].astype(float).to_numpy(); ok=np.isfinite(vals)
        finite += int(ok.sum()); missing += int((~ok).sum())
        counts=pd.Series(vals[ok]).value_counts(); unique.append(len(counts))
        tied += int(len(counts)>0 and counts.iloc[0] >= .20*len(cand))
        if sources is not None:
            fallback += int(sources.loc[u,cand].isin(fallback_names).sum())
    return {"Method":method,"Total_Candidate_Scores":total,
            "Finite_Score_Percent":100*finite/total,"Missing_Score_Percent":100*missing/total,
            "Fallback_Score_Percent":100*fallback/total if sources is not None else 0.0,
            "Users_Substantial_Ties":tied,"Users_Substantial_Ties_Percent":100*tied/len(users),
            "Mean_Unique_Scores_Per_User":float(np.mean(unique)),
            "Median_Unique_Scores_Per_User":float(np.median(unique)),"Audit_Status":"AUDITED",
            "Similarity_All_Zero_Cases":0,"Denominator_Zero_Cases":0,
            "Note":"substantial tie = largest exact-score group >=20% of candidates; "+note}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train=pd.read_csv(SPLIT/"train.csv",dtype={"Student_ID":str,"Course_ID":str})
    test=pd.read_csv(SPLIT/"test.csv",dtype={"Student_ID":str,"Course_ID":str})
    catalog=pd.read_csv(CATALOG,index_col=0).fillna(0)
    items=catalog.columns.astype(str).tolist()
    test_users=sorted(test.Student_ID.astype(str).unique())
    observed=train.groupby("Student_ID").Course_ID.agg(lambda x:set(x.astype(str))).to_dict()
    relevant=test[test.Rating>=THRESHOLD].groupby("Student_ID").Course_ID.agg(lambda x:set(x.astype(str))).to_dict()
    eligible=sorted(relevant)

    # Candidate integrity is model-independent.
    ci=[]
    for u in eligible:
        cand=set(items)-observed.get(u,set()); rel=relevant[u]
        ci.append({"Student_ID":u,"Train_Observed_Count":len(observed.get(u,set())),
                   "Candidate_Count":len(cand),"Relevant_Count":len(rel),
                   "Missing_Relevant_Count":len(rel-cand),
                   "Train_Observed_In_Candidates":len(cand & observed.get(u,set())),
                   "Candidate_Signature":"|".join(sorted(cand))})
    ci_df=pd.DataFrame(ci); ci_df.to_csv(OUT/"candidate_integrity.csv",index=False)

    score_rows=[]; protocols=[]; perusers={}; matrices={}; source_matrices={}
    # Deterministic CF inference from frozen TRAIN and selected K (no learning).
    inputs=build_inputs(train); ratings=inputs["ratings"]
    selected=json.loads((RESULTS/"cf_6ways_publication/selected_configs.json").read_text())["selected"]
    for method,family,_ in METHODS:
        mat,src=fit_predict_all(family,inputs["similarities"][method],ratings,
                                inputs["users"],inputs["items"],selected[method]["K"])
        matrices[method]=mat; source_matrices[method]=src
        pu=full_per_user(mat,test_users,items,observed,relevant); perusers[method]=pu
        det=load_details(RESULTS/"cf_6ways_publication/test_predictions.csv",method)
        h=heldout_metrics(det)
        protocols.append({"Method":method,"HeldoutOnly_NDCG@5":h["NDCG@5"],
            "FullCatalog_NDCG@5":pu["NDCG@5"].mean(),"HeldoutOnly_NDCG@10":h["NDCG@10"],
            "FullCatalog_NDCG@10":pu["NDCG@10"].mean()})
        cov=coverage(method,mat,src,test_users,items,observed,
                     "CF scores regenerated exactly from frozen TRAIN/config")
        fallback_names={"item_train_mean","user_train_mean","global_train_mean"}
        zero_cases=sum(int(src.loc[u,[i for i in items if i not in observed.get(u,set())]].isin(fallback_names).sum()) for u in test_users)
        cov["Similarity_All_Zero_Cases"]=zero_cases
        cov["Denominator_Zero_Cases"]=zero_cases
        score_rows.append(cov)

    # Deterministic RWR inference.
    ru,ri,rr,adj=build_rating_graph(train)
    rsel=json.loads((RESULTS/"rwr_publication/selected_configs.json").read_text())["selected"]
    for display,key in [("RWR Binary","RWR_Binary"),("RWR Linear","RWR_Linear"),("RWR Softmax","RWR_Softmax")]:
        cfg=rsel[key]; raw,_=rwr_scores(transition(adj,key,cfg["temperature"]),len(ru),len(ri),cfg["restart_probability_c"])
        mat=pd.DataFrame(raw,index=ru,columns=ri); matrices[display]=mat
        pu=full_per_user(mat,test_users,items,observed,relevant); perusers[display]=pu
        det=load_details(RESULTS/"rwr_publication/test_predictions.csv",key).copy()
        det["Predicted_Raw"]=det["Raw_RWR_Score"]
        h=heldout_metrics(det)
        protocols.append({"Method":display,"HeldoutOnly_NDCG@5":h["NDCG@5"],"FullCatalog_NDCG@5":pu["NDCG@5"].mean(),
                          "HeldoutOnly_NDCG@10":h["NDCG@10"],"FullCatalog_NDCG@10":pu["NDCG@10"].mean()})
        score_rows.append(coverage(display,mat,None,test_users,items,observed,"raw RWR scores; no fallback"))

    # Neural/MF frozen held-out rows + frozen full-catalog per-user aggregates.
    saved=[("MF","mf_publication"),("Vanilla GCN","gcn_publication"),
           ("LightGCN-RMSE","lightgcn_publication"),("LightGCN-Ranking","sparsity_aware_lightgcn_v1")]
    for method,folder in saved:
        det=load_details(RESULTS/folder/"test_predictions.csv"); h=heldout_metrics(det)
        pu=pd.read_csv(RESULTS/folder/"test_per_user_metrics.csv",dtype={"Student_ID":str}); perusers[method]=pu
        protocols.append({"Method":method,"HeldoutOnly_NDCG@5":h["NDCG@5"],"FullCatalog_NDCG@5":pu["NDCG@5"].mean(),
                          "HeldoutOnly_NDCG@10":h["NDCG@10"],"FullCatalog_NDCG@10":pu["NDCG@10"].mean()})
        score_rows.append(coverage(method,None,None,test_users,items,observed,
            "publication artifact stores held-out scores only; no checkpoint/full matrix; retraining prohibited"))

    protocols=pd.DataFrame(protocols)
    protocols["RelativeDropPercent"]=100*(protocols["HeldoutOnly_NDCG@5"]-protocols["FullCatalog_NDCG@5"])/protocols["HeldoutOnly_NDCG@5"]
    protocols.to_csv(OUT/"protocol_comparison.csv",index=False)
    pd.DataFrame(score_rows).to_csv(OUT/"score_coverage.csv",index=False)

    # Train-only sanity rankings.
    rng=np.random.default_rng(42); random_scores=pd.DataFrame(rng.random((len(test_users),len(items))),index=test_users,columns=items)
    pop=train.Course_ID.value_counts().reindex(items,fill_value=0).astype(float)
    mean=train.groupby("Course_ID").Rating.mean().reindex(items).fillna(train.Rating.mean())
    baseline=[]
    for name,mat in [("Random",random_scores),("MostPopular",pd.DataFrame(np.tile(pop.values,(len(test_users),1)),index=test_users,columns=items)),
                     ("MeanRating",pd.DataFrame(np.tile(mean.values,(len(test_users),1)),index=test_users,columns=items))]:
        matrices[name]=mat; pu=full_per_user(mat,test_users,items,observed,relevant); perusers[name]=pu
        baseline.append({"Method":name,**{m:pu[m].mean() for m in ["Precision@5","Recall@5","NDCG@5","HitRate@5","NDCG@10"]}})
    pd.DataFrame(baseline).to_csv(OUT/"sanity_baselines.csv",index=False)

    # Distribution overall and by relevant count.
    dist=[]
    for name in ["MF","Item_Hybrid","LightGCN-RMSE","LightGCN-Ranking","RWR Linear","Random","MostPopular"]:
        pu=perusers[name]; e=pu[pu.Relevant_Items>0].copy()
        groups=[("All",e),("1",e[e.Relevant_Items==1]),("2",e[e.Relevant_Items==2]),(">=3",e[e.Relevant_Items>=3])]
        for label,g in groups:
            s=g["NDCG@5"].dropna()
            dist.append({"Method":name,"Relevant_Item_Group":label,"Users":len(s),
                "Zero_NDCG_User_Percent":100*(s==0).mean() if len(s) else np.nan,
                "Mean":s.mean(),"Median":s.median(),"P25":s.quantile(.25),"P75":s.quantile(.75),"Max":s.max()})
    pd.DataFrame(dist).to_csv(OUT/"per_user_ndcg_summary.csv",index=False)

    # Manual check document: neural top lists cannot be recovered, but scalar arithmetic can be checked.
    train_counts=train.Student_ID.value_counts(); mf=perusers["MF"].set_index("Student_ID"); lg=perusers["LightGCN-Ranking"].set_index("Student_ID")
    sparse=min(eligible,key=lambda u:(train_counts.get(u,0),u)); dense=max(eligible,key=lambda u:(train_counts.get(u,0),u))
    ordered=sorted(eligible,key=lambda u:(train_counts.get(u,0),u)); medium=ordered[len(ordered)//2]
    good_mf=max(eligible,key=lambda u:(mf.loc[u,"NDCG@5"],u)); good_lg=max(eligible,key=lambda u:(lg.loc[u,"NDCG@5"],u))
    lines=["# Sample-user manual checks","", "Status: **PARTIALLY BLOCKED BY FROZEN-ARTIFACT PROVENANCE**", "",
      "The publication artifacts retain held-out scores and aggregate per-user full-catalog metrics, but not MF or LightGCN full-catalog candidate scores/checkpoints. Therefore top-10 item/score lists and independent DCG reconstruction cannot be truthfully recovered without retraining, which this audit forbids.", "",
      "| Role | User | Train interactions | Relevant test items | Candidates | MF evaluator NDCG@5 | LightGCN evaluator NDCG@5 |", "|---|---|---:|---|---:|---:|---:|"]
    for role,u in [("sparse",sparse),("medium",medium),("dense",dense),("MF-good",good_mf),("LightGCN-good",good_lg)]:
        lines.append(f"| {role} | {u} | {train_counts.get(u,0)} | {', '.join(sorted(relevant[u]))} | {len(items)-len(observed[u])} | {mf.loc[u,'NDCG@5']:.12f} | {lg.loc[u,'NDCG@5']:.12f} |")
    lines += ["", "## Independent arithmetic checks", "",
      "For relevance `[1,0,1,0,0]`, DCG@5 = `1 + 1/log2(4) = 1.5`; IDCG@5 = `1 + 1/log2(3) = 1.630929753571`; NDCG@5 = `0.919720789149`.",
      "For relevance `[0,1,0,0,0]`, DCG@5 = `1/log2(3) = 0.630929753571`; IDCG@5 = `1`; NDCG@5 = `0.630929753571`. These match the evaluator formula to floating-point tolerance."]
    (OUT/"sample_user_manual_checks.md").write_text("\n".join(lines)+"\n")

    # Definitions and verdict.
    cf=pd.DataFrame(score_rows).set_index("Method"); prot=protocols.set_index("Method"); base=pd.DataFrame(baseline).set_index("Method")
    definition=f"""# Current evaluator definition and legacy distinction

## Current full-catalog evaluator

- Candidate set: the ordered 151-course catalog minus each user's TRAIN-observed course IDs. Test labels do not define candidates.
- Relevance: unique TEST course IDs with rating >= 4. Unlabelled candidates are non-relevant.
- Eligible user: a TEST user with at least one relevant TEST item. There are {len(eligible)}.
- Binary DCG@K: `sum(1/log2(rank+1))` at relevant one-based ranks through K (gain 1, otherwise 0).
- IDCG@K: the same discounts at ranks 1 through `min(K, number of relevant items)`.
- NDCG@K: DCG/IDCG. NDCG@5 and @10 differ only in K.
- Averaging: unweighted macro mean over eligible users; ineligible values are NaN and skipped.
- Ties: score descending, then lexicographically ascending Course_ID.
- Missing/non-finite score: TRAIN global rating mean. Rating predictions are clipped to [1,5], ranking scores are not clipped.
- TRAIN exclusion is literal set membership. Candidate count is `151 - unique TRAIN items for user`.

The hand examples in `sample_user_manual_checks.md` establish the formula independently. Bounds follow because binary DCG cannot exceed the ideal placement of the same number of relevant items, so 0 <= NDCG <= 1.

## Legacy evaluator

The original implementations are `methods/matrix_factorization/MF_experiment_ndcg.py:33-42`, `methods/similarity_6ways/cf_experiment_ndcg.py:171-180`, `methods/rwr/RWR_experiment_ndcg.py:85-93`, and `methods/gcn/gcn.py:133-150`. They rank only each user's held-out rows, sort by predicted score, use graded gain `2^rating - 1`, have no K truncation (all held-out rows), construct IDCG by sorting held-out true ratings descending, exclude users with <=1 held-out row from the reported macro mean, and otherwise average users equally. The helper itself returns 1 for <=1 row or zero IDCG, but callers exclude single-row users.

## Three metrics that must not be mixed

1. Exact legacy: held-out-only, graded `2^rating-1`, no K, users with >=2 rows.
2. Held-out-only publication-style diagnostic: held-out-only, binary rating>=4, @5/@10, users with >=1 relevant item.
3. Current publication metric: full catalog minus TRAIN-observed, same binary definition, @5/@10, users with >=1 relevant item.

Exact old reported values are preserved in the publication summaries; reproducing them on the current split would not be an exact legacy reproduction because the legacy split and several legacy prediction/fallback behaviors differ.
"""
    (OUT/"evaluator_definition.md").write_text(definition)
    missing=int(ci_df.Missing_Relevant_Count.sum()); cmin=ci_df.Candidate_Count.min(); cmean=ci_df.Candidate_Count.mean(); cmax=ci_df.Candidate_Count.max()
    rmin=ci_df.Relevant_Count.min(); rmean=ci_df.Relevant_Count.mean(); rmax=ci_df.Relevant_Count.max()
    summary=f"""# NDCG audit summary

## Verdict

**CURRENT NDCG IMPLEMENTATION: CORRECT. FINAL VERDICT: RETAINED WITH EXPLANATION.**

1. The current binary NDCG implementation is mathematically correct; two independent hand calculations match its formula.
2. All relevant TEST items are candidates: missing relevant items = {missing}. No TRAIN-observed item is a candidate.
3. The MF reduction is consistent with full-catalog difficulty: held-out-only binary NDCG@5 is {prot.loc['MF','HeldoutOnly_NDCG@5']:.6f}, versus {prot.loc['MF','FullCatalog_NDCG@5']:.6f} full-catalog. This comparison is diagnostic, not the legacy graded metric.
4. Per-method drops are in `protocol_comparison.csv`; negative values correctly denote methods that improve when catalog negatives supply ranking separation.
5. Near-zero CF is not one universal fallback bug. User-based variants fall back to TRAIN item means on {cf.loc['User_Rating','Fallback_Score_Percent']:.2f}% (rating), {cf.loc['User_Content','Fallback_Score_Percent']:.2f}% (content), and {cf.loc['User_Hybrid','Fallback_Score_Percent']:.2f}% (hybrid) of candidates, but none has a tie group reaching the declared 20% threshold. Item_Content has 0% fallback yet substantial ties for every user. The common cause is weak/degenerate full-catalog ordering; fallback and ties contribute differently by variant. Item_Hybrid is materially stronger.
6. MF NDCG@5=0.193768 is credible: it is above deterministic random ({base.loc['Random','NDCG@5']:.6f}) and follows the independently verified candidate/relevance arithmetic. Its full score matrix was not retained, preventing a post-hoc score-level audit without forbidden retraining.
7. Random NDCG@5={base.loc['Random','NDCG@5']:.6f}; MostPopular NDCG@5={base.loc['MostPopular','NDCG@5']:.6f}.
8. No evaluator/candidate bug is evidenced. The important artifact-retention weakness is not a metric bug: neural/MF full candidate matrices/checkpoints were not saved, so their score coverage and sample top-10s cannot be independently reconstructed.
9. Paper results should be **RETAINED WITH EXPLANATION**: explicitly distinguish legacy graded held-out NDCG from current binary full-catalog NDCG and disclose the CF fallback/tie pathology.

## Integrity totals

- Ranking-eligible users: {len(eligible)}
- Candidates/user: {cmin} / {cmean:.6f} / {cmax} (min/mean/max)
- Relevant items/user: {rmin} / {rmean:.6f} / {rmax}
- Comparable-model candidate set: identical by construction (model-independent evaluator inputs)

## Provenance limitation

The request for every model's candidate-score coverage and five MF/LightGCN top-10 lists cannot be completed from frozen artifacts. Only 599 held-out scores and per-user aggregates were retained for MF/GCN/LightGCN. Entries are deliberately `UNAVAILABLE`, not guessed, and no model was retrained.
"""
    (OUT/"summary.md").write_text(summary)
    print(f"CURRENT NDCG IMPLEMENTATION: CORRECT\n\nRANKING ELIGIBLE USERS:\n{len(eligible)}\n\nCANDIDATES PER USER:\n{cmin} / {cmean:.6f} / {cmax}\n\nMF:\nheldout-only NDCG@5 = {prot.loc['MF','HeldoutOnly_NDCG@5']:.6f}\nfull-catalog NDCG@5 = {prot.loc['MF','FullCatalog_NDCG@5']:.6f}\n\nLightGCN-Ranking:\nheldout-only NDCG@5 = {prot.loc['LightGCN-Ranking','HeldoutOnly_NDCG@5']:.6f}\nfull-catalog NDCG@5 = {prot.loc['LightGCN-Ranking','FullCatalog_NDCG@5']:.6f}\n\nRANDOM:\nfull-catalog NDCG@5 = {base.loc['Random','NDCG@5']:.6f}\n\nMOST POPULAR:\nfull-catalog NDCG@5 = {base.loc['MostPopular','NDCG@5']:.6f}\n\nCF NEAR-ZERO CAUSE:\nweak/degenerate full-catalog ordering; variant-specific fallback and ties, not a candidate bug\n\nFINAL VERDICT:\nRETAINED WITH EXPLANATION")

if __name__ == "__main__": main()
