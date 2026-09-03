# RWR publication evaluation audit

Status: **PASS**

| Check | Verification |
|---|---|
| Immutable split | Seed-42 TRAIN 4845, validation 600, test 599 rows for every variant. |
| Test rating keys | One shared test manifest; 599 identical keys. |
| Full catalog | 151 courses for every variant. |
| Candidate sets | Model-independent shared-evaluator construction. Validation signature `12153af0b9a82af12cb51c7a90fc491f93d0ec5872f4fbf2f74c73f301e1bbd1`; test signature `12153af0b9a82af12cb51c7a90fc491f93d0ec5872f4fbf2f74c73f301e1bbd1`. |
| Candidate counts | Validation 64317 (121–150/user); test 64317 (121–150/user). |
| TRAIN-item exclusion | Shared evaluator excludes items observed by that user in TRAIN. |
| Relevance | Binary `rating >= 4.0` from the supplied evaluation manifest. |
| Ranking eligibility | Shared evaluator includes only users with at least one relevant evaluation item. |
| Ranking tie-breaking | Raw RWR score descending, then Course ID ascending in shared evaluator. |
| Rating clipping | Legacy mapping is already [1,5]; shared evaluator still applies its standard [1,5] clip. |
| Leakage | Graph, transitions, and rating mapping use TRAIN only; no validation/test labels enter them. |
| Selection/test order | 64 validation configurations; per-variant selections persisted before test load; one test evaluation per variant. |

The candidate semantics are identical to publication MF, Vanilla GCN, LightGCN, and CF because all call the same evaluator with the same catalog and split manifests.
