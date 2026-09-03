# Validation artifact consistency audit

Status: **PASS**

| Check | Result |
|---|---|
| Identical validation rating keys | True |
| Identical ranking-eligible users | True |
| Identical candidate item sets per user | True |
| Identical relevance labels | True |
| Train-observed items excluded | True |
| Identical candidate counts per user | True |
| Identical number of validation users | True (458) |

Candidate scores contain only full-catalog items not observed by that user in TRAIN. `train_observed` is therefore always false. Relevance is `validation rating >= 4`, exactly matching `EvaluationPolicy`. Ranking order remains shared-evaluator order: descending raw score, then ascending course ID for ties.
