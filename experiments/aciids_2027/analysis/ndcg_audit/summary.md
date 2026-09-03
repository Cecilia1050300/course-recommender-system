# NDCG audit summary

## Verdict

**CURRENT NDCG IMPLEMENTATION: CORRECT. FINAL VERDICT: RETAINED WITH EXPLANATION.**

1. The current binary NDCG implementation is mathematically correct; two independent hand calculations match its formula.
2. All relevant TEST items are candidates: missing relevant items = 0. No TRAIN-observed item is a candidate.
3. The MF reduction is consistent with full-catalog difficulty: held-out-only binary NDCG@5 is 0.973873, versus 0.193768 full-catalog. This comparison is diagnostic, not the legacy graded metric.
4. Per-method drops are in `protocol_comparison.csv`; negative values correctly denote methods that improve when catalog negatives supply ranking separation.
5. Near-zero CF is not one universal fallback bug. User-based variants fall back to TRAIN item means on 25.96% (rating), 62.42% (content), and 18.45% (hybrid) of candidates, but none has a tie group reaching the declared 20% threshold. Item_Content has 0% fallback yet substantial ties for every user. The common cause is weak/degenerate full-catalog ordering; fallback and ties contribute differently by variant. Item_Hybrid is materially stronger.
6. MF NDCG@5=0.193768 is credible: it is above deterministic random (0.022051) and follows the independently verified candidate/relevance arithmetic. Its full score matrix was not retained, preventing a post-hoc score-level audit without forbidden retraining.
7. Random NDCG@5=0.022051; MostPopular NDCG@5=0.251592.
8. No evaluator/candidate bug is evidenced. The important artifact-retention weakness is not a metric bug: neural/MF full candidate matrices/checkpoints were not saved, so their score coverage and sample top-10s cannot be independently reconstructed.
9. Paper results should be **RETAINED WITH EXPLANATION**: explicitly distinguish legacy graded held-out NDCG from current binary full-catalog NDCG and disclose the CF fallback/tie pathology.

## Integrity totals

- Ranking-eligible users: 185
- Candidates/user: 121 / 138.064865 / 150 (min/mean/max)
- Relevant items/user: 1 / 1.189189 / 3
- Comparable-model candidate set: identical by construction (model-independent evaluator inputs)

## Provenance limitation

The request for every model's candidate-score coverage and five MF/LightGCN top-10 lists cannot be completed from frozen artifacts. Only 599 held-out scores and per-user aggregates were retained for MF/GCN/LightGCN. Entries are deliberately `UNAVAILABLE`, not guessed, and no model was retrained.
