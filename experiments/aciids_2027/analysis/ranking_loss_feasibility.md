# Train-only ranking-loss feasibility audit

Scope: immutable `seed_42/train.csv` only. No validation, test, or model-result artifact was read. The train split contains 4,845 interactions from 460 users and 151 courses; the train-observed course IDs define the catalog for unobserved-course counts.

## 1. Rating distribution

| Rating | Interactions | Percentage |
|---:|---:|---:|
| 1 | 988 | 20.39% |
| 2 | 820 | 16.92% |
| 3 | 1,207 | 24.91% |
| 4 | 1,227 | 25.33% |
| 5 | 603 | 12.45% |
| **Total** | **4,845** | **100.00%** |

- Rating ≥4: 1,830 interactions (37.77%).
- Rating <4: 3,015 interactions (62.23%).

## 2. Per-user availability

| Condition | Users | Percentage of 460 |
|---|---:|---:|
| At least one rating ≥4 | 313 | 68.04% |
| At least one rating <4 | 404 | 87.83% |
| Both ≥4 and <4 | 257 | 55.87% |
| Only positive/high-rated observations | 56 | 12.17% |
| No rating ≥4 (only <4) | 147 | 31.96% |

## 3. Candidate BPR pair definitions

### A. Explicit preference pairs

Definition: an observed rating ≥4 is positive and an observed rating <4 is negative.

- Eligible users: **257** (55.87%).
- Possible directed positive–negative pairs: **13,907**, computed as the sum over users of `number_high × number_low`.

Semantic strengths and risks:

- Both sides are observed, so the construction does not equate missing exposure with dislike.
- The threshold matches the evaluator's relevance definition.
- It excludes 203 users who lack one side of the threshold.
- It discards ordering within the two groups: rating 5 is treated like 4, and rating 1 like 3.
- A low course grade may reflect difficulty, prerequisites, timing, or assessment performance rather than negative preference.
- Users with many observations generate disproportionately many pairs unless sampling or user balancing is applied.

### B. Positive versus unobserved

Definition: an observed rating ≥4 is positive and an unobserved course is sampled as negative.

- Eligible users: **313** (68.04%).
- High-rated positive observations: **1,830**.
- Possible positive–unobserved pairs: **242,135**, computed against the 151-course train-derived catalog before sampling.

Semantic strengths and risks:

- It covers every user with a high-rated observation and supplies many candidate pairs.
- Unobserved does not mean disliked: a course may be unavailable, unknown, not yet taken, unsuitable for the student's program, or merely missing because of exposure.
- False negatives and course-popularity/exposure bias can dominate the learning signal.
- The sampled objective becomes sensitive to the negative-sampling distribution and random sampling rate.
- It ignores the 3,015 observed sub-threshold ratings as explicit comparative evidence.

## 4. Graded observed preference pairs

Definition: for two courses observed by the same user, form the directed pair `(i, j)` whenever `rating_i > rating_j`. Ties are omitted.

- Eligible users with at least two distinct observed rating levels: **314** (68.26%).
- Possible rating-ordered pairs: **28,076**.

| Rating gap | Ordered pairs | Percentage |
|---:|---:|---:|
| 1 | 16,528 | 58.87% |
| 2 | 8,280 | 29.49% |
| 3 | 2,779 | 9.90% |
| 4 | 489 | 1.74% |

Semantic strengths and risks:

- It uses observed evidence only, covers slightly more users than positive-vs-unobserved, and preserves all strict rating orderings.
- It is less dependent on each user's absolute use of the rating scale than a fixed threshold.
- Pair counts grow quadratically with user activity, so user-balanced sampling or weighting is important.
- Pairs are statistically dependent because interactions recur across many pairs.
- Small one-point differences may be noisy; gap-aware weights or sampling can distinguish weak from strong preferences.
- Grades can encode achievement and course difficulty rather than pure preference, so the pairwise interpretation must remain qualified.

## 5. Recommendation

The most defensible ranking construction for this explicit-rating educational dataset is **graded pairwise comparison between observed courses**. It avoids assuming that non-exposure is negative, retains more of the ordinal rating information than the ≥4 threshold, and supports 314 users with 28,076 possible pairs.

For a future implementation, sample or weight pairs per user so highly active users do not dominate, and consider rating-gap-aware weighting while preserving a separately reported observed-rating MSE objective. If a strictly binary BPR baseline is required, definition A is more defensible than definition B because both positive and negative sides are observed, but its reduced user coverage and threshold information loss must be reported.

This audit establishes feasibility only; it does not evaluate downstream validation or test performance.
