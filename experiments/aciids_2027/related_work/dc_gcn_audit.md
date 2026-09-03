# Technical literature audit: DC-GCN

## Audit status and evidence standard

**Paper audited:** Furong Peng, Fujin Liao, Xuan Lu, Jianxing Zheng, and Ru Li, “Revisiting explicit recommendation with DC-GCN: Divide-and-Conquer Graph Convolution Network,” *Information Systems*, vol. 130, article 102513, 2025. DOI: [10.1016/j.is.2024.102513](https://doi.org/10.1016/j.is.2024.102513).

**Audit date:** 2026-09-02.

This is a conservative audit. The publisher landing page and indexed publisher section snippets were accessible, as was the SSRN record for the 13-page author manuscript. However, the actual SSRN PDF download returned an anti-bot HTML page rather than a PDF, and the ScienceDirect full-text endpoint returned HTTP 403. Consequently, facts visible in the publisher header, abstract, Introduction, Problem definition, and Methodology snippet are reported as verified. Experimental tables, figure pixels/captions beyond the indexed text, and the experimental-setup section were not accessible and are marked **NOT VERIFIED**. No values were inferred from standard dataset versions, related papers, or chart geometry.

Primary records used:

- [ScienceDirect publisher record](https://www.sciencedirect.com/science/article/abs/pii/S0306437924001716): bibliographic header, abstract, and indexed section snippets (Introduction, Problem definition, Methodology).
- [SSRN author-manuscript record](https://ssrn.com/abstract=4935785): title, authors, 13-page manuscript record, posting date, abstract, and keywords. The PDF itself was not retrievable during this audit.
- [DBLP record](https://dblp.org/rec/journals/is/PengLLZL25): independent bibliographic cross-check only.

“Paper-reported” below means text directly visible in one of those primary records. “Audit interpretation” is explicitly labelled. `NOT VERIFIED` means the requested detail could not be checked in the paper itself under the access conditions above.

## 1. Bibliographic information

| Field | Verified detail | Exact source |
|---|---|---|
| Full title | *Revisiting explicit recommendation with DC-GCN: Divide-and-Conquer Graph Convolution Network* | ScienceDirect bibliographic header; SSRN title |
| Authors | Furong Peng; Fujin Liao; Xuan Lu; Jianxing Zheng; Ru Li | ScienceDirect/SSRN bibliographic metadata |
| Venue | *Information Systems* | ScienceDirect bibliographic header |
| Volume / date / article | Volume 130, April 2025, article 102513 | ScienceDirect bibliographic header |
| DOI | 10.1016/j.is.2024.102513 | ScienceDirect bibliographic header |
| Publication year | 2025 (the DOI contains 2024, but that is not the publication year) | ScienceDirect bibliographic header; DBLP cross-check |
| Author manuscript | 13 pages, posted 24 August 2024 | SSRN record |

### Task definition

**Paper-reported:** The Problem definition represents observed explicit interactions by a sparse rating matrix \(\mathcal R\in\mathcal K^{m\times n}\), where \(\mathcal K\) is a finite discrete rating set (the paper gives \(\{1,2,3,4,5\}\) as an example). It states two possible recommendation outputs: filling missing entries with predicted ratings, or generating a ranked item list for each user. Source: **Problem definition** indexed on the ScienceDirect record.

**What the proposed experiments actually optimize:** DC-GCN uses point-wise MSE according to the final paragraph of the **Introduction**. The accessible text therefore verifies an **explicit rating-prediction training task**. Whether the main experimental protocol also evaluates Top-K ranked lists is **NOT VERIFIED** because the metrics/results section was inaccessible.

## 2. Dataset audit

The abstract states that DC-GCN was evaluated on **four public datasets**. Source: **Abstract**. The names and all dataset-level statistics are absent from the accessible primary-source text.

| Dataset | Users | Items | Ratings/interactions | Rating scale | Density/sparsity | Preprocessing/minimum filter | Split protocol/ratio | Cold-start presence | Separate cold-start experiment |
|---|---:|---:|---:|---|---|---|---|---|---|
| Dataset 1 — name **NOT VERIFIED** | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Yes, an analysis is claimed; exact setup **NOT VERIFIED** |
| Dataset 2 — name **NOT VERIFIED** | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Yes, an analysis is claimed; exact setup **NOT VERIFIED** |
| Dataset 3 — name **NOT VERIFIED** | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Yes, an analysis is claimed; exact setup **NOT VERIFIED** |
| Dataset 4 — name **NOT VERIFIED** | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Yes, an analysis is claimed; exact setup **NOT VERIFIED** |

No counts, ratios, densities, or derived values are supplied because the paper-reported inputs needed to compute them were unavailable. The abstract/Introduction verifies that the authors analyze “cold-start” and “popularity bias” scenarios, but does not expose their operational definitions.

## 3. Evaluation metrics

| Metric | Used by DC-GCN? | Definition / direction / scope | Exact paper source |
|---|---|---|---|
| MSE | **YES (training objective)** | Point-wise squared error; lower is better; rating-prediction oriented. Whether it is also a reported test metric is NOT VERIFIED. | Introduction, final method-summary paragraph |
| MAE | NOT VERIFIED | Definition, direction, and test scope NOT VERIFIED | Metrics section inaccessible |
| RMSE | NOT VERIFIED | Definition, direction, and test scope NOT VERIFIED | Metrics section inaccessible |
| NDCG | NOT VERIFIED | K and candidate scope NOT VERIFIED | Metrics section inaccessible |
| Recall | NOT VERIFIED | K and candidate scope NOT VERIFIED | Metrics section inaccessible |
| Precision | NOT VERIFIED | K and candidate scope NOT VERIFIED | Metrics section inaccessible |
| HR / HitRate | NOT VERIFIED | K and candidate scope NOT VERIFIED | Metrics section inaccessible |

The Problem definition mentions ranked-list generation as one possible recommendation task, but that statement is not evidence that any ranking metric was used. It is therefore unsafe to treat DC-GCN as a joint rating-and-ranking evaluation without the experimental section.

## 4. Baselines

### Main-experiment baseline list

**NOT VERIFIED.** The main comparison table and experimental-baseline subsection were inaccessible, so an exhaustive baseline list cannot be publication-grade.

The following methods are visible in accessible text, but their roles must not be overstated:

| Category | Method | Verified role | Objective/original setting/adaptation |
|---|---|---|---|
| A. Classical explicit-feedback | MF(MSE) | Comparator in Fig. 2 discussion | Matrix factorization with MSE; explicit rating prediction. Further optimizer/settings NOT VERIFIED. Source: Introduction discussion of Fig. 2. |
| B. Neural / MF | DMF | Mentioned in related work, **not verified as a main baseline** | Related-work snippet says it uses a neural network to approximate the rating matrix. Main-experiment loss/adaptation NOT VERIFIED. Source: “Collaborative filtering for explicit recommendation” section snippet. |
| B. Neural / MF | NMF | Mentioned in related work, **not verified as a main baseline** | Related-work snippet says it decomposes the rating matrix; experimental role/settings NOT VERIFIED. |
| C. GCN / graph recommendation | LightGCN(MSE) | Comparator in Fig. 2 discussion | LightGCN trained with MSE for explicit prediction. Detailed architecture/settings NOT VERIFIED. |
| D. Implicit-to-explicit adaptation | LightGCN(BPR+Explicit) | Comparator in Fig. 2 discussion | Described only as BPR “adjusted for ratings.” Exact pair construction and rating incorporation NOT VERIFIED. |
| D. Implicit baseline | LightGCN(BPR) | Requested target, but its presence as a distinct experiment is NOT VERIFIED | Objective/adaptation/settings/results NOT VERIFIED. |

Other methods mentioned in the Introduction (NGCF, simplified GCN models, ApeGNN) establish context; the accessible text does not prove that they appear in the main experiments. An exhaustive categorization into A–D is therefore **NOT VERIFIED**.

## 5. LightGCN variants

### Verified qualitative comparison

The **Introduction’s discussion of Fig. 2** states both of the following:

1. `LightGCN(BPR+Explicit)` underperforms `LightGCN(MSE)`.
2. `LightGCN(MSE)` underperforms the shallower `MF(MSE)`.

The authors interpret the first result as evidence that point-wise MSE is preferable to pair-wise BPR for fine-grained explicit prediction, and the second as a warning that a deep model may overfit very sparse data. These are **paper-reported interpretations**, not conclusions generated by this audit.

### Variant details

| Variant | Objective | Explicit-rating incorporation | Positive/negative definition; unobserved negatives | Embedding dim / layers | Optimizer / LR / regularization | Batch / epochs / stopping | Selection metric | Numerical results |
|---|---|---|---|---|---|---|---|---|
| MF(MSE) | MSE (verified by name and Fig. 2 discussion) | Predict observed explicit ratings | Not applicable/NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Fig. 2 qualitative ordering only; exact values NOT VERIFIED |
| LightGCN(MSE) | MSE | Explicit ratings are regression targets; exact prediction head NOT VERIFIED | Not applicable to MSE; treatment of unobserved entries NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Fig. 2 qualitative ordering only; exact values NOT VERIFIED |
| LightGCN(BPR+Explicit) | BPR, adjusted for ratings | Exact adjustment NOT VERIFIED | Positive/negative pair definition and whether unobserved items are negatives: NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Fig. 2 qualitative ordering only; exact values NOT VERIFIED |
| LightGCN(BPR) | Presence as a separate reported variant NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| DC-GCN | Point-wise MSE | Rating graph divided into one subgraph per rating type; rating-aware embeddings; subgraph representations fused by a three-layer MLP | No pair construction for the verified MSE objective | GCN depth/embedding size NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Main numerical results NOT VERIFIED |

Requested numerical layout (the four dataset names and reported metric names were not accessible):

| Dataset | Metric | MF(MSE) | LightGCN(MSE) | LightGCN(BPR+Explicit) | DC-GCN | Source |
|---|---|---:|---:|---:|---:|---|
| All four datasets | All reported metrics | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Fig. 2/table content inaccessible; no bar heights approximated |

## 6. Main DC-GCN results

The abstract reports state-of-the-art performance on four public datasets. No exact dataset/metric/result cell or strongest baseline was visible. Therefore:

| Dataset | Metric | DC-GCN | Best baseline | Absolute improvement | Relative improvement | Source |
|---|---|---:|---:|---:|---:|---|
| Each of four unnamed datasets | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Results table inaccessible |

No improvement was computed, because doing so would require two verified paper numbers.

## 7. Ablation study

| Component / question | Verified architecture claim | Numerical ablation evidence | What it can empirically support |
|---|---|---|---|
| Divide-and-conquer rating subgraphs | One subgraph is constructed for each rating type and processed by a GCN. | NOT VERIFIED | Architecture motivation is verified; empirical contribution is NOT VERIFIED. |
| Rating-aware embeddings | A rating-aware embedding is created for each subgraph to model rating-related relations. | NOT VERIFIED | Claimed mechanism verified; isolated benefit NOT VERIFIED. |
| Shared node embeddings | Embeddings are shared across GCNs to reduce parameters and mitigate overfitting/sparsity. | NOT VERIFIED | Claimed rationale verified; isolated benefit NOT VERIFIED. |
| Random mask convolution | Randomly selects columns of node features to update in GCN layers to alleviate homogeneous representations/over-smoothing. | NOT VERIFIED | Claimed mechanism verified; measured benefit NOT VERIFIED. |
| Three-layer MLP aggregation | Representations from rating subgraphs are aggregated by a three-layer MLP. | NOT VERIFIED | Architecture verified; ablation NOT VERIFIED. |
| Number of GCN layers | NOT VERIFIED | NOT VERIFIED | No depth conclusion can be extracted. |
| Over-smoothing | Paper explicitly motivates random column mask as mitigation. | NOT VERIFIED | Existence of the design goal is verified; layer-wise/quantitative evidence is not. |

Source for verified architecture claims: **Abstract** and **Introduction**; three-layer MLP also appears in the Introduction and Methodology snippet. The actual ablation table/figure was inaccessible.

## 8. Sparsity and cold-start analysis

| Concept | Does the accessible paper text say it is studied? | Definition/buckets/methods/metrics/results | Classification |
|---|---|---|---|
| A. Global matrix sparsity | Sparsity is central motivation; four datasets are called sparse in general terms. | Dataset density and controlled global-sparsity experiment NOT VERIFIED. | Motivation verified; experiment UNKNOWN. |
| B. User interaction-count sparsity | NOT VERIFIED | Buckets and results NOT VERIFIED. | UNKNOWN. |
| C. Item popularity sparsity / bias | Yes, “popularity bias scenarios” are claimed. | Popularity grouping, long-tail threshold, metrics and numbers NOT VERIFIED. | Analysis exists, operationalization UNKNOWN. |
| D. True unseen-user cold start | A “cold-start” analysis is claimed. | Whether users are unseen, merely low-degree, or progressively subsampled is NOT VERIFIED. | UNKNOWN; do not equate with true unseen-user cold start. |

The abstract’s only safely extractable conclusion is that DC-GCN exhibited “competitive performance” in cold-start and popularity-bias scenarios. Exact bucket definitions, compared methods, metrics, and numerical results are **NOT VERIFIED**. In particular, the paper’s word “cold-start” must not be cited as proof of inductive prediction for entirely unseen users or items.

## 9. Does DC-GCN study our rating-versus-ranking question?

Target question: *Does optimizing explicit rating prediction select the best Top-K recommendation model under sparse feedback?*

| Required analysis | YES / NO / PARTIAL | Exact evidence |
|---|---|---|
| Compares RMSE/MAE and NDCG/Recall over the same hyperparameter grid | **UNKNOWN** | Metrics and tuning sections inaccessible; no accessible text describes such a grid. |
| Studies correlation between rating and ranking metrics | **NO (in accessible evidence)** | Neither abstract, Introduction, Problem definition, nor Methodology snippet reports a correlation analysis. Full-paper verification remains blocked. |
| Demonstrates RMSE-optimal versus NDCG-optimal configurations | **NO (in accessible evidence)** | Fig. 2 compares loss/model variants qualitatively, not verified metric-optimal configurations. |
| Performs rating-ranking Pareto analysis | **NO (in accessible evidence)** | No Pareto analysis is described in accessible primary text. |
| Analyzes rating-ranking alignment by user interaction sparsity | **NO (in accessible evidence)** | Cold-start is claimed, but no cross-metric alignment analysis or interaction-count buckets are visible. |
| Uses multiple random seeds for this trade-off | **UNKNOWN** | Repetition/seeds section inaccessible. |

**Audit interpretation:** The paper is adjacent to our question because Fig. 2 contrasts BPR and MSE variants in explicit recommendation. It does **not**, on the verifiable record, establish whether an RMSE-optimal model is also Top-K optimal. Thus the overall answer is **PARTIAL conceptual overlap, no verified direct test**.

## 10. Novelty overlap with our current study

The “our study” claims below are supplied by the ACIIDS 2027 project brief; this audit did not re-run or re-check them.

| Our verified finding | DC-GCN relation | Evidence and boundary |
|---|---|---|
| 1. MF has lower RMSE than LightGCN across five seeds. | **PARTIALLY OVERLAPS** | Fig. 2 discussion says MF(MSE) outperforms LightGCN(MSE), but exact metric and multi-seed robustness are NOT VERIFIED. Source: Introduction/Fig. 2 discussion. |
| 2. LightGCN has higher NDCG@5 than MF across five seeds. | **NOT ADDRESSED** | No verified NDCG@5 comparison or multi-seed ranking result. |
| 3. Across 72 LightGCN configurations, RMSE–NDCG@5 Spearman is about 0.06. | **NOT ADDRESSED** | No verified metric-correlation analysis. |
| 4. RMSE-optimal and NDCG-optimal LightGCN configurations differ. | **NOT ADDRESSED** | No verified dual-objective selection analysis. |
| 5. In a controlled depth slice, layers 1→3 worsen RMSE but improve NDCG@5. | **PARTIALLY OVERLAPS** | DC-GCN targets over-smoothing and uses random mask convolution, but no verified rating-versus-ranking depth slice. |
| 6. LightGCN ranking advantage is strongest for users with 1–10 train interactions. | **UNKNOWN** | DC-GCN claims cold-start analysis, but cold-start definition, interaction-count buckets, and NDCG are NOT VERIFIED. |
| 7. Rule-based sparsity-aware layer weighting did not improve LightGCN. | **NOT ADDRESSED** | No verified user-specific layer aggregation or matching negative result. ApeGNN is mentioned only as related work. |

## 11. Method-space comparison

Verified DC-GCN design: rating-specific subgraphs + rating-aware embeddings + shared node embeddings + random column mask convolution + point-wise MSE. Sources: Abstract and Introduction.

| Possible direction for our study | Direct overlap | Indirect overlap | Likely novelty risk | DC-GCN evidence that must be cited |
|---|---|---|---|---|
| A. Observed graded ordinal ranking | Low/UNKNOWN | Medium: DC-GCN explicitly contrasts BPR+Explicit with MSE and argues for point-wise regression. | **Moderate** if framed merely as “use ratings in a graph loss”; lower if the ordinal pair construction and ranking evaluation are distinct and demonstrated. | Fig. 2 qualitative BPR+Explicit vs MSE finding; DC-GCN’s rating-specific graph treatment and MSE choice. Exact BPR construction is NOT VERIFIED and must not be asserted. |
| B. Rating-ranking Pareto optimization | No verified direct overlap | Medium: both concern objective choice in explicit recommendation. | **Low-to-moderate**, pending full-paper verification. | Cite the Fig. 2 loss comparison and acknowledge that DC-GCN chooses MSE; state that no verified Pareto/correlation analysis was found. |
| C. Uncertainty-aware graph/MF mixture | No verified direct overlap | Medium: DC-GCN motivates when shallow MF can beat deep LightGCN under sparsity. | **Low**, unless its inaccessible baselines/ablations contain a mixture. | Cite Introduction’s MF(MSE) > LightGCN(MSE) observation and sparsity/overfitting rationale. |
| D. Sparse-user-specific confidence modeling | No verified direct overlap | Medium-high: DC-GCN addresses sparsity and claims cold-start analysis. | **Moderate**; exact cold-start mechanism/results must be checked before a novelty claim. | Cite shared/rating-aware embeddings as sparsity mitigation and the claimed cold-start analysis, while keeping true unseen-user versus low-degree users separate. |
| E. Metric-aware model selection | No verified direct overlap | Medium: Fig. 2 varies the training objective. | **Low-to-moderate**, pending metrics/tuning verification. | Cite the MSE-versus-BPR discussion; do not claim DC-GCN selected by RMSE, NDCG, or a Pareto rule without the missing setup section. |

## Concise comparison

| Aspect | DC-GCN | Our current study | Overlap risk |
|---|---|---|---|
| Domain | General explicit-feedback recommendation on four public datasets (names NOT VERIFIED) | Sparse explicit-feedback course recommendation | Moderate |
| Primary method | Rating-specific GCN subgraphs, rating-aware/shared embeddings, random mask convolution, MLP fusion | MF, Vanilla GCN, LightGCN, sparsity-aware aggregation analyses | Moderate for graph explicit recommendation; low for exact architecture |
| Verified objective | Point-wise MSE | Rating MSE plus shared Top-K evaluation; later objective design audit | Moderate |
| Rating-versus-ranking alignment | No verified correlation, Pareto, or metric-optimal comparison | Five-seed cross-model result, 72-config correlation, metric-specific optima | Low on currently verified evidence |
| Depth | Over-smoothing explicitly targeted; numerical depth study NOT VERIFIED | Controlled 1/2/3-layer slice across rating and ranking metrics | Low-to-moderate |
| Sparsity | Global sparsity motivation; cold-start and popularity analyses claimed but definitions unavailable | Train-count buckets and multi-seed sparse-user comparisons | UNKNOWN until full text is obtained |
| Robustness | Seeds/repetitions NOT VERIFIED | Five predetermined seeds | Low on available evidence |

## What DC-GCN already solved

DC-GCN proposes a concrete explicit-feedback graph architecture that does not collapse rating-valued edges into one undifferentiated interaction type. It divides the rating graph by rating value, performs graph convolution within each rating-specific subgraph, shares node embeddings to control parameter growth, adds rating-aware embeddings to distinguish subgraphs, fuses subgraph representations through a three-layer MLP, and uses random column masking to address over-smoothing. It trains with point-wise MSE. The paper also explicitly surfaces the difficulty of applying deep graph models and pair-wise BPR to sparse explicit feedback. Sources: Abstract and Introduction.

The accessible paper text additionally reports qualitative Fig. 2 findings that LightGCN(BPR+Explicit) is worse than LightGCN(MSE), and LightGCN(MSE) is worse than MF(MSE). Exact datasets, metrics, and effect sizes remain NOT VERIFIED.

## What DC-GCN did NOT study

On the evidence that could be verified, DC-GCN did **not** report a rating-error/Top-K correlation analysis, RMSE-optimal versus NDCG-optimal configurations, a Pareto frontier, a controlled cross-metric propagation-depth slice, or a five-seed test of the MF-versus-LightGCN trade-off. Because the experimental full text was inaccessible, these are statements about the **verified record**, not categorical proof of absence from every page. Dataset-specific user-count sparsity buckets and true unseen-user cold start are also not established by the accessible material.

## What we must cite DC-GCN for

We should cite it for: (1) adapting graph convolution to explicit rating prediction via rating-specific subgraphs; (2) rating-aware embeddings combined with shared node parameters; (3) random column mask convolution as an anti-over-smoothing mechanism; (4) the use of point-wise MSE in this architecture; and (5) its reported qualitative observation that MSE-based LightGCN beats its rating-adjusted BPR variant while MF(MSE) beats LightGCN(MSE) in Fig. 2. Any citation of datasets, exact metrics, numerical gains, cold-start definitions, or hyperparameters must wait until the experimental tables/full text can be retrieved.

## Remaining defensible research space for our paper

The defensible space is the **measurement and model-selection problem across rating and ranking objectives**, especially under controlled user interaction sparsity: reproducible same-split MAE/RMSE and full-catalog Top-K evaluation; metric-correlation analysis over a fixed grid; RMSE-optimal versus NDCG-optimal configurations; controlled propagation-depth comparisons; multi-seed robustness; and sparsity-bucket localization of ranking gains. Graded ordinal ranking, rating-ranking Pareto optimization, uncertainty-aware MF/graph mixtures, sparse-user confidence modeling, and metric-aware selection remain potentially distinct, but novelty claims concerning DC-GCN’s inaccessible experimental details must be qualified until the full manuscript is available.

## Verification gaps requiring the actual PDF

For a complete publication-ready citation audit, the following must be rechecked directly against the PDF: all four dataset names and Table-level statistics; preprocessing and split rules; every reported metric and its formula; the exhaustive main-baseline list; all LightGCN variant implementation details; Fig. 2 numeric values; main result tables and improvements; ablation numbers; cold-start/popularity bucket definitions and results; hyperparameter ranges; model-selection procedure; and seed/repetition policy. This file deliberately contains no reconstructed or approximate values for those fields.
