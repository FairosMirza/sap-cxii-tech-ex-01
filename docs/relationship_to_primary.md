# Relationship to the primary submission

**Submission 1 is the final deliverable for the original assignment. Submission 2 is an optional experimental
companion, not its replacement or a claim of universal improvement.** Start with the primary solution, then use
this repository to discuss additional evaluation, alternative designs and the v2 requirements.

## Reviewed versions and scope

| Repository | Purpose | Reviewed commit |
|---|---|---|
| [Submission 1: primary](https://github.com/FairosMirza/sap-cxii-tech-ex-01_base) | Product-to-product similarity, including served text and images, configurable component weights, ANN, API and deployment | [85fed9e6af4205457211bae2d34256e0ba060d28](https://github.com/FairosMirza/sap-cxii-tech-ex-01_base/commit/85fed9e6af4205457211bae2d34256e0ba060d28) |
| [Submission 2: companion](https://github.com/FairosMirza/sap-cxii-tech-ex-01) | Alternative similarity pipeline, pooled human evaluation, free-text search and isolated experiments | [3683952d5d46ec26462d3d361f055563b376e8d7](https://github.com/FairosMirza/sap-cxii-tech-ex-01/commit/3683952d5d46ec26462d3d361f055563b376e8d7) |

Reviewed on 27 October 2026 against source, documentation and committed reports; Submission 2's SHA is the
baseline before this documentation-only update. No experiments were rerun. Its **served experimental
configuration** is C_bge with text 0.5, attributes 0.5 and images 0, plus BGE semantic search
([../configs/default.yaml](../configs/default.yaml)); the evaluation freeze at `77372d7` remains recorded in
[../eval/freeze.json](../eval/freeze.json). Neither is the final primary submission.

## Progression in reasoning

The primary establishes a working end-to-end solution: cleaned attributes, explainable component contributions,
explicit missing-data rules, MiniLM and ResNet features, and a fused HNSW index followed by exact re-ranking. Its
[report][primary-report] supplies functional, deployment and proxy evidence; its [decision record][primary-decisions]
already acknowledges correlated prices, ambiguous colours, imputation assumptions and the absence of relevance labels.

**Retrospective interpretation:** this provides a useful starting point for asking how directly judged substitute
relevance changes the choices, how to support free-text search, and what different evidence and retrieval policies
cost. Submission 2 investigates those questions; it does not demonstrate that the primary's choices were defects.
This is a conceptual presentation order, not a reconstructed development chronology: companion decisions are dated
21-23 October and the primary records verification on 23-27 October. Those records overlap; they do not establish
a sequential implementation history.

Keep four kinds of statement separate: **recorded rationale** in the dated decision logs; **measured findings** in
reports; **retrospective interpretation** connecting the implementations here; and **hypotheses** still needing a
controlled comparison. In particular, a plausible benefit from stricter parsing is not a measured accuracy gain.

## Compact comparison

Coverage below means fields available *after each repository's processing*, not conflicting facts about the raw data.
The final column interprets trade-offs retrospectively, not as newly discovered historical motives; unmeasured
benefits remain hypotheses.
Primary details: [parsing][primary-data], [features][primary-features], [scoring][primary-engine] and
[ANN][primary-ann]. Companion details: [../app/features.py](../app/features.py),
[../app/scoring.py](../app/scoring.py), [../app/retrieval.py](../app/retrieval.py) and
[../reports/data_profile.json](../reports/data_profile.json).

| Area | Primary implementation | Experimental companion | Reasoning, accepted cost and evidence limit |
|---|---|---|---|
| Text | MiniLM over names; 64-token limit; centred, normalised vectors | Title plus additional keyword tokens; 128-token limit; uncentred vectors with calibrated cosine; BGE served | Adds search-oriented representation and shared endpoint infrastructure; extra model/runtime cost. Model evidence is internal to Submission 2. |
| Price | Sale price weight 0.08; derived list price 0.02, falling back to sale price without a valid discount | Sale price scored; derived regular price retained only for valid discounts and never scored | Removes correlated price evidence but also list-price/discount-position information and a separately scored assignment attribute. No isolated ablation. |
| Colour | URL, name and technical colour combined; available-colour fallback; 79.3% known | Title wins, keywords only as fallback; available colours separate; 43.0% item colour known | Seeks more specific item evidence at a substantial coverage cost. Extraction precision was not compared. |
| Missing data and scaling | Query-side renormalisation; candidate mean-vector imputation except unknown brand/category score 0; centred cosine and percentile-scaled numeric angles | Pairwise availability masks; calibrated text, log price/weight kernels, Jaccard colour; evidence floor | Avoids treating missing candidates as typical but changes the denominator and penalises thin evidence. No controlled comparison of these policies. |
| ANN candidates | Fused 798-dimensional representation including images/attributes; normally 100 candidates for top-10 | 384-dimensional text index; 800 for similar products, 200 for search; selective filters use exact scoring | One reusable text index, but attribute/image winners can be missed; wider pools cost more scoring. Coverage is against different exact scorers. |
| Images | Served at weight 0.20; ResNet-50, PCA to 256 dimensions | Weight 0 in serving; full 2,048-dimensional ResNet-50 block tested separately | Keeps this serving configuration frozen despite promising development gains. Its promotion requirements do not judge the primary's image implementation. |
| Rating and audience | Rating similarity weight 0.03 plus tie-breaks; shopper segment weight 0.05 | Rating only breaks ties; audience supports semantic filtering, not a separate similarity component | De-emphasises noisy quality metadata; loses explicit rating/segment similarity. No isolated relevance gain established. |
| Category | Last category in the parent/child map; unthresholded majority browse-node filling; 99.6% known | Child sales-rank categories; browse-node filling only with at least 5 observations and 80% purity; 95.8% known | Seeks confidence over coverage, but errors remain. The primary's reported 95.8% browse-node agreement is not the companion's coverage statistic. |
| Brand | Normalises and fills missing brands from name prefixes matching brands seen at least twice; 76.5% known | Normalises supplied brands, no name inference; 71.8% known | Avoids inferred-brand mistakes but leaves more unknown, affecting matching, filters and family identification. No comparative accuracy audit. |
| Duplicates | Excludes the query's ASIN; one result per ASIN; other size/colour variants remain eligible | Also excludes the query's brand-plus-normalised-title family; one per other family; unknown-brand titles are not grouped | Seeks less repetition but may remove useful variants or group distinct products. The measured title-cap test does not isolate this family policy. |
| Ties | Scores rounded to 6 decimals, then rating, reviews, lower price, ID | Rounded to 9 decimals, then rating, reviews, ID | Both deterministic for fixed scores; different near-tie sensitivity and price preference. Neither policy has a demonstrated relevance advantage. |

## Price and colour: information versus confidence

**Recorded rationale:** the primary explicitly says its small list-price weight limits, but does not eliminate,
double-counting; list price equals sale price for 46% of priced products. D03 excludes that extra signal to avoid
correlated evidence. Only 14,624 companion records have derived regular-price metadata, versus the primary's
90.4% list-price coverage with its fallback. Exclusion is a design simplification, not a measured improvement.
It also means the companion does not independently score every attribute named in the [original assignment][primary-assignment]:
list price is metadata, rating is a tie-breaker, and images are experimental here.

The primary merges conflicting extracted colours and gives its available-colour fallback the same colour weight,
without a confidence discount. The companion retains title colours when keywords disagree, flags 176 conflicts,
and neither merges the keyword alternatives nor downweights the retained title colour. **The conflict flag is
diagnostic: neither similarity scoring nor semantic filtering consults it.** Colour dictionaries also differ
(for example, teal maps to green in the primary and blue here). Lower coverage alone does not establish higher precision.

In [../app/semantic.py](../app/semantic.py), an available-colour match can preserve eligibility even when item colour
differs and receives a 0.03 bonus, versus 0.06 for item colour. Unknown item colours remain eligible. The available
list never enters product-to-product colour similarity. Thus "available in" search and item-colour similarity use
different evidence intentionally; neither extraction policy makes the raw variant list an unambiguous item label.

## Scoring and retrieval: two interpretable designs

The primary already exposes per-component contributions and drops unavailable query components. Missing candidate
image, colour, segment and numeric vectors are mean-imputed; missing brand/category scores zero. **Source/documentation
qualification:** its report includes text in the imputation description, but `build_store` marks all text rows
available and does not mean-impute them. Centred text/images and percentile-scaled numeric angles can yield negative
similarities and support weighted inner-product retrieval.
Random 64-dimensional brand/category codes introduce candidate-stage noise; exact re-ranking restores equality
scores for retrieved candidates. Concatenation does not inherently remove interpretability or require naive zero-filling.

**Recorded companion rationale (D05):** calibrate typical random text pairs to 0 and typical best matches to 0.9,
use bounded log kernels for price/weight, and average only evidence both products possess. Its denominator is
`max(shared weight, 0.5 * total configured weight)`. The floor still includes query-missing components: this is
not simply query renormalisation, and its sparse-evidence penalty is deliberate, not proof of unbiased ranking.
Pair-dependent denominators complicate using one fixed fused vector for retrieval. Text-only candidates simplify
the index but can miss strong attribute matches; [D08](decisions.md#d08-default-search-mode-rule) measures that loss.
The text-weight sweep supports retaining 0.5 within this design, not its superiority over primary imputation,
centred cosine or fused retrieval. Calibration and evidence-floor benefits remain unisolated by the reported ablations.

## What the model results establish

**C_minilm is not Submission 1.** It uses the companion's title-plus-keyword text, 128-token encoding, calibrated
scoring, five-attribute block, pairwise missing-data policy and family exclusions, with **no images**, and is
evaluated on the companion's human labels. The primary uses name-only 64-token MiniLM, centred features, ten
components including images, different weights/imputation/deduplication and proxy diagnostics. Sharing an encoder
name does not make these the same system.

**Measured:** on development, C_bge and C_minilm are not separated for similar products (0.760 versus 0.749
nDCG@10). The original per-task selection rule chose C_minilm. The recorded D07 amendment, made after seeing
development results, chose BGE for both endpoints because its additional semantic-search endpoint favoured BGE
(+0.183 nDCG@10 over MiniLM) and sharing a model avoided a second vector set/index. Sharing also aligns with v2's
same-model requirement for semantic search. The primary has no free-text endpoint or online text encoder.
Both repositories use ONNX; D17's PyTorch comparison is
not a comparison with the primary's serving runtime.

On the frozen holdout, C_bge leads C_minilm by 0.077 nDCG@10 (95% interval 0.006 to 0.170; 9 scored product
queries), but the search lead does not replicate: BGE 0.758, MiniLM 0.778, with no pair of search models separated.
This supports a limited within-companion product-ranking result, **not Submission 2 outperforming Submission 1**.
Evidence: [../reports/dev_evaluation.json](../reports/dev_evaluation.json),
[../reports/holdout_evaluation.json](../reports/holdout_evaluation.json) and
[../reports/evaluation_checks.json](../reports/evaluation_checks.json).

## Why images can be served in one and experimental in the other

The primary's [image pipeline][primary-images] pads 224-pixel renditions, tries up to three image URLs, and uses
ONNX ResNet features reduced by PCA and centred/normalised for scoring. It reports 29,885 usable images and a
1.6-percentage-point co-purchase proxy gain from images, not a human relevance gain. The companion uses the first
eligible image URL, downloads to at most 320 pixels, and applies torchvision's ImageNet V2 preprocessing and full
normalised ResNet features with calibration ([../experiments/image_features.py](../experiments/image_features.py)).
It reports 29,861 usable images. Different preprocessing, downloads, fusion and candidates prevent transferring
either coverage or gains between implementations.

**Measured here:** weight 0.1 is the smallest tested image weight meeting D14's development rule: +0.048 nDCG@10
(0.019 to 0.082) over C_bge on 20 products, using 56 supplemental blind judgements. Higher weights score higher
but are not the rule's selection. The expanded judgement pool changes C_bge's baseline from 0.760 to 0.740;
only comparisons within the same pool are valid ([../reports/experiments/images.json](../reports/experiments/images.json)).

**Accepted cost and unresolved promotion:** no image gain is delivered by this companion's serving path. Removing
the query image reproduces C_bge's top-10 for 990/995 sampled products at weight 0.1, not all: the evidence floor
and float32 weighting explain the differences. With images, 800 text candidates cover only 0.930 of the exact
multimodal top-10 on development; 1,600 meet the 0.95 bar at higher latency. Promotion needs a verified missing-image
policy, adequate retrieval, integrated/validated image artifacts and a fresh holdout. Those are gates for this
configuration, not reasons to reject the primary's different image path. See
[../reports/experiments/images_fallback.json](../reports/experiments/images_fallback.json),
[../reports/experiments/images_ann_coverage.json](../reports/experiments/images_ann_coverage.json) and
[D14](decisions.md#d14-images). Its protocol, thresholds and results remain unchanged.

Semantic IDs likewise remain a reported alternative, not a serving improvement: smaller structures and faster
builds failed the pre-registered coverage rule against this companion's HNSW, including at comparable latency
([../reports/experiments/semantic_ids.json](../reports/experiments/semantic_ids.json); [D15](decisions.md#d15-semantic-ids)).

## Evidence boundaries and a fair future comparison

Rejecting co-purchase as substitute-relevance ground truth is compatible with the primary's explicitly labelled
weak proxy. Its name matching and mixture of complements/substitutes limit interpretation; the companion's pooled
human labels address relevance more directly but are small, pool-dependent and from one labeller. Neither
co-purchase percentages nor category agreement are interchangeable with nDCG or precision. The early mpnet
screening also measured category agreement and cost, not judged relevance.

ANN coverage means fidelity to each implementation's own exact scorer, not relevance. The reported primary
0.33/1.42 ms and companion 1.5/6.3 ms ANN/exact medians are not a controlled comparison: scorers, candidate counts,
query samples and measurement setups differ. The primary benchmark disables its result cache; HTTP timings and
the companion's online query encoding have different boundaries. Similar hardware labels do not remove these differences.

Before claiming superiority, a future comparison would need:
1. Pinned complete systems and the same catalogue snapshot, product-query sample, substitute rubric, result count
   and explicit policy for variants. Compare native end-to-end systems first; use separate controlled ablations to
   attribute effects to parsing, scoring, images or retrieval. Semantic search is an additional capability, not a
   measured win against a primary endpoint that does not exist.
2. Blind pooled judgements including both systems' results, preferably multiple assessors, missing-data/rare-category
   slices, paired uncertainty estimates and a **fresh** held-out query set after freezing choices. The existing
   holdout is spent and must not be reused for selection.
3. Matched hardware, threads, cache state, warm-up, concurrency and timing boundaries; report relevance separately
   from ANN fidelity, latency, memory and build cost. None of this comparison was run for this documentation review.

[primary-report]: https://github.com/FairosMirza/sap-cxii-tech-ex-01_base/blob/85fed9e6af4205457211bae2d34256e0ba060d28/docs/REPORT.md
[primary-decisions]: https://github.com/FairosMirza/sap-cxii-tech-ex-01_base/blob/85fed9e6af4205457211bae2d34256e0ba060d28/docs/DECISION.md
[primary-assignment]: https://github.com/FairosMirza/sap-cxii-tech-ex-01_base/blob/85fed9e6af4205457211bae2d34256e0ba060d28/ASSIGNMENT.md
[primary-data]: https://github.com/FairosMirza/sap-cxii-tech-ex-01_base/blob/85fed9e6af4205457211bae2d34256e0ba060d28/similarity/data.py
[primary-features]: https://github.com/FairosMirza/sap-cxii-tech-ex-01_base/blob/85fed9e6af4205457211bae2d34256e0ba060d28/similarity/features.py
[primary-engine]: https://github.com/FairosMirza/sap-cxii-tech-ex-01_base/blob/85fed9e6af4205457211bae2d34256e0ba060d28/similarity/engine.py
[primary-ann]: https://github.com/FairosMirza/sap-cxii-tech-ex-01_base/blob/85fed9e6af4205457211bae2d34256e0ba060d28/similarity/ann.py
[primary-images]: https://github.com/FairosMirza/sap-cxii-tech-ex-01_base/blob/85fed9e6af4205457211bae2d34256e0ba060d28/similarity/images.py
