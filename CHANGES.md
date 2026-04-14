# CHANGES.md

All changes from the confidence-aware metrics, feature ranking, and joint (k, alpha) optimization work.

---

## Part 0: Bug Fixes

### Bug 0A — `scripts/run_subgroup_steering.py`
- **Line 165 (deleted):** Removed dead variable `all_optimals: list[dict] = {}` — annotated as list but initialized as dict, never used. `all_manifests` is the actual accumulator.

### Bug 0B — `scripts/run_sae_steering.py` (E2b block)
- **Denominator fix:** Added `n_evaluated` counter that only increments for non-skipped items. Replaced `n = len(items)` with `n = max(n_evaluated, 1)` as the denominator for accuracy and flip rate calculations. Previously, skipped items (via `continue` on lines ~621, 625, 630, 638) were counted in the denominator, inflating accuracy and deflating flip rates.

### Bug 0C — `scripts/run_sae_steering.py` (E2b block)
- **Baseline caching:** Pre-compute `baseline_cache` for ALL items (tagged + untagged) before the condition loop. Each item now gets exactly one `evaluate_baseline_mcq()` call instead of up to 4 (one per condition it appears in). Lookup by `item_idx`.

### Bug 0D — `scripts/run_sae_steering.py` (E2b block)
- **Condition split:** Replaced 6 conditions with 8, splitting `mismatched_suppress/amplify` into `mismatched_within_suppress/amplify` (same-category different-subgroup) and `mismatched_cross_suppress/amplify` (different-category subgroup). Within-category tests fragmentation backfire; cross-category tests specificity. Selection logic uses `k[0] == cat` / `k[0] != cat` to filter.
- **Condition ordering fix:** Reordered if/elif chain so `mismatched_within` and `mismatched_cross` are checked BEFORE `matched`, since `"matched" in "mismatched_..."` is True (substring match). Without this fix, all mismatched conditions would fall into the matched branch.

### Bug 0E — `src/sae_localization/steering.py` and `src/sae_localization/subgroup_steering.py`
- **Docstring notes:** Added magnitude convention documentation to `get_composite_steering()` and `build_subgroup_steering_vector()`. Documents that the two functions produce different perturbation magnitudes for the same alpha value (factor of k difference) due to sum-of-scaled vs mean-then-scale approaches.

---

## Part 1: Confidence-Aware Bias Metrics

### New file: `src/metrics/__init__.py`
- Package init for the metrics module.

### New file: `src/metrics/bias_metrics.py`
- `compute_margin(logits, model_choice)` — logit gap between chosen answer and runner-up.
- `compute_rcr(results, tau)` — Robust Correction Rate: fraction corrected among items with baseline margin >= tau. Computed at tau in {0.5, 1.0, 2.0}.
- `compute_mwcs(results, tau)` — Margin-Weighted Correction Score: sigmoid-weighted correction rate with temperature tau.
- `compute_logit_shift(results)` — Mean/std/median shift of stereotyped-option logit under steering, with per-margin-bin breakdown.
- `compute_all_metrics(results)` — Unified entry point returning all three metrics plus raw correction/corruption rates.
- `MARGIN_BINS` — Standard binning: near_indifferent [0, 1), moderate [1, 2.5), confident [2.5, inf).

---

## Part 2: Feature Ranking Script

### Modified file: `scripts/rank_subgroup_features.py`
- **Layer auto-detection:** `--layers` is now optional. When omitted, scans `analysis_dir/features/` for `per_subcategory_layer_*.parquet` files and extracts layer numbers automatically.
- **Injection layer computation:** Computes optimal injection layer per subgroup (mode of layers among top-10 pro-bias features). Saves to `injection_layers.json`.
- **Anti-bias overlap:** Now computes overlap matrices for both `pro_bias` and `anti_bias` directions. Saves unified `feature_overlap.json` with both.
- **Summary logging:** Prints per-category summary with subgroup counts and total features.

---

## Part 3: Joint (k, alpha) Optimization

### Modified file: `src/sae_localization/subgroup_steering.py` — `run_stepwise_sweep()`
- **Vector norm tracking:** Records `vector_norm = float(vec.float().norm())` for each (k, alpha) configuration.
- **Confidence-aware metrics:** Builds per-item result dicts with all fields needed for `compute_all_metrics()`. Full metrics dict stored in each grid record under `"metrics"`.
- **Steering efficiency (eta):** Selection criterion changed from raw correction rate to `eta = RCR_1.0 / ||v||_2`. Higher eta = more correction per unit perturbation.
- **Tie-breaking:** Among configs within 1% of best eta, prefer smallest ||v||_2 (then highest eta).
- **Per-item records:** Accumulates per-item result dicts and returns them for the optimal (k*, alpha*) configuration.
- **Enhanced logging:** Each grid point now logs eta and ||v|| alongside correction/corruption rates.

### Modified file: `scripts/run_subgroup_steering.py`
- **Phase 1 alpha pruning:** Before the main k x alpha grid, runs k=1 across all alphas. Prunes alphas where RCR_1.0 = 0 or degeneration >= 0.05. Keeps all if fewer than 3 are viable. Resume-safe with per-alpha checkpoint files.
- **Exacerbation test:** After finding optimal (k*, alpha*), runs with flipped alpha sign on all items (not just stereotyped) to measure corruption under exacerbation. Results saved in manifest and `exacerbation_results.json`.
- **Per-item parquet:** Saves per-item results for each subgroup's optimal config as `per_item/{cat}_{sub}_optimal.parquet`. Includes item_idx, baseline/steered answers, logits (JSON-encoded), margin, correction status, etc.
- **Full manifest schema:** Builds manifests with all fields from the spec including: steering_efficiency_eta, metrics (all three confidence-aware), phase1_viable_alphas, exacerbation, and null placeholders for downstream benchmark deltas.
- **New figures:**
  - `fig_pareto_frontier_{category}.png/.pdf` — RCR_1.0 vs ||v||_2 scatter per subgroup, colored by k, star marker for optimum, constant-alpha lines.
  - `fig_marginal_analysis_{category}.png/.pdf` — RCR_1.0(k) and ||v||_2(k) on dual axes at optimal alpha, vertical line at k*.
  - `fig_exacerbation_asymmetry.png/.pdf` — Paired bars comparing debiasing RCR vs exacerbation corruption across all subgroups.
- **New CLI flag:** `--skip_exacerbation` to skip the exacerbation phase.

---

## New output structure

```
results/subgroup_steering/<model_id>/<date>/
├── optimal_configs.json
├── steering_manifests.json
├── stepwise_results.json
├── exacerbation_results.json
├── steering_vectors/
│   ├── sexual_orientation_gay.npz
│   └── ...
├── per_item/
│   ├── sexual_orientation_gay_optimal.parquet
│   └── ...
├── stepwise/                    (per-(k,alpha) checkpoints)
│   └── ...
└── figures/
    ├── fig_pareto_frontier_*.png/.pdf
    ├── fig_marginal_analysis_*.png/.pdf
    ├── fig_exacerbation_asymmetry.png/.pdf
    ├── fig_stepwise_correction_*.png/.pdf
    ├── fig_optimal_k_distribution.png/.pdf
    ├── fig_alpha_vs_k_heatmaps_*.png/.pdf
    └── fig_margin_conditioned_correction_*.png/.pdf
```

---

## Part 4: Feature Interpretability Deep Dive

### New file: `scripts/analyze_top_features.py`

Characterizes the features selected by the joint (k, alpha) optimization.

- **Analysis A — Max-activating items:** For each subgroup's optimal features, loads Stage 1 hidden states, encodes through SAE, finds top-20 items by activation magnitude. Reports activation stats (mean, std, stereotyped vs non-stereotyped, fraction nonzero).
- **Analysis B — Stereotype specificity:** Measures whether each feature fires preferentially on items targeting its source subgroup vs other subgroups in the same category. Computes specificity score = `mean_activation(this_sub) / mean_all_items`.
- **Analysis C — Feature co-occurrence:** Pairwise Pearson correlation of activations across items for features within each subgroup's optimal set. Low correlation = each feature captures distinct information.
- **Analysis D — Cross-subgroup activation matrix:** Per category, builds a (features × target_subgroups) matrix of mean activations. If fragmentation holds at the feature level, this should be approximately block-diagonal.
- **`--token_level` flag:** Placeholder for expensive token-level attribution (Option 2 from spec). Default uses last-token activations only (Option 1).

**Reuses logic from:** `src/sae_localization/feature_characterization.py` (data loading patterns, subgroup breakdown approach). Does not duplicate — builds new analyses on top.

**Figures:**
- `fig_top_feature_activations_{category}.png/.pdf` — Horizontal bar chart of top items per subgroup's #1 feature
- `fig_cross_subgroup_activation_heatmap_{category}.png/.pdf` — Features × subgroups heatmap
- `fig_specificity_distribution.png/.pdf` — Histogram of specificity scores with median annotation
- `fig_feature_cooccurrence_{category}.png/.pdf` — Pairwise correlation matrices per subgroup

**Output:** `results/feature_interpretability/<model_id>/<date>/`

---

## Part 5: Universal Backfire Prediction

### New file: `scripts/analyze_universal_backfire.py`

Tests whether pairwise cosine similarity between subgroup directions predicts cross-subgroup steering transfer effects.

- **Step 1 — SAE-based cosines:** For each category with ≥2 subgroups, computes mean of unit-normalized decoder columns for optimal features, then pairwise cosines.
- **Step 1b — DIM-based cosines:** If `subgroup_directions.npz` exists from prior runs, loads DIM directions and computes pairwise cosines for cross-validation.
- **Step 2 — Transfer effects:** For each (source, target) subgroup pair within a category, applies source's steering vector to target's stereotyped items and measures bias change. Resume-safe with per-pair checkpoint files.
- **Step 3 — OLS regression:** Fits `bias_change ~ cosine` with bootstrap 95% CI. Reports r², p-value, slope. Runs with and without Disability category.
- **Step 4 — DIM cross-validation:** If DIM cosines available, repeats regression using DIM cosines as X.

**Figures:**
- `fig_universal_backfire_scatter.png/.pdf` — THE key figure. Two panels: all categories and excl. Disability. Points colored by category with markers, OLS line, 95% CI band.
- `fig_cross_subgroup_transfer_heatmaps.png/.pdf` — Per-category heatmap (source × target) of bias change. RdBu_r colormap centered at 0.
- `fig_cosine_vs_backfire_by_category.png/.pdf` — Faceted scatter, one panel per category with per-category regression lines.
- `fig_sae_vs_dim_cosine_comparison.png/.pdf` — SAE vs DIM cosines scatter (only if DIM available).

**Output:** `results/universal_backfire/<model_id>/<date>/`

---

## Part 6: Probe Selectivity Controls

### New file: `scripts/run_probe_controls.py`

Tests whether linear probes learn identity-specific features or exploit surface cues. Does NOT require the model — uses saved Stage 1 hidden states.

- **Control A — Permutation baseline:** For each category and layer, trains subgroup probe on real vs permuted labels (N=10 permutations). Reports selectivity = real_acc - mean(perm_acc). PCA-50 + LogisticRegression + stratified 5-fold CV.
- **Control B — Structural controls:** Trains probes for context condition (ambiguous vs disambiguated) and answer position (which letter is stereotyped). The gap between identity probe accuracy and structural probe accuracy is the identity-attributable excess.
- **Control C — Cross-category generalization:** Trains binary stereotyped/non-stereotyped probe on one category, tests on another. Prediction: cross-category probes should be near chance (0.5) if bias representations are category-specific.
- **Within-category cross-subgroup generalization:** Trains on subgroup A's items, tests on subgroup B's items. Anti-correlated subgroup pairs should fail.

**Compatibility fix:** Removed deprecated `multi_class="multinomial"` kwarg from `LogisticRegression` (removed in scikit-learn 1.8).

**Figures:**
- `fig_probe_selectivity.png/.pdf` — Real vs permuted accuracy per layer, shaded selectivity gap
- `fig_probe_structural_comparison.png/.pdf` — Identity vs context vs position probes per layer
- `fig_probe_generalization_matrix.png/.pdf` — Cross-category binary probe heatmap
- `fig_within_category_generalization.png/.pdf` — Within-category cross-subgroup heatmaps

**Output:** `results/probe_controls/<model_id>/`

---

## Part 7: Fix `scripts/evaluate_generalization.py`

### 7A — Exacerbation always runs
- **Removed** `--exacerbation` flag. Both debiasing and exacerbation directions run by default on ALL conditions (matched, within-cat mismatched, cross-cat mismatched, no-demographic) and on MMLU.

### 7B — Split mismatched conditions
- **Replaced** single `mismatched` condition with `within_cat_mismatched` (same-category different-subgroup) and `cross_cat_mismatched` (different-category subgroup). Uses `cat_subs` for within-category and `other_cat_subs` for cross-category filtering.

### 7C — Remove hardcoded item limits
- **Replaced** `no_demo[:200]` and `mmlu_items[:200]` with `[:args.max_items]` when `--max_items` is set, otherwise uses all items.

### 7D — Per-item parquet output
- Saves per-item MedQA results as `per_item/medqa_per_item.parquet` with columns: item_idx, condition, steering_vector_key, category, subgroup, baseline/steered answers, margin, logit_shift, demographic_subgroups, etc.

### 7E — Confidence-aware metrics
- `evaluate_items()` now returns fields needed for `compute_all_metrics()`: margin, logit_baseline, logit_steered, stereotyped_option, corrected, corrupted.
- `_summarise()` now includes full `compute_all_metrics()` output under `"metrics"` key.

### 7F — Update manifests
- After running, updates manifests with: `medqa_matched_delta`, `medqa_within_cat_mismatched_delta`, `medqa_cross_cat_mismatched_delta`, `medqa_nodemo_delta`, `medqa_exacerbation_matched_delta`, `mmlu_delta`, `mmlu_worst_subject`, `mmlu_worst_subject_delta`.

### 7G — Output figures
- `fig_medqa_matched_vs_mismatched.png/.pdf` — Per-category grouped bars: accuracy delta by condition (matched, within-cat mismatch, cross-cat mismatch, no-demo)
- `fig_medqa_exacerbation.png/.pdf` — Paired bars: debiasing vs exacerbation delta
- `fig_side_effect_heatmap.png/.pdf` — Heatmap of steering vector × knowledge domain accuracy deltas (MedQA no-demo, MMLU overall, MMLU STEM/humanities/social science)
- `fig_debiasing_vs_exacerbation_asymmetry.png/.pdf` — Scatter: BBQ RCR_1.0 vs MedQA exacerbation accuracy drop

**Output:** `results/generalization/<model_id>/<date>/`

---

## Robustness Fixes (applied across all Parts 4-7)

- **analyze_top_features.py:** Added SAE feature index bounds checking in `analyze_max_activating`, `compute_specificity`, and `compute_cooccurrence`. Added NaN-to-zero cleaning in correlation matrices. Added early-exit with clear warning when no manifests match requested categories. Fixed `compute_cooccurrence` to return identity matrix when n < 3 items (instead of crashing on empty corrcoef).
- **analyze_universal_backfire.py:** Added explicit error + exit when `steering_vectors/` directory is missing or empty. Added warning when bootstrap CI has <80% successful resamples.
- **run_probe_controls.py:** Added `np.nan_to_num` in `_get_raw_hs` to prevent NaN/inf propagation from denormalized hidden states into PCA/probes. Added log message when categories are skipped due to <2 subgroups. Removed deprecated `multi_class="multinomial"` kwarg (scikit-learn 1.8+).
- **evaluate_generalization.py:** `evaluate_items` now uses per-item `letters` field when available (from MedQA loader), falling back to caller default. This prevents silent errors when items have variable answer counts.

---

## Pipeline Integration

### Modified file: `run_local_smoketest.sh`

Added optional SAE pipeline stages gated behind `RUN_SAE=1`:

- **New config vars:** `RUN_SAE`, `SAE_SOURCE`, `SAE_EXPANSION`, `SAE_LAYERS`, `SAE_CATEGORIES`, `SAE_ALPHA_VALUES`, `SAE_K_STEPS`, `MEDQA_PATH`, `MMLU_PATH`
- **SAE Stage 1:** `run_sae_localization.py` — extract hidden states
- **SAE Stage 2:** `run_sae_analysis.py` — SAE feature analysis
- **SAE Stage 2b:** `rank_subgroup_features.py` — rank features per subgroup
- **SAE Stage 3:** `run_subgroup_steering.py` — joint (k, alpha) optimization
- **SAE Stage 4:** `analyze_top_features.py` — feature interpretability
- **SAE Stage 5:** `analyze_universal_backfire.py` — backfire prediction
- **SAE Stage 6:** `run_probe_controls.py` — probe selectivity controls
- **SAE Stage 7:** `evaluate_generalization.py` — MedQA + MMLU generalization

Each stage checks for prerequisite outputs before running and skips with a warning if dependencies are missing. All stages propagate `MAX_ITEMS` for smoketest mode.

**Usage:**
```bash
RUN_SAE=1 SAE_CATEGORIES=so,disability MAX_ITEMS=10 bash run_local_smoketest.sh
```
