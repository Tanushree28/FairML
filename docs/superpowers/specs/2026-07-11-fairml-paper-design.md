# FairML Conference Paper — Design Spec

**Date:** 2026-07-11
**Status:** Approved design, pending implementation plan
**Target:** Full conference paper (FAccT / AIES / ML-conference fairness track)

## 1. Central claim and research questions

**Claim:** An empirical study of the tension between group fairness and
individual fairness. A differentiable composite loss with a β knob
interpolates between the two objectives; sweeping it maps the trade-off
across three standard datasets and two model families. The composite loss is
the *instrument*; the trade-off map is the *contribution*.

- **RQ1:** Is there a measurable tension between group fairness (DPD/EOD)
  and individual fairness (Theil/GE) when each is optimized directly?
- **RQ2:** Is the tension consistent across datasets (race/COMPAS,
  sex/German, sex/Adult) and model classes (LR, MLP)?
- **RQ3:** Where does the composite-loss Pareto frontier sit relative to
  established pre-, in-, and post-processing fairness methods?

## 2. Datasets

| Dataset | Target | Sensitive attr | Change from current state |
|---|---|---|---|
| COMPAS | `two_year_recid` | race | Switch to ProPublica standard features (age, sex, priors_count, charge degree, juvenile counts). **Drop the leaking `duration` feature** (corr −0.78 with target, mechanical). Primary analysis on African-American vs. Caucasian; all-groups numbers kept as supplementary. Expected accuracy drops to literature-typical ~0.67–0.70 — this is correct, not a regression. |
| German Credit | `risk` | sex | Unchanged. |
| Adult (Census) | income > 50K | sex (primary), race (secondary) | **New.** Completes the field-standard dataset trio. |

Shared preprocessing: StandardScaler + one-hot encoding, 60/20/20
train/val/test split, split re-drawn per seed.

## 3. Methods

**Method carriers (differentiable, trained with our loss):**
- Logistic Regression, MLP (64-64)
- Loss: `total = α·BCE_with_logits + (1−α)·[β·SoftDP + (1−β)·SoftGE]`
  (`new_experiment/losses.py`, verified correct — unchanged)
- 7×7 α×β sweep, early stopping on validation loss

**Baselines (priority-ordered per user decision):**

| Priority | Method | Category | Status |
|---|---|---|---|
| ✅ Required | Reweighing (Kamiran & Calders) | pre-processing | new |
| ✅ Required | ExponentiatedGradient (DP + EO) | in-processing | existing |
| ✅ Required | ThresholdOptimizer (DP) | post-processing | existing |
| ✅ Kept (zero cost) | Random Forest | accuracy reference | existing |
| ⏳ Time permitting | XGBoost | accuracy reference | new, optional |
| ❌ Skipped | Adversarial debiasing | in-processing | only if everything else is complete and stable; ExpGrad keeps the in-processing category covered |

**Degenerate-predictor guard:** any model with positive-prediction rate
<2% or >98% is excluded from model selection and flagged in results.

## 4. Experimental protocol

- **10 seeds** per configuration. Seed controls the split, weight init, and
  any stochastic baseline. All tables report mean ± std; all figures carry
  error bars/bands.
- **Model selection:** validation-only. Among α<1 combos, minimize distance
  to utopia point (accuracy=1, DPD-largest-2=0). Test metrics are reported
  only, never used for selection.
- **Significance:** Wilcoxon signed-rank across seeds (fair model vs. its
  baseline, per metric), Holm correction across the metric family.
- **Scale:** ~7×7 × 2 archs × 3 datasets × 10 seeds ≈ 3,000 small tabular
  model runs; cacheable per (dataset, arch, α, β, seed); CPU-feasible.

## 5. Analyses and figures

1. **Fig 1 — Pareto frontiers** (per dataset): accuracy vs. DPD scatter of
   the full sweep, frontier drawn, baselines overlaid with error bars.
2. **Fig 2 — β cross-effect curves** (core contribution): at fixed α, group
   metrics and individual metrics vs. β, with seed bands, per dataset.
   Includes the DPD-vs-Theil trade-off plot.
3. **Fig 3 — Staged improvement bars:** baseline → +fairness loss →
   +leakage fix (COMPAS), error bars + significance stars.
4. **Table 1 — Master comparison:** all methods × metrics × datasets,
   mean ± std, significance marks.
5. **Table 2 — Ablation** (free from the sweep): α=1 (no fairness),
   β=0 (SoftGE only), β=1 (SoftDP only), best interior β.
6. **Supplementary:** seed-averaged α×β heatmaps; surrogate-validation
   scatter (SoftDP vs. hard DPD, SoftGE vs. Theil).

**Emphasized metrics:** DPD (largest-2-groups), Equalized Odds Difference,
Theil index. Secondary: all-groups DPD, Gini. Calibration: related-work
discussion only, no experiments.

## 6. Codebase changes (evolve `new_experiment/` in place)

```
new_experiment/
  data_loading.py      + Adult loader, COMPAS standard-feature mode, per-seed splits
  losses.py            unchanged
  models.py            unchanged
  baselines.py         NEW — Reweighing (+ XGBoost wrapper if time permits)
  run_experiment.py    + --seeds N, per-seed checkpoint caching, always-rewrite CSVs
  analysis/
    stats.py           NEW — Wilcoxon + Holm
    figures.py         NEW — Figs 1–3, Tables 1–2 → RESULTS/paper/
```

Legacy handling: `Compas.py`, `German.py`, `Loss.py`, and `Models.py` move
to `legacy/` with a README note. `GroupFairness.py` and
`IndividualFairness.py` stay at the repo root — the new pipeline imports
them for evaluation. `new_experiment/` is the canonical paper pipeline.

## 7. Build order (user-approved priority)

1. **Stage 1 — Soundness foundation:** COMPAS leakage fix (standard
   features) + per-seed infrastructure. Built together because every
   downstream item consumes per-seed results.
2. **Stage 2 — Adult dataset** onboarding.
3. **Stage 3 — Reweighing baseline** (ExpGrad, ThresholdOptimizer, RF
   already exist).
4. **Stage 4 — Full 10-seed sweep** across all three datasets.
5. **Stage 5 — Analysis layer:** Wilcoxon significance, Pareto frontiers,
   DPD-vs-Theil trade-off plots, staged-improvement figures, master tables.
6. **Stage 6 — Legacy archival + docs.**
7. **Optional (time permitting):** XGBoost reference; adversarial debiasing
   only if everything else is complete and stable.

## 8. Known risks

- COMPAS accuracy will drop to ~0.67–0.70 after the leakage fix — expected
  and correct; frame as the honest baseline in the paper.
- Skipping adversarial debiasing may draw a reviewer request; all three
  method categories remain covered, so this is a revision-time add if asked.
- Tiny-group DPD pinning on COMPAS is handled by the largest-2-groups
  metric with justification in the paper text.
