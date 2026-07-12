# New Experiment — fixed fairness loss + model comparison

This folder is an updated version of the original `Compas.py` / `German.py`
pipeline with the training bugs fixed and a proper model comparison added.
The original code at the repo root is untouched.

## What was fixed / changed vs. the original

| # | Original | Here |
|---|----------|------|
| 1 | Fairness terms used `y_pred.round().detach().numpy()` → no gradients, only BCE trained; alpha/beta had almost no effect | `losses.py` computes soft demographic parity + soft generalized entropy in torch on raw probabilities — **all loss terms are differentiable** |
| 2 | Double sigmoid: `forward` applied sigmoid, then the loss applied sigmoid again | Models return logits; loss uses `binary_cross_entropy_with_logits` |
| 3 | `epoch` argument shadowed, hard-coded 100 epochs, no early stopping | `--epochs` / `--patience` flags; early stopping on validation loss with a deep-copied best-weights snapshot |
| 4 | Heatmaps averaged over the group/individual metric choices (collapsed two sweep axes) | One surrogate per term in training (SoftDP + SoftGE), so each alpha×beta heatmap cell is a single model — nothing is averaged away |
| 5 | `results.csv` was never regenerated if it existed (stale-results trap) | Result CSVs are always rewritten; only model checkpoints are cached (`--retrain` to force) |
| 6 | Logistic regression only | Adds an MLP, fairlearn ExponentiatedGradient (DP + Equalized Odds), ThresholdOptimizer post-processing, and a Random Forest accuracy reference |

Loss composition is unchanged in spirit:

```
total = α·BCE + (1−α)·[ β·SoftDP + (1−β)·SoftGE ]
```

- **SoftDP** (group fairness): largest gap in mean predicted probability between sensitive groups.
- **SoftGE** (individual fairness): generalized entropy index (alpha=2) on the benefit vector `b = 1 + p − y` (Speicher et al. 2018) — the differentiable version of the Theil index used in evaluation.

Data loading, features, splits (60/20/20, `random_state=42`), and the
evaluation metrics (fairlearn / aif360, via the root `GroupFairness.py` and
`IndividualFairness.py`) are identical to the original, so numbers are
directly comparable.

## How to run

From the repo root (`FairML/`):

```bash
# Full paper run: all three datasets, seeds 0-9 (~1-3 h)
.venv/bin/python new_experiment/run_experiment.py --dataset all --seeds 0-9

# Single dataset / ad-hoc seed (also produces inline plots)
.venv/bin/python new_experiment/run_experiment.py --dataset german --seeds 42

# Paper figures + tables from the per-seed CSVs
.venv/bin/python new_experiment/analysis/run_analysis.py --dataset all
```

## Outputs

Everything lands under `new_experiment/`:

```
new_experiment/MODELS/<dataset>/seed<k>/    cached .pth checkpoints
new_experiment/RESULTS/<dataset>/seed<k>/   sweep_results.csv, model_comparison.csv
new_experiment/RESULTS/paper/<dataset>/     fig1_pareto, fig2_beta_cross_effect,
                                     fig2b_dpd_vs_theil, fig3_fair_vs_baseline,
                                     table1/table2 (csv + tex), heatmaps,
                                     supp_surrogate_validation
```

## Models in the comparison

| Model | Type |
|-------|------|
| Logistic Regression baseline (BCE only) | baseline, α=1 |
| MLP baseline (BCE only) | baseline, α=1 |
| Fair Logistic Regression (best α, β) | in-processing (this work) |
| Fair MLP (best α, β) | in-processing (this work) |
| ExpGrad LogReg (Demographic Parity) | in-processing (fairlearn reduction) |
| ExpGrad LogReg (Equalized Odds) | in-processing (fairlearn reduction) |
| ThresholdOptimizer LogReg (Demographic Parity) | post-processing (fairlearn) |
| Random Forest | accuracy reference, no fairness |

"Best α, β" is selected on the **validation** set only: among combos with
α < 1, the point closest to utopia (accuracy = 1, DPD = 0), i.e. minimizing
`sqrt((1 − val_acc)² + val_DPD²)` with the largest-two-groups DPD. Test
metrics are only reported, never used for selection.
