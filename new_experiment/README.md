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
# COMPAS (default)
.venv/bin/python new_experiment/run_experiment.py

# German Credit
.venv/bin/python new_experiment/run_experiment.py --dataset german

# Both datasets
.venv/bin/python new_experiment/run_experiment.py --dataset both

# Force retraining (ignore cached checkpoints)
.venv/bin/python new_experiment/run_experiment.py --retrain

# Training knobs
.venv/bin/python new_experiment/run_experiment.py --epochs 500 --patience 30
```

(If your shell already has the venv activated, plain `python` works too.)

## Outputs

Everything lands under `new_experiment/`:

```
MODELS/<dataset>/                      cached .pth checkpoints (one per arch × α × β)
RESULTS/<dataset>/sweep_results.csv    full α×β sweep, both architectures, test + val metrics
RESULTS/<dataset>/model_comparison.csv all models side by side on the test set
RESULTS/<dataset>/PLOTS/
    heatmap_logreg_<metric>.png        α×β heatmap per metric, logistic regression
    heatmap_mlp_<metric>.png           α×β heatmap per metric, MLP
    model_comparison.png               bar charts: all models × all 5 metrics
```

Metrics reported everywhere (test set): Accuracy, Demographic Parity
Difference (all groups), **DPD (Largest 2 Groups)**, Equalized Odds
Difference, Theil Index, Gini Coefficient.

**Why the extra DPD column:** the standard fairlearn DPD/EOD take the max–min
gap across *all* sensitive groups. In COMPAS the test split contains groups
with 3–5 people (Native American, Asian), so those metrics are pinned by
tiny-group noise (almost every model shows DPD ≈ 0.47 regardless of
training). The largest-two-groups version (African-American vs Caucasian —
the comparison ProPublica's analysis focused on) is stable across splits and
shows the real effect of the fairness loss. For German Credit (two sex
groups) the two columns are identical.

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
