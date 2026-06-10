# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install dependencies:
```
pip install -r requirements.txt
```

Run the full experiment sweep (downloads ProPublica COMPAS dataset, trains/loads models, writes `results.csv` and heatmaps to `PLOTS/`):
```
python Compas.py
```

There is no test suite, linter, or build step configured.

Notes on re-running:
- `Compas.py` skips training when a `MODELS/<model_name>.pth` checkpoint already exists. Delete the relevant `.pth` files in `MODELS/` to force retraining.
- `Compas.py` skips writing `results.csv` if it already exists (it loads it instead). Delete `results.csv` to regenerate from a new sweep.

## Architecture

This is a research script that trains a logistic-regression classifier on COMPAS recidivism data with a **composite loss combining binary cross-entropy, a group-fairness term, and an individual-fairness term**, then sweeps over the weighting hyperparameters and produces fairness/accuracy heatmaps.

Entry point and orchestration: [Compas.py](Compas.py)
- Loads COMPAS data from a remote URL, splits 60/20/20 train/val/test, preprocesses with `ColumnTransformer` (StandardScaler + OneHotEncoder), keeps `race` as the sensitive attribute.
- Builds the cartesian product of `alpha_list × beta_list × group_fairness_list × individual_fairness_list` (~196 combinations) and trains/evaluates one model per combo.

Loss composition: [Loss.py](Loss.py)
- `custom_loss_function` blends three signals:
  - `total = alpha * bce + (1 - alpha) * (beta * group_fairness + (1 - beta) * individual_fairness)`
- **Important:** the fairness terms call into `fairlearn`/numpy on `y_pred.round().detach().numpy()`, which breaks autograd for the fairness components — only the BCE term contributes gradients during training. Treat any change to this loss carefully if true differentiable fairness is intended.

Model: [Models.py](Models.py)
- `LogisticRegressionModel` is a single `nn.Linear` + sigmoid. `train_model` runs 100 Adam epochs (`lr=0.01`, `weight_decay=1e-4`); the `epoch` argument is shadowed by the loop and unused.

Fairness metrics:
- Group metrics in [GroupFairness.py](GroupFairness.py): demographic parity, equalized odds, equal opportunity, disparate impact (all built on `fairlearn.metrics.MetricFrame`).
- Individual metrics in [IndividualFairness.py](IndividualFairness.py): Theil index, generalized entropy, Atkinson, Gini.

Outputs:
- `MODELS/` — one `.pth` per `(alpha, beta, group_fairness, individual_fairness)` combination, named `model_alpha_{a}_beta_{b}_group_{g}_individual_{i}.pth`.
- `results.csv` — accuracy + fairness metrics per combination.
- `PLOTS/heatmap_<metric>.png` — alpha×beta heatmaps per metric. **Caveat:** the pivot in `Compas.py` aggregates over `group_fairness` and `individual_fairness`, collapsing those axes via mean.

Unused/dead code to be aware of: `EarlyStopping.py` is not imported anywhere; `average_odds_difference` in `GroupFairness.py` is defined but not used in the sweep.
