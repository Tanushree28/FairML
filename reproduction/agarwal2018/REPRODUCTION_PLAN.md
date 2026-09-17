# Agarwal et al. 2018 Adult Reproduction Plan

This directory is for an isolated reproduction attempt for Agarwal, Beygelzimer,
Dudik, Langford, and Wallach (2018), "A Reductions Approach to Fair
Classification." It must not modify the existing FairML experiment code,
datasets, checkpoints, or results.

Status: planning only. Do not run the experiment yet.

## External Source Provenance

Primary paper:

- Paper page: https://proceedings.mlr.press/v80/agarwal18a.html
- Paper PDF: https://proceedings.mlr.press/v80/agarwal18a/agarwal18a.pdf
- Supplement PDF: https://proceedings.mlr.press/v80/agarwal18a/agarwal18a-supp.pdf

Official authors' code link:

- The paper footnote points to `https://github.com/Microsoft/fairlearn`.
- The Microsoft Research blog post also links "Reductions for Fair Machine
  Learning" to the same repository: https://www.microsoft.com/en-us/research/blog/machine-learning-for-fair-decisions/
- The repository now redirects to https://github.com/fairlearn/fairlearn.

Public Fairlearn code states inspected:

- `v0.1`: `edf973fe2a6d808a4df1af187ff4042c2462c593`
  - Date: 2018-05-14
  - URL: https://github.com/fairlearn/fairlearn/tree/edf973fe2a6d808a4df1af187ff4042c2462c593
  - Contains `fairlearn.classred.expgrad`, `fairlearn.moments.DP`,
    `fairlearn.moments.EO`, and a synthetic `test_fairlearn.py`.
- `v0.2.0`: `aeaa92d536406b54354a6c2db0d0ac5d14897782`
  - Date: 2018-06-20
  - URL: https://github.com/fairlearn/fairlearn/tree/aeaa92d536406b54354a6c2db0d0ac5d14897782
  - Closest public release before ICML 2018 presentation/publication.
  - Adds PyPI packaging but still does not include Adult experiment scripts.
- Modern package in this repo's environment: `fairlearn==0.11.0`
  - Git tag target from upstream: `ea33211bd9c0c8d2102bdd1e4f5cc17c37a97796`
  - URL: https://github.com/fairlearn/fairlearn/tree/ea33211bd9c0c8d2102bdd1e4f5cc17c37a97796

Important provenance conclusion:

- The public official Fairlearn repository appears to publish the reduction
  algorithm implementation, not the exact Adult experiment/data-preprocessing
  pipeline used to generate the paper figures. An exact reproduction of the
  authors' Adult experiment is therefore not currently possible from public
  official code alone.
- A modern Fairlearn reimplementation is possible, but it must be labeled as a
  reimplementation, not an exact authors-code reproduction.

## Original Paper Setup

From the paper's Section 4:

- Task: binary classification subject to either demographic parity (DP) or
  equalized odds (EO).
- Main method: exponentiated-gradient reduction.
- Additional reduction in supplement: grid-search reduction.
- Baselines: Hardt et al. score-based post-processing, Kamiran and Calders
  reweighting and relabeling for demographic parity, plus unconstrained
  classifiers.
- Base classifiers: weighted scikit-learn logistic regression and
  gradient-boosted decision trees.
- Datasets: Adult, Adult4, COMPAS, Law Schools, Dutch census.
- Adult:
  - Paper states `48,842` examples.
  - Target: predict income greater than `$50k`.
  - Protected attribute for `adult`: gender.
  - Protected attribute for `adult4`: joint gender and race, where race is
    binarized into white vs non-white, producing four protected values.
- Split: random `75%` training and `25%` test split.
- Feature access: protected attribute is included in the feature vector for all
  algorithms, so all methods have access to it at train and test time.
- Evaluation metrics:
  - Classification error on test examples.
  - DP violation:
    `max_a E[h(X) | A=a] - E[h(X)]`, implemented in the public 2018 code as
    the maximum signed constraint violation over groups.
  - EO violation:
    `max_{a,y} E[h(X) | A=a,Y=y] - E[h(X) | Y=y]`, again as maximum signed
    constraint violation.
- Fairness tradeoff:
  - The paper says it considered `epsilon in {0.001, ..., 0.1}` and ran
    Algorithm 1 with `c_k = epsilon` for all constraints.
  - The exact spacing/cardinality of this epsilon grid is not specified in the
    paper text.
- Reported results:
  - Main paper Figure 1: test classification error vs fairness-constraint
    violation for DP and EO, including Adult and Adult4, with logistic
    regression and boosting panels.
  - Supplement Figure 2: training error vs violation.
  - Supplement Figure 3: test error vs violation with both grid search and
    exponentiated gradient.
  - The source reports qualitative conclusions and plotted frontiers, not exact
    numeric Adult result tables. Therefore this plan does not set exact numeric
    expected values.

## Official Public Code Findings

The `v0.2.0` public code exposes:

- `fairlearn.classred.expgrad(dataX, dataA, dataY, learner, cons=moments.DP(),
  eps=0.01, T=50, nu=None, eta_mul=2.0, debug=False)`.
- Default algorithm parameters:
  - `eps=0.01`
  - `T=50`
  - `nu=None`, automatically set from empirical uncertainty in error
  - `eta_mul=2.0`
  - `B=1/eps`
  - `_RUN_LP_STEP=True`
  - `_MIN_T=5`
  - `_ACCURACY_MUL=0.5`
- Constraint moments:
  - `moments.DP()` for demographic parity.
  - `moments.EO()` for equalized odds.
- Objective:
  - `moments.MisclassError`, empirical 0-1 error.
- Required learner interface:
  - `fit(X, Y, W)` and `predict(X)`, where `W` is the sample-weight vector.

The public code does not expose:

- Adult data loading code.
- Missing-value handling.
- Categorical encoding.
- Train/test random seed.
- The exact scikit-learn estimator classes or hyperparameters used in the paper
  figures.
- Epsilon grid values beyond the textual `{0.001, ..., 0.1}`.
- The code that generated Figures 1, 2, or 3.

## Settings We Can Reproduce Exactly

Using the public 2018 Fairlearn code:

- The exponentiated-gradient algorithm as published in the official repository,
  pinned to `v0.2.0` commit `aeaa92d536406b54354a6c2db0d0ac5d14897782`.
- The DP and EO moment definitions from that same code.
- The default public-code optimization parameters listed above.
- The randomized-classifier output as a weighted mixture of learned base
  classifiers, evaluated as probabilistic decisions or sampled hard decisions
  depending on the reproduction script's evaluation choice.

Using the paper text:

- Adult target, sensitive attribute for `adult`, and joint sensitive attribute
  construction for `adult4`.
- 75/25 random train/test split as the experimental protocol.
- Protected attribute included in the feature matrix.
- Error and constraint-violation metric definitions.
- Logistic regression and gradient-boosted decision trees as base classifier
  families.

## Settings We Cannot Reproduce Exactly From Public Sources

- Exact Adult preprocessing. The paper says Adult has `48,842` examples, which
  suggests no complete-case drop to `45,222`, but it does not specify how `?`
  missing values were encoded or handled.
- Exact categorical encoding and numerical scaling.
- Exact scikit-learn estimator class names, solver choices, regularization,
  boosting parameters, random states, or package versions.
- Exact train/test split seed.
- Exact epsilon grid values between `0.001` and `0.1`.
- Exact evaluation convention for randomized classifiers in the figures:
  expected predictions/probabilities vs sampled hard predictions vs convex
  envelopes of selected classifiers.
- Exact code for reweighting, relabeling, and post-processing baselines as used
  in the paper figures.
- Exact numeric Adult results, since the paper reports them graphically rather
  than as a numeric table.

## Difference From `new_experiment/`

Current FairML Adult pipeline:

- Data: OpenML Adult v2 cached at `data/adult.csv`.
- Missing values: replaces `?` with missing and drops rows, yielding `45,222`
  rows.
- Target: `class == ">50K"` encoded as `1`.
- Sensitive attribute: `sex`.
- Features: excludes `fnlwgt`; drops `education` in favor of `education-num`;
  retains `sex` as a feature.
- Split: unstratified `60/20/20` train/validation/test, using the same seed for
  both split calls.
- Preprocessing: fit train-only `StandardScaler` for numeric features and
  `OneHotEncoder(drop="first", handle_unknown="ignore")` for categoricals.
- Fairlearn baselines:
  - `ExponentiatedGradient(SkLogisticRegression(max_iter=1000),
    DemographicParity())`
  - `ExponentiatedGradient(SkLogisticRegression(max_iter=1000),
    EqualizedOdds())`
  - Predictions evaluated on the FairML test split.
- Metrics: accuracy, Fairlearn demographic parity difference, DPD on largest
  two groups, Fairlearn equalized odds difference, AIF360 generalized entropy
  error with `alpha=2`, and custom Gini coefficient.

Differences from Agarwal et al. setup:

- Paper uses 75/25 train/test; FairML uses 60/20/20 and validation selection.
- Paper Adult count is `48,842`; FairML drops missing values and uses `45,222`.
- Paper evaluates error, not accuracy.
- Paper evaluates constraint violation relative to the moment definitions;
  FairML reports Fairlearn metric differences and additional individual
  fairness metrics.
- Paper sweeps fairness bounds/tradeoffs; FairML uses modern Fairlearn ExpGrad
  as a comparison row without reproducing the paper's epsilon frontier.
- Paper includes both logistic regression and gradient-boosted decision trees;
  FairML's ExpGrad baseline uses logistic regression only.
- Paper includes `adult4`; FairML's Adult experiment uses `sex` only.
- Paper's public 2018 API used `expgrad(..., eps=...)` where `eps` acted as the
  constraint bound and algorithm scale. Modern Fairlearn separates constraint
  objects from `ExponentiatedGradient(eps=...)`, so a modern implementation is
  not mechanically identical.

## Exact Reproduction vs Modern Reimplementation

Exact reproduction attempt:

- Should use the 2018 public Fairlearn code at
  `aeaa92d536406b54354a6c2db0d0ac5d14897782`.
- Should implement an Adult loader and preprocessing layer that follows the
  paper text as closely as possible, while explicitly marking all assumptions.
- Should produce error-vs-violation frontiers for:
  - `adult / DP / log. reg.`
  - `adult / EO / log. reg.`
  - optionally the boosting and `adult4` panels.
- Cannot claim exactness unless the missing preprocessing, split seed,
  estimator hyperparameters, and epsilon grid are recovered from another
  official source.

Modern Fairlearn reimplementation:

- May use installed `fairlearn==0.11.0` and current
  `fairlearn.reductions.ExponentiatedGradient`.
- Should be labeled as a modern API reimplementation of the reduction idea, not
  an authors-code reproduction.
- Should use explicit `DemographicParity(difference_bound=...)` or
  `EqualizedOdds(difference_bound=...)` to emulate the paper's `c_k=epsilon`
  sweep, rather than relying on default constraint bounds.
- Should still use a 75/25 split, include `sex` in features, and report error
  and moment-style constraint violation if the goal is paper comparability.

## Expected Paper Results

Only explicitly reported expectations:

- The Adult experiment appears in the paper's Figure 1 and supplement Figures 2
  and 3 as error-vs-constraint-violation frontiers for DP and EO, logistic
  regression and boosting.
- The paper states that the exponentiated-gradient reduction generally
  dominated or matched the baselines up to statistical uncertainty, and that
  almost all approaches substantially reduced or removed disparity without much
  impact on classifier accuracy.
- The paper does not provide exact numeric Adult coordinates in a table. Any
  future numeric target should be derived only by rerunning the recovered code
  or by explicitly digitizing the published figures and labeling the result as
  approximate.

## Proposed Implementation Steps, Not Yet Run

1. Create a self-contained script under `reproduction/agarwal2018/` that does
   not import `new_experiment` modules.
2. Add two modes:
   - `--mode authors-code`: vendored or dynamically imported 2018 Fairlearn
     code pinned to `aeaa92d536406b54354a6c2db0d0ac5d14897782`.
   - `--mode modern-fairlearn`: installed `fairlearn==0.11.0`.
3. Implement Adult data preparation with a clearly documented assumption switch:
   - complete-case drop to match common modern practice, or
   - retain `?` as a category to preserve the paper's stated `48,842` rows.
4. Use a 75/25 train/test split with an explicit seed argument.
5. Use `sex` as the `adult` sensitive attribute and include it in `X`.
6. Build optional `adult4` by crossing `sex` with `race == White`.
7. Encode categoricals with a train-fitted one-hot encoder and document that the
   paper did not specify this step.
8. Sweep an explicit epsilon grid. If no official grid is recovered, use a
   documented grid such as log-spaced values from `0.001` to `0.1` and label it
   approximate.
9. Report:
   - test classification error
   - DP/EO moment violation
   - number of base classifiers/oracle calls
   - exact code/data/package hashes
10. Do not write into existing `new_experiment/RESULTS`, `MODELS`, `data`, or
    paper output directories.
