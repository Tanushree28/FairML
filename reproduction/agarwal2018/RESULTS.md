# Results

## What Was Reproduced Directly From Historical Public Code

- The historical `fairlearn.classred.expgrad` algorithm and `moments.DP` constraint were run from Fairlearn v0.2.0 commit `aeaa92d536406b54354a6c2db0d0ac5d14897782`.
- Synthetic historical ExpGrad verification: `{"dp_violation": 0.050000000000000044, "error": 0.1699672825250192, "n_oracle_calls": 18, "n_predictors": 2, "passed": true}`.

## What Required Assumptions

- Adult preprocessing, encoding, scaling, split seed, logistic-regression solver, and exact epsilon-grid spacing are not available in the paper/public code and are documented in `ASSUMPTIONS.md`.

## Historical Implementation Results

Historical rows are labeled `authors-code` in `results/runs.csv` and `results/summary.csv`.

## Modern Fairlearn Reimplementation Results

Modern rows are labeled `modern-fairlearn`. They are not exact reproductions of the authors' experiment.

## Key Result Table

| implementation   | preprocessing    |   epsilon |   error_mean |   error_std |   paper_dp_violation_mean |   paper_dp_violation_std |   fairlearn_dpd_mean |   n_predictors_mean |
|:-----------------|:-----------------|----------:|-------------:|------------:|--------------------------:|-------------------------:|---------------------:|--------------------:|
| authors-code     | complete_case    |  0.001000 |     0.171768 |    0.002553 |                  0.003955 |                 0.003001 |             0.011526 |            2.000000 |
| authors-code     | complete_case    |  0.003162 |     0.171189 |    0.002632 |                  0.004661 |                 0.003836 |             0.014774 |            2.000000 |
| authors-code     | complete_case    |  0.010000 |     0.169425 |    0.002690 |                  0.010804 |                 0.004701 |             0.019469 |            2.000000 |
| authors-code     | complete_case    |  0.031623 |     0.164731 |    0.002685 |                  0.032448 |                 0.005084 |             0.043467 |            2.000000 |
| authors-code     | complete_case    |  0.100000 |     0.154977 |    0.003080 |                  0.099926 |                 0.006263 |             0.140741 |            2.000000 |
| authors-code     | preserve_missing |  0.001000 |     0.165937 |    0.001953 |                  0.004015 |                 0.002612 |             0.010244 |            2.000000 |
| authors-code     | preserve_missing |  0.003162 |     0.165307 |    0.001908 |                  0.004378 |                 0.003511 |             0.010319 |            2.000000 |
| authors-code     | preserve_missing |  0.010000 |     0.163429 |    0.001877 |                  0.009566 |                 0.005136 |             0.014311 |            2.000000 |
| authors-code     | preserve_missing |  0.031623 |     0.158911 |    0.002130 |                  0.030700 |                 0.005167 |             0.043204 |            2.000000 |
| authors-code     | preserve_missing |  0.100000 |     0.148992 |    0.002306 |                  0.098412 |                 0.005133 |             0.144819 |            2.000000 |
| modern-fairlearn | complete_case    |  0.001000 |     0.171881 |    0.002609 |                  0.003832 |                 0.003112 |             0.012776 |            2.000000 |
| modern-fairlearn | complete_case    |  0.003162 |     0.171216 |    0.002632 |                  0.004634 |                 0.003810 |             0.015260 |            2.000000 |
| modern-fairlearn | complete_case    |  0.010000 |     0.169431 |    0.002663 |                  0.010896 |                 0.004686 |             0.018382 |            2.000000 |
| modern-fairlearn | complete_case    |  0.031623 |     0.164584 |    0.002687 |                  0.032503 |                 0.005162 |             0.045928 |            2.000000 |
| modern-fairlearn | complete_case    |  0.100000 |     0.155043 |    0.002961 |                  0.100026 |                 0.006134 |             0.144462 |            2.000000 |
| modern-fairlearn | preserve_missing |  0.001000 |     0.165894 |    0.001919 |                  0.004025 |                 0.002709 |             0.010221 |            2.000000 |
| modern-fairlearn | preserve_missing |  0.003162 |     0.165301 |    0.001881 |                  0.004375 |                 0.003512 |             0.010295 |            2.000000 |
| modern-fairlearn | preserve_missing |  0.010000 |     0.163416 |    0.001872 |                  0.009563 |                 0.005115 |             0.014311 |            2.000000 |
| modern-fairlearn | preserve_missing |  0.031623 |     0.158912 |    0.002214 |                  0.030613 |                 0.005177 |             0.043373 |            2.000000 |
| modern-fairlearn | preserve_missing |  0.100000 |     0.149081 |    0.002261 |                  0.098452 |                 0.005135 |             0.138073 |            2.000000 |

## Comparison With Agarwal et al. (2018)

The paper reports Adult results graphically as error-vs-constraint-violation frontiers, not exact numeric coordinates. This replication therefore compares qualitatively: lower epsilon values generally trace lower demographic-parity violation at some error cost, but no claim of numerical reproduction is made.

## Differences From Existing FairML ExpGrad-DP Experiment

- Existing `new_experiment/` uses a 60/20/20 split and validation workflow; this replication uses paper-style 75/25 train/test.
- Existing `new_experiment/` drops Adult missing rows; this replication runs both full-row missing-as-category and complete-case variants.
- Existing FairML reports accuracy and several additional fairness metrics; this replication emphasizes test error and paper-style DP violation.
- Existing FairML uses modern Fairlearn as a baseline row; this replication separates pinned historical authors-code from modern Fairlearn reimplementation.

## Validation

- Pre-run sanity failures: `[]`
- Result sanity failures: `[]`
- Failed experiment rows: `0`
