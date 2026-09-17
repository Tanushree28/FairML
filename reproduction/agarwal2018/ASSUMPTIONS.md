# Assumptions

- The paper states Adult has 48,842 examples but does not specify missing-value handling. This replication runs both `preserve_missing`, which fills missing categorical values with `__MISSING__` and retains 48,842 rows, and `complete_case`, which drops rows with missing values and retains 45,222 rows.
- The paper does not specify the Adult categorical encoding. This replication uses train-fitted one-hot encoding with `handle_unknown='ignore'` and no dropped category.
- The paper does not specify numeric scaling. This replication uses train-fitted `StandardScaler` for numeric columns.
- The paper says the protected attribute is included in X. This replication includes `sex` and all other non-target Adult columns, including `fnlwgt`, `education`, and `education-num`.
- The original train/test random seed is unavailable. This replication runs seeds 0 through 9.
- The paper says epsilon in `{0.001, ..., 0.1}` but does not specify intermediate spacing. This replication uses the documented log-spaced grid: `[0.001, 0.0031622776601683794, 0.01, 0.03162277660168379, 0.1]`.
- Test error is computed as randomized expected classification error from the positive-class probability/mixed classifier, with `accuracy = 1 - error`. `hard_accuracy` is also reported for thresholded predictions.
- `paper_dp_violation` is the maximum signed Adult demographic-parity moment violation, `max(max_a E[h|A=a]-E[h], max_a E[h]-E[h|A=a])`, computed on positive-class probabilities. `fairlearn_dpd` is Fairlearn's `demographic_parity_difference` on thresholded hard predictions.
- Modern Fairlearn is a reimplementation, not exact authors' code. It uses `DemographicParity(difference_bound=epsilon)` and `ExponentiatedGradient(..., eps=0.01, max_iter=50)`.
