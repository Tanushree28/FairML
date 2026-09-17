# Agarwal et al. (2018) Adult DP LR Replication

This directory contains an isolated replication/reimplementation study for the Adult + demographic-parity + logistic-regression experiment associated with Agarwal et al. (2018).

## Rerun From Scratch

From the repository root:

```bash
.venv/bin/python -m pytest reproduction/agarwal2018/tests
MPLCONFIGDIR=/private/tmp .venv/bin/python reproduction/agarwal2018/run_replication.py
.venv/bin/python -m pytest reproduction/agarwal2018/tests
```

Outputs:

- `results/runs.csv`
- `results/summary.csv`
- `figures/error_vs_dp_violation.png`
- `ASSUMPTIONS.md`
- `ENVIRONMENT.md`
- `RESULTS.md`

The script writes only under `reproduction/agarwal2018/`.
