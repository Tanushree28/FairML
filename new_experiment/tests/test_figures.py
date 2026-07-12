import itertools

import numpy as np
import pandas as pd

from analysis.figures import pareto_mask
from analysis.stats import DPD2, METRICS


def test_pareto_mask_basic():
    acc = np.array([0.90, 0.80, 0.85])
    dpd = np.array([0.10, 0.05, 0.20])
    # (0.85, 0.20) is dominated by (0.90, 0.10); the other two are optimal.
    assert pareto_mask(acc, dpd).tolist() == [True, True, False]


def test_pareto_mask_all_optimal_on_tradeoff_line():
    acc = np.array([0.7, 0.8, 0.9])
    dpd = np.array([0.01, 0.05, 0.10])
    assert pareto_mask(acc, dpd).all()


def _fake_sweep_dir(tmp_path, n_seeds=2):
    alphas = betas = [0, 0.25, 0.5, 0.75, 1]
    for seed in range(n_seeds):
        rows = []
        for arch, a, b in itertools.product(["logreg", "mlp"], alphas, betas):
            row = {"Arch": arch, "Alpha": a, "Beta": b, "Seed": seed,
                   "Soft DP": 0.1 * b, "Soft GE": 0.1 * (1 - b),
                   "Val Accuracy": 0.7, "Val DPD (Largest 2 Groups)": 0.05,
                   "Val Positive Rate": 0.4}
            for m in METRICS:
                row[m] = 0.5 + 0.1 * a - 0.02 * b + 0.01 * seed
            rows.append(row)
        d = tmp_path / "german" / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(d / "sweep_results.csv", index=False)
    return tmp_path


def test_beta_cross_effect_writes_png(tmp_path):
    from analysis.figures import plot_beta_cross_effect
    root = _fake_sweep_dir(tmp_path)
    out = plot_beta_cross_effect("german", tmp_path, results_root=root)
    assert out.exists() and out.stat().st_size > 0


def test_dpd_vs_theil_writes_png(tmp_path):
    from analysis.figures import plot_dpd_vs_theil
    root = _fake_sweep_dir(tmp_path)
    out = plot_dpd_vs_theil("german", tmp_path, results_root=root)
    assert out.exists() and out.stat().st_size > 0
