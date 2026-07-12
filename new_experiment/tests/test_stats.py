import numpy as np
import pandas as pd
import pytest

from analysis.stats import (METRICS, holm_correction, load_seed_csvs,
                            significance_vs_baseline)


def test_holm_correction_known_values():
    p = np.array([0.01, 0.04, 0.03])
    adj = holm_correction(p)
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_holm_is_monotone_and_capped():
    p = np.array([0.5, 0.9, 0.001])
    adj = holm_correction(p)
    assert (adj <= 1.0).all() and (adj >= p).all()


def _fake_comparison(n_seeds=10):
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(n_seeds):
        for fam, shift in [("base", 0.0), ("fair", -0.05)]:
            row = {"Family": fam, "Seed": seed}
            for m in METRICS:
                row[m] = 0.5 + shift + rng.normal(0, 0.005)
            rows.append(row)
    return pd.DataFrame(rows)


def test_significance_detects_consistent_shift():
    df = _fake_comparison()
    sig = significance_vs_baseline(df, "fair", "base")
    assert list(sig.columns) == ["Metric", "p", "p_holm"]
    assert (sig["p_holm"] < 0.05).all()      # -0.05 shift >> 0.005 noise


def test_load_seed_csvs_reads_seed_dirs(tmp_path):
    for seed in (0, 1):
        d = tmp_path / "german" / f"seed{seed}"
        d.mkdir(parents=True)
        pd.DataFrame({"Family": ["base"], "Seed": [seed], "Accuracy": [0.7]}).to_csv(
            d / "model_comparison.csv", index=False)
    df = load_seed_csvs("german", "model_comparison", results_root=tmp_path)
    assert len(df) == 2 and set(df["Seed"]) == {0, 1}


def test_load_seed_csvs_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_seed_csvs("nope", "model_comparison", results_root=tmp_path)


def test_significance_zero_diff_returns_p1():
    rows = []
    for seed in range(6):
        for fam in ("base", "fair"):
            row = {"Family": fam, "Seed": seed}
            for m in METRICS:
                row[m] = 0.5          # identical -> diff is exactly zero
            rows.append(row)
    df = pd.DataFrame(rows)
    sig = significance_vs_baseline(df, "fair", "base")
    assert (sig["p"] == 1.0).all()
    assert (sig["p_holm"] == 1.0).all()
