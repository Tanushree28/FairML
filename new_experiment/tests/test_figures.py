import numpy as np

from analysis.figures import pareto_mask


def test_pareto_mask_basic():
    acc = np.array([0.90, 0.80, 0.85])
    dpd = np.array([0.10, 0.05, 0.20])
    # (0.85, 0.20) is dominated by (0.90, 0.10); the other two are optimal.
    assert pareto_mask(acc, dpd).tolist() == [True, True, False]


def test_pareto_mask_all_optimal_on_tradeoff_line():
    acc = np.array([0.7, 0.8, 0.9])
    dpd = np.array([0.01, 0.05, 0.10])
    assert pareto_mask(acc, dpd).all()
