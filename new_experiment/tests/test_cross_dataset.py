import itertools

import numpy as np
import pandas as pd

from analysis.figures import PAIRS
from analysis.stats import METRICS

DATASETS = ["compas", "adult", "acs_pubcov", "acs_employment", "taiwan", "german"]


def _fake_results(root, n_seeds=6):
    rng = np.random.default_rng(0)
    grid = [0, 0.25, 0.5, 0.75, 1]
    for name, seed in itertools.product(DATASETS, range(n_seeds)):
        d = root / name / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        sweep = []
        for arch, a, b in itertools.product(["logreg", "mlp"], grid, grid):
            dpd = 0.2 * (1 - b * (1 - a)) + rng.normal(0, 0.01)
            row = {"Arch": arch, "Alpha": a, "Beta": b, "Seed": seed,
                   "Val Accuracy": 0.8 - 0.05 * (1 - a) + rng.normal(0, 0.005),
                   "Val DPD (Largest 2 Groups)": dpd + rng.normal(0, 0.01),
                   "Val Positive Rate": 0.4, "Soft DP": dpd, "Soft GE": 0.1}
            for m in METRICS:
                row[m] = 0.5 + rng.normal(0, 0.01)
            row["DPD (Largest 2 Groups)"] = dpd
            row["Accuracy"] = row["Val Accuracy"]
            sweep.append(row)
        pd.DataFrame(sweep).to_csv(d / "sweep_results.csv", index=False)
        comp = []
        for base, fair in PAIRS:
            for fam, shift in [(base, 0.0), (fair, -0.05)]:
                row = {"Model": fam, "Family": fam, "Seed": seed, "Alpha": 0.5, "Beta": 0.75}
                for m in METRICS:
                    row[m] = 0.3 + shift + rng.normal(0, 0.005)
                comp.append(row)
        pd.DataFrame(comp).to_csv(d / "model_comparison.csv", index=False)
    return root


def test_cross_dataset_writes_all_outputs(tmp_path):
    from analysis.cross_dataset import run_cross_dataset
    root = _fake_results(tmp_path / "RESULTS")
    out = run_cross_dataset(tmp_path / "out", results_root=root, with_profile=False,
                            report_fig_dir=tmp_path / "reports" / "fig")
    for stem in ["cross_table1_headline", "cross_table2_dissociation",
                 "cross_table3_selection_diagnostics"]:
        assert (out / f"{stem}.csv").exists() and (out / f"{stem}.tex").exists()
    for fig in ["cross_fig1_pareto", "cross_fig2_beta", "cross_fig3_dissociation",
                "cross_fig4_selection_diagnostic"]:
        assert (out / f"{fig}.png").stat().st_size > 0

    headline = pd.read_csv(out / "cross_table1_headline.csv")
    assert len(headline) == 8                     # 3 main + 1 boundary, x 2 archs
    assert set(headline["Role"]) == {"main", "boundary"}
    assert not headline["Dataset"].str.contains("German|Taiwan").any()
    diag = pd.read_csv(out / "cross_table3_selection_diagnostics.csv")
    assert len(diag) == 12                        # all six datasets x 2 archs
    assert set(diag.loc[diag["Dataset"].str.contains("German|Taiwan"), "Role"]) == {"discussion"}
    assert sorted(f.name for f in (tmp_path / "reports" / "fig").iterdir()) == [
        "fig1_pareto.png", "fig2_beta.png", "fig3_dissociation.png", "fig4_selection_diagnostic.png"]
