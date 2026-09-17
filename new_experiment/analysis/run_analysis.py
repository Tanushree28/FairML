"""Generate every paper asset from the per-seed result CSVs.

Usage (from repo root):
  .venv/bin/python new_experiment/analysis/run_analysis.py --dataset all
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # analysis/
NEW_EXP = HERE.parent                            # new_experiment/
ROOT = NEW_EXP.parent                            # repo root
for p in (str(ROOT), str(NEW_EXP)):
    if p not in sys.path:
        sys.path.insert(0, p)

from analysis.figures import (ablation_table, master_table, plot_beta_cross_effect,
                              plot_dpd_vs_theil, plot_fair_vs_baseline, plot_pareto,
                              seed_mean_heatmaps, surrogate_scatter)
from analysis.cross_dataset import ALL_DATASETS, run_cross_dataset
from analysis.stats import RESULTS


def main():
    parser = argparse.ArgumentParser(description="Build paper figures and tables")
    parser.add_argument("--dataset", choices=["compas", "german", "adult", "taiwan", "taiwan_edu", "acs_employment", "acs_pubcov", "all"],
                        default="all")
    args = parser.parse_args()
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        out = RESULTS / "paper" / ds
        out.mkdir(parents=True, exist_ok=True)
        print(f"== {ds} -> {out}")
        for fn in (plot_pareto, plot_beta_cross_effect, plot_dpd_vs_theil,
                   plot_fair_vs_baseline, master_table, ablation_table,
                   seed_mean_heatmaps, surrogate_scatter):
            result = fn(ds, out)
            print(f"   {fn.__name__}: {result if isinstance(result, Path) else 'ok'}")

    if args.dataset == "all":
        report_fig = ROOT / "reports" / "fig"
        out = run_cross_dataset(RESULTS / "paper" / "cross_dataset", report_fig_dir=report_fig)
        print(f"== cross-dataset side-by-side -> {out} (report figures -> {report_fig})")


if __name__ == "__main__":
    main()
