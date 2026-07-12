# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install dependencies:
```
pip install -r requirements.txt
```

Run experiments (canonical pipeline — trains the α×β sweep + baselines, writes per-seed CSVs):
```
.venv/bin/python new_experiment/run_experiment.py --dataset all --seeds 0-9
```

Generate all paper figures/tables from the per-seed CSVs:
```
.venv/bin/python new_experiment/analysis/run_analysis.py --dataset all
```

Run tests:
```
.venv/bin/pytest new_experiment/tests/ -v
```

## Architecture

Canonical pipeline: `new_experiment/` — differentiable composite fairness
loss (`losses.py`: α·BCE + (1−α)·[β·SoftDP + (1−β)·SoftGE]), LR + MLP
(`models.py`), three datasets (`data_loading.py`: COMPAS [ProPublica
standard features, no duration leakage], German Credit, Adult), reweighing
baseline (`baselines.py`), per-seed sweep results in
`new_experiment/RESULTS/<dataset>/seed<k>/`, paper assets via `analysis/`
into `new_experiment/RESULTS/paper/<dataset>/`.

Evaluation metrics are imported from the repo root: `GroupFairness.py`,
`IndividualFairness.py` — keep them there.

`legacy/` holds the frozen original scripts (known training bugs — see
`legacy/README.md`). Do not extend them.

Design spec: `docs/superpowers/specs/2026-07-11-fairml-paper-design.md`.
