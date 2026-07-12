# FairML Paper Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve `new_experiment/` into the conference-paper pipeline: leakage-free COMPAS, Adult dataset, 10-seed runs, Reweighing baseline, Wilcoxon significance, and paper-grade figures/tables (Pareto frontiers, β cross-effect, comparison bars).

**Architecture:** The existing corrected pipeline (`data_loading.py`, `losses.py`, `models.py`, `run_experiment.py`) gains seed parameterization and one new dataset; per-seed results land in `RESULTS/<dataset>/seed<k>/`. A new `analysis/` package reads those CSVs and produces all paper assets into `RESULTS/paper/<dataset>/`. Training code and analysis code never share state except the CSV schema.

**Tech Stack:** Python 3.13 (`.venv`), PyTorch 2.6, scikit-learn 1.5, fairlearn 0.11, scipy 1.17, pandas, seaborn/matplotlib, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-11-fairml-paper-design.md`. Follow its priority order; adversarial debiasing is explicitly skipped.
- Interpreter is always `.venv/bin/python` / `.venv/bin/pytest` from repo root `/Users/tanushreenepal/Desktop/FairML`.
- Evaluation metrics stay imported from root `GroupFairness.py` / `IndividualFairness.py` — do not reimplement or move them.
- `losses.py` and `models.py` are verified correct — do not modify them.
- CSV schema contract (consumed by `analysis/`): sweep rows have columns `Model, Arch, Alpha, Beta, Seed, <6 metrics>, Soft DP, Soft GE, Val Accuracy, Val DPD (Largest 2 Groups), Val Positive Rate, Train Time (s), Trained This Run`; comparison rows have `Model, Family, Seed, Alpha, Beta, <6 metrics>`. The 6 metrics are exactly: `Accuracy`, `Demographic Parity Difference`, `DPD (Largest 2 Groups)`, `Equalized Odds Difference`, `Theil Index`, `Gini Coefficient`.
- Family names (stable across seeds, used as join keys): `Logistic Regression baseline`, `Fair Logistic Regression`, `MLP (64-64) baseline`, `Fair MLP (64-64)`, `ExpGrad LogReg (Demographic Parity)`, `ExpGrad LogReg (Equalized Odds)`, `ThresholdOptimizer LogReg (Demographic Parity)`, `Reweighing LogReg (Kamiran-Calders)`, `Random Forest (no fairness)`.
- Paper seeds are `0-9`. Default CLI seed stays `42` for ad-hoc runs.
- Commit after every task (messages given per task).

---

### Task 1: Test infrastructure + seed-parameterized data loading

**Files:**
- Create: `new_experiment/tests/__init__.py` (empty), `new_experiment/tests/conftest.py`, `new_experiment/tests/test_data_loading.py`
- Modify: `new_experiment/data_loading.py` (function `load_dataset`, lines 62–75)
- Modify: `requirements.txt` (append `pytest`, `scipy`)

**Interfaces:**
- Consumes: existing `load_dataset(name) -> dict`.
- Produces: `load_dataset(name: str, seed: int = 42) -> dict` — same return keys as today; `seed` sets `random_state` of both `train_test_split` calls. All later tasks call it with an explicit seed.

- [ ] **Step 1: Install pytest and pin it in requirements**

```bash
.venv/bin/pip install pytest
```

Append to `requirements.txt` (scipy is already installed as a transitive dep; pin it explicitly since `analysis/` will import it):

```
scipy==1.17.1
pytest==8.4.1
```

(Use the version `pip show pytest` reports if different.)

- [ ] **Step 2: Write the failing test**

`new_experiment/tests/conftest.py`:

```python
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # new_experiment/
ROOT = HERE.parent                              # repo root
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
```

`new_experiment/tests/test_data_loading.py` (German only — local files, no network):

```python
from data_loading import load_dataset


def test_same_seed_is_reproducible():
    a = load_dataset("german", seed=0)
    b = load_dataset("german", seed=0)
    assert a["y_test"].equals(b["y_test"])
    assert (a["X_test"] == b["X_test"]).all()


def test_different_seeds_give_different_splits():
    a = load_dataset("german", seed=0)
    b = load_dataset("german", seed=1)
    assert not a["y_test"].equals(b["y_test"])


def test_default_seed_matches_seed_42():
    a = load_dataset("german")
    b = load_dataset("german", seed=42)
    assert a["y_test"].equals(b["y_test"])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest new_experiment/tests/test_data_loading.py -v`
Expected: FAIL / ERROR with `TypeError: load_dataset() got an unexpected keyword argument 'seed'`

- [ ] **Step 4: Implement seed parameter**

In `new_experiment/data_loading.py`, change the `load_dataset` signature and the two split lines:

```python
def load_dataset(name, seed=42):
    """Returns a dict with torch tensors, numpy arrays, sensitive-feature
    series, and integer group ids for train/val/test. `seed` controls the
    train/val/test split."""
```

and

```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest new_experiment/tests/ -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add new_experiment/tests new_experiment/data_loading.py requirements.txt
git commit -m "feat: seed-parameterized splits + pytest infrastructure"
```

---

### Task 2: COMPAS leakage fix (ProPublica standard features)

**Files:**
- Modify: `new_experiment/data_loading.py` (`_load_compas_frame`, lines 27–38)
- Test: `new_experiment/tests/test_data_loading.py` (append)
- Delete (stale artifacts, regenerated later): `new_experiment/MODELS/compas/`, and move `new_experiment/RESULTS/compas/*.csv` to `new_experiment/RESULTS/compas/legacy_leaky/`

**Interfaces:**
- Produces: `_load_compas_frame()` returns features **without** `duration`, with ProPublica's standard row filters applied. `load_dataset("compas")["input_dim"]` changes (old checkpoints incompatible — hence the deletion). The archived `legacy_leaky/model_comparison.csv` is consumed by Task 10's Fig 3 as the "leaky features" reference.

- [ ] **Step 1: Write the failing test** (append to `test_data_loading.py`)

```python
def test_compas_has_no_duration_feature():
    ds = load_dataset("compas", seed=0)
    # duration = end - start leaks the label (corr -0.78, mechanical); the
    # paper uses ProPublica's standard feature set instead.
    total = len(ds["y_train"]) + len(ds["y_val"]) + len(ds["y_test"])
    assert 6000 <= total <= 6300          # ProPublica filters => ~6172 rows
    from data_loading import _load_compas_frame
    X, y, sens = _load_compas_frame()
    assert "duration" not in X.columns
    assert sens == "race"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest new_experiment/tests/test_data_loading.py::test_compas_has_no_duration_feature -v`
Expected: FAIL on `"duration" not in X.columns` (or the row-count assert)

- [ ] **Step 3: Implement the fix**

Replace `_load_compas_frame` in `new_experiment/data_loading.py`:

```python
def _load_compas_frame():
    if COMPAS_CACHE.exists():
        data = pd.read_csv(COMPAS_CACHE)
    else:
        data = pd.read_csv(COMPAS_URL)
        COMPAS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(COMPAS_CACHE, index=False)

    # ProPublica's standard row filters (github.com/propublica/compas-analysis).
    # NOTE: the previous version added duration = end - start, which correlates
    # -0.78 with the label for mechanical reasons (re-offending truncates the
    # observation window) — label leakage, removed for the paper.
    data = data[
        (data["days_b_screening_arrest"] <= 30)
        & (data["days_b_screening_arrest"] >= -30)
        & (data["is_recid"] != -1)
        & (data["c_charge_degree"] != "O")
        & (data["score_text"] != "N/A")
    ]

    features = ["age", "sex", "juv_fel_count", "juv_misd_count",
                "juv_other_count", "priors_count", "c_charge_degree", "race"]
    return data[features], data["two_year_recid"], "race"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest new_experiment/tests/ -v`
Expected: 4 passed

- [ ] **Step 5: Archive stale COMPAS artifacts**

```bash
rm -rf new_experiment/MODELS/compas
mkdir -p new_experiment/RESULTS/compas/legacy_leaky
mv new_experiment/RESULTS/compas/sweep_results.csv \
   new_experiment/RESULTS/compas/model_comparison.csv \
   new_experiment/RESULTS/compas/legacy_leaky/ 2>/dev/null || true
```

- [ ] **Step 6: Commit**

```bash
git add -A new_experiment/data_loading.py new_experiment/tests new_experiment/RESULTS
git commit -m "fix: remove COMPAS duration leakage, use ProPublica standard features"
```

---

### Task 3: Adult dataset loader

**Files:**
- Modify: `new_experiment/data_loading.py` (add `_load_adult_frame`, register in `load_dataset`)
- Test: `new_experiment/tests/test_data_loading.py` (append)

**Interfaces:**
- Produces: `load_dataset("adult", seed)` — same dict shape; sensitive attribute `"sex"`; positive class = income > 50K. Cached at `data/adult.csv` after first download (via `sklearn.datasets.fetch_openml("adult", version=2)`).

- [ ] **Step 1: Write the failing test** (append to `test_data_loading.py`)

```python
def test_adult_loads_with_sex_sensitive():
    ds = load_dataset("adult", seed=0)
    assert ds["sensitive_col"] == "sex"
    total = len(ds["y_train"]) + len(ds["y_val"]) + len(ds["y_test"])
    assert 40000 < total < 49000          # 48842 rows minus missing-value rows
    assert set(ds["y_test"].unique()) <= {0, 1}
    from data_loading import _load_adult_frame
    X, y, sens = _load_adult_frame()
    assert "fnlwgt" not in X.columns      # sampling weight, standard exclusion
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest new_experiment/tests/test_data_loading.py::test_adult_loads_with_sex_sensitive -v`
Expected: FAIL with `ValueError: Unknown dataset: adult` (and ImportError for `_load_adult_frame`)

- [ ] **Step 3: Implement the loader**

In `new_experiment/data_loading.py`, add below `COMPAS_CACHE`:

```python
ADULT_CACHE = ROOT / "data" / "adult.csv"
```

Add below `_load_german_frame`:

```python
def _load_adult_frame():
    """Adult Census Income (OpenML 'adult' v2, the 48842-row two-file version).

    Positive class: income > 50K. Sensitive attribute: sex. `fnlwgt` is a
    census sampling weight, not a person-level feature — standard exclusion;
    `education` is dropped in favor of the ordinal `education-num`.
    """
    if ADULT_CACHE.exists():
        data = pd.read_csv(ADULT_CACHE)
    else:
        from sklearn.datasets import fetch_openml
        raw = fetch_openml("adult", version=2, as_frame=True)
        raw.frame.to_csv(ADULT_CACHE, index=False)
        data = pd.read_csv(ADULT_CACHE)   # re-read so dtypes match the cached path

    data = data.replace("?", np.nan).dropna()
    y = (data["class"].astype(str).str.strip() == ">50K").astype(int).rename("income_gt_50k")
    features = ["age", "workclass", "education-num", "marital-status", "occupation",
                "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country"]
    return data[features], y, "sex"
```

Register it in `load_dataset` (insert before the `else: raise`):

```python
    elif name == "adult":
        X, y, sensitive_col = _load_adult_frame()
        label = "Adult Census Income (sensitive attribute: sex)"
```

- [ ] **Step 4: Run tests to verify they pass** (first run downloads ~4 MB from OpenML)

Run: `.venv/bin/pytest new_experiment/tests/ -v`
Expected: 5 passed. Also verify the cache: `ls -la data/adult.csv` exists.

- [ ] **Step 5: Commit**

```bash
git add new_experiment/data_loading.py new_experiment/tests/test_data_loading.py
git commit -m "feat: add Adult Census Income dataset (sensitive attribute: sex)"
```

---

### Task 4: Multi-seed support in run_experiment.py

**Files:**
- Modify: `new_experiment/run_experiment.py` (`run_sweep`, `run_fairlearn_baselines`, `run_dataset`, `main`; add `parse_seeds`, add soft-surrogate columns and `Family`/`Seed` columns)
- Test: `new_experiment/tests/test_run_experiment_units.py` (new)

**Interfaces:**
- Consumes: `load_dataset(name, seed)` (Task 1), `soft_demographic_parity(probs, group_ids)` and `soft_generalized_entropy(probs, y_true)` from `losses.py`.
- Produces:
  - `parse_seeds(spec: str) -> list[int]` — `"0-2,7"` → `[0, 1, 2, 7]`.
  - Per-seed outputs: `new_experiment/MODELS/<ds>/seed<k>/<arch>_alpha_<a>_beta_<b>.pth`, `new_experiment/RESULTS/<ds>/seed<k>/sweep_results.csv` and `model_comparison.csv` following the Global-Constraints schema (including `Seed`, `Family`, `Soft DP`, `Soft GE`).
  - CLI: `--seeds "0-9"` (default `"42"`), `--dataset {compas,german,adult,both,all}` where `all` = all three.
  - Inline PNG plots are produced only when exactly one seed is requested (paper figures come from `analysis/`, Task 8+).

- [ ] **Step 1: Write the failing unit test**

`new_experiment/tests/test_run_experiment_units.py`:

```python
from run_experiment import parse_seeds


def test_parse_seeds_single():
    assert parse_seeds("42") == [42]


def test_parse_seeds_range_and_list():
    assert parse_seeds("0-2,7") == [0, 1, 2, 7]
    assert parse_seeds("0-9") == list(range(10))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest new_experiment/tests/test_run_experiment_units.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_seeds'`

- [ ] **Step 3: Implement the changes in `run_experiment.py`**

Add near the top (after the metric imports):

```python
from losses import soft_demographic_parity, soft_generalized_entropy


def parse_seeds(spec):
    """'0-2,7' -> [0, 1, 2, 7]."""
    seeds = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds
```

`run_sweep` — new signature `run_sweep(ds, arch_key, model_dir, args, seed)`; replace the hard-coded seeding and prediction block:

```python
        torch.manual_seed(seed)
        np.random.seed(seed)
```

and after training/loading:

```python
        probs_test = predict_proba(model, ds["X_test_t"])
        y_pred_test = (probs_test >= 0.5).astype(int)
        y_pred_val = (predict_proba(model, ds["X_val_t"]) >= 0.5).astype(int)

        probs_t = torch.tensor(probs_test, dtype=torch.float32)
        y_true_t = torch.tensor(np.asarray(ds["y_test"], dtype=np.float32))
        soft_dp = float(soft_demographic_parity(probs_t, ds["group_ids"]["test"]))
        soft_ge = float(soft_generalized_entropy(probs_t, y_true_t))
```

and extend the appended row dict with:

```python
            "Seed": seed,
            "Soft DP": round(soft_dp, 4),
            "Soft GE": round(soft_ge, 4),
```

`run_fairlearn_baselines` — new signature `run_fairlearn_baselines(ds, seed)`; the local `add` gains a stable family key and seed:

```python
    def add(name, y_pred):
        rows.append({"Model": name, "Family": name, "Seed": seed,
                     **{k: round(v, 4) for k, v in
                        evaluate_predictions(ds["y_test"], y_pred, s_te).items()}})
        print(f"  {name}: done")
```

and seed the stochastic pieces: `eg_dp.predict(X_te, random_state=seed)`, `eg_eo.predict(X_te, random_state=seed)`, `thr.predict(X_te, sensitive_features=s_te, random_state=seed)`, `RandomForestClassifier(n_estimators=300, random_state=seed)`.

`run_dataset` — new signature `run_dataset(name, args, seed)`:

```python
def run_dataset(name, args, seed):
    print(f"\n{'=' * 70}\nDataset: {name} | seed {seed}\n{'=' * 70}")
    ds = load_dataset(name, seed=seed)
    print(f"  {ds['label']} | input_dim={ds['input_dim']} | "
          f"train/val/test = {len(ds['y_train'])}/{len(ds['y_val'])}/{len(ds['y_test'])}")

    model_dir = HERE / "MODELS" / name / f"seed{seed}"
    results_dir = HERE / "RESULTS" / name / f"seed{seed}"
    plots_dir = results_dir / "PLOTS"
    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    sweep_df = pd.concat([run_sweep(ds, arch, model_dir, args, seed) for arch in ARCHITECTURES],
                         ignore_index=True)
    sweep_df.to_csv(results_dir / "sweep_results.csv", index=False)

    comparison_rows = []
    for arch_key, (arch_label, _) in ARCHITECTURES.items():
        arch_df = sweep_df[sweep_df["Arch"] == arch_key]
        baseline = arch_df[(arch_df["Alpha"] == 1) & (arch_df["Beta"] == 0)].iloc[0]
        comparison_rows.append({"Model": f"{arch_label} baseline (BCE only)",
                                "Family": f"{arch_label} baseline",
                                "Seed": seed, "Alpha": 1, "Beta": 0,
                                **{m: baseline[m] for m in METRICS}})
        best = pick_best_fair(arch_df)
        comparison_rows.append({
            "Model": f"Fair {arch_label} (α={best['Alpha']}, β={best['Beta']})",
            "Family": f"Fair {arch_label}",
            "Seed": seed, "Alpha": best["Alpha"], "Beta": best["Beta"],
            **{m: best[m] for m in METRICS}})

    print("  Running baselines (ExpGrad, ThresholdOptimizer, RandomForest)...")
    comparison_rows.extend(run_fairlearn_baselines(ds, seed))

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)
    print(comparison_df.to_string(index=False))

    if len(args.seed_list) == 1:      # ad-hoc runs still get inline plots
        plot_heatmaps(sweep_df, ds, plots_dir)
        plot_comparison(comparison_df, ds, plots_dir)
```

`main` — dataset choices and seed loop:

```python
    parser.add_argument("--dataset", choices=["compas", "german", "adult", "both", "all"],
                        default="compas")
    parser.add_argument("--seeds", default="42",
                        help="comma list / ranges, e.g. '0-9' or '0-2,7'")
    ...
    args = parser.parse_args()
    args.seed_list = parse_seeds(args.seeds)

    dataset_map = {"both": ["compas", "german"],
                   "all": ["compas", "german", "adult"]}
    datasets = dataset_map.get(args.dataset, [args.dataset])
    start = time.perf_counter()
    for name in datasets:
        for seed in args.seed_list:
            run_dataset(name, args, seed)
    print(f"\nTotal time: {time.perf_counter() - start:.1f}s")
```

- [ ] **Step 4: Run unit tests**

Run: `.venv/bin/pytest new_experiment/tests/ -v`
Expected: 7 passed

- [ ] **Step 5: Smoke-run German (small dataset) for one seed**

Run: `.venv/bin/python new_experiment/run_experiment.py --dataset german --seeds 0`
Expected: completes in a few minutes; verify:

```bash
head -2 new_experiment/RESULTS/german/seed0/sweep_results.csv
head -3 new_experiment/RESULTS/german/seed0/model_comparison.csv
```

Both exist; sweep header contains `Seed`, `Soft DP`, `Soft GE`; comparison header contains `Family`.

- [ ] **Step 6: Commit**

```bash
git add new_experiment/run_experiment.py new_experiment/tests/test_run_experiment_units.py
git commit -m "feat: multi-seed runs, per-seed result dirs, soft-surrogate columns"
```

---

### Task 5: Reweighing baseline (Kamiran & Calders)

**Files:**
- Create: `new_experiment/baselines.py`
- Modify: `new_experiment/run_experiment.py` (`run_fairlearn_baselines`)
- Test: `new_experiment/tests/test_baselines.py`

**Interfaces:**
- Produces: `kamiran_calders_weights(y, s) -> np.ndarray` (per-sample weights `P(s)·P(y)/P(s,y)`) and `reweighing_logreg(X_train, y_train, s_train, X_test, seed) -> np.ndarray` (0/1 predictions). Comparison rows gain family `Reweighing LogReg (Kamiran-Calders)`.

- [ ] **Step 1: Write the failing test**

`new_experiment/tests/test_baselines.py`:

```python
import numpy as np
from baselines import kamiran_calders_weights


def test_weights_equalize_weighted_base_rates():
    # group a: 3/4 positive, group b: 1/4 positive, overall 1/2.
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    s = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    w = kamiran_calders_weights(y, s)
    for g in ("a", "b"):
        m = s == g
        weighted_rate = (w[m] * y[m]).sum() / w[m].sum()
        assert abs(weighted_rate - 0.5) < 1e-9


def test_weights_are_uniform_when_independent():
    y = np.array([1, 0, 1, 0])
    s = np.array(["a", "a", "b", "b"])
    w = kamiran_calders_weights(y, s)
    assert np.allclose(w, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest new_experiment/tests/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'baselines'`

- [ ] **Step 3: Implement**

`new_experiment/baselines.py`:

```python
"""Extra fairness baselines beyond the fairlearn ones in run_experiment.py.

Reweighing (Kamiran & Calders 2012): pre-processing. Each training sample
gets weight w(s, y) = P(s) * P(y) / P(s, y), which makes the label
statistically independent of the sensitive attribute in the weighted
training distribution. Implemented directly (the aif360 version requires
its BinaryLabelDataset wrapper; the formula is four lines).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SkLogisticRegression


def kamiran_calders_weights(y, s):
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(s)})
    p_s = df["s"].value_counts(normalize=True)
    p_y = df["y"].value_counts(normalize=True)
    p_sy = df.value_counts(["s", "y"], normalize=True)
    return np.array([p_s[r.s] * p_y[r.y] / p_sy[(r.s, r.y)]
                     for r in df.itertuples(index=False)])


def reweighing_logreg(X_train, y_train, s_train, X_test, seed):
    w = kamiran_calders_weights(y_train, s_train)
    clf = SkLogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train, sample_weight=w)
    return clf.predict(X_test)
```

In `run_experiment.py`, import it (`from baselines import reweighing_logreg`) and add to `run_fairlearn_baselines`, before the Random Forest block:

```python
    add("Reweighing LogReg (Kamiran-Calders)",
        reweighing_logreg(X_tr, y_tr, s_tr, X_te, seed))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest new_experiment/tests/ -v`
Expected: 9 passed

- [ ] **Step 5: Verify integration on German**

Run: `.venv/bin/python new_experiment/run_experiment.py --dataset german --seeds 0`
Expected: comparison table now includes a `Reweighing LogReg (Kamiran-Calders)` row (checkpoints are cached, so only baselines rerun — fast).

- [ ] **Step 6: Commit**

```bash
git add new_experiment/baselines.py new_experiment/run_experiment.py new_experiment/tests/test_baselines.py
git commit -m "feat: add Kamiran-Calders reweighing baseline"
```

---

### Task 6: Full 10-seed sweep (long-running, start in background)

**Files:** none created by hand — this task produces `RESULTS/{compas,german,adult}/seed{0..9}/{sweep_results,model_comparison}.csv`.

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: 30 sweep CSVs + 30 comparison CSVs, the raw material for all of `analysis/`.

- [ ] **Step 1: Launch the full run in the background**

```bash
nohup .venv/bin/python new_experiment/run_experiment.py --dataset all --seeds 0-9 \
    > new_experiment/full_run.log 2>&1 &
```

Expected duration: roughly 1–3 hours (Adult is the slow one: 98 models × 10 seeds on ~27k training rows, plus ExpGrad). Tasks 7–11 can be developed while this runs — their unit tests use synthetic data; only their final verification steps need the real CSVs.

- [ ] **Step 2: Verify completion**

```bash
tail -5 new_experiment/full_run.log        # expect "Total time: ..."
ls new_experiment/RESULTS/*/seed*/sweep_results.csv | wc -l      # expect 30
ls new_experiment/RESULTS/*/seed*/model_comparison.csv | wc -l   # expect 30
```

- [ ] **Step 3: Commit the results CSVs** (small text files; checkpoints stay untracked)

```bash
git add new_experiment/RESULTS/*/seed*/*.csv
git commit -m "data: full 10-seed sweep results for compas, german, adult"
```

---

### Task 7: analysis/stats.py — aggregation, Wilcoxon, Holm

**Files:**
- Create: `new_experiment/analysis/__init__.py` (empty), `new_experiment/analysis/stats.py`
- Test: `new_experiment/tests/test_stats.py`

**Interfaces:**
- Produces (all consumed by Tasks 8–11):
  - `DPD2 = "DPD (Largest 2 Groups)"`, `METRICS` (the 6-metric list).
  - `load_seed_csvs(dataset: str, kind: str, results_root: Path = RESULTS) -> pd.DataFrame` — concatenation of `RESULTS/<dataset>/seed*/<kind>.csv` (`kind` ∈ {`"sweep_results"`, `"model_comparison"`}).
  - `holm_correction(pvals: np.ndarray) -> np.ndarray`.
  - `significance_vs_baseline(comp_df, family, baseline_family, metrics=METRICS) -> pd.DataFrame` with columns `Metric, p, p_holm` (paired Wilcoxon over seeds).

- [ ] **Step 1: Write the failing tests**

`new_experiment/tests/test_stats.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest new_experiment/tests/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis'`

- [ ] **Step 3: Implement**

`new_experiment/analysis/stats.py`:

```python
"""Seed aggregation and significance testing for the paper pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS = Path(__file__).resolve().parent.parent / "RESULTS"

DPD2 = "DPD (Largest 2 Groups)"
METRICS = ["Accuracy", "Demographic Parity Difference", DPD2,
           "Equalized Odds Difference", "Theil Index", "Gini Coefficient"]


def load_seed_csvs(dataset, kind, results_root=RESULTS):
    """Concat RESULTS/<dataset>/seed*/<kind>.csv into one frame."""
    frames = []
    for f in sorted((results_root / dataset).glob(f"seed*/{kind}.csv")):
        df = pd.read_csv(f)
        if "Seed" not in df.columns:
            df["Seed"] = int(f.parent.name.removeprefix("seed"))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No {kind}.csv under {results_root / dataset}/seed*/")
    return pd.concat(frames, ignore_index=True)


def holm_correction(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def significance_vs_baseline(comp_df, family, baseline_family, metrics=METRICS):
    """Paired Wilcoxon signed-rank across seeds, Holm-corrected over `metrics`."""
    a = comp_df[comp_df["Family"] == family].sort_values("Seed")
    b = comp_df[comp_df["Family"] == baseline_family].sort_values("Seed")
    assert list(a["Seed"]) == list(b["Seed"]), "seed sets must match for a paired test"
    pvals = []
    for m in metrics:
        diff = a[m].to_numpy() - b[m].to_numpy()
        pvals.append(1.0 if np.allclose(diff, 0) else wilcoxon(diff).pvalue)
    return pd.DataFrame({"Metric": metrics, "p": pvals,
                         "p_holm": holm_correction(pvals)})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest new_experiment/tests/ -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add new_experiment/analysis new_experiment/tests/test_stats.py
git commit -m "feat: seed aggregation + Wilcoxon/Holm significance module"
```

---

### Task 8: Pareto frontier figure (Fig 1)

**Files:**
- Create: `new_experiment/analysis/figures.py`
- Test: `new_experiment/tests/test_figures.py`

**Interfaces:**
- Consumes: `load_seed_csvs`, `DPD2`, `METRICS` from `analysis.stats`.
- Produces:
  - `pareto_mask(acc: np.ndarray, dpd: np.ndarray) -> np.ndarray[bool]` — True where no other point has ≥ accuracy and ≤ DPD (one strict).
  - `plot_pareto(dataset: str, out_dir: Path, results_root=RESULTS) -> Path` — writes `fig1_pareto.png`.

- [ ] **Step 1: Write the failing test**

`new_experiment/tests/test_figures.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest new_experiment/tests/test_figures.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`

- [ ] **Step 3: Implement**

`new_experiment/analysis/figures.py`:

```python
"""Paper figures and tables. Every function writes into out_dir and returns the path."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.stats import (DPD2, METRICS, RESULTS, load_seed_csvs,
                            significance_vs_baseline)

ARCH_LABELS = {"logreg": "Logistic Regression", "mlp": "MLP (64-64)"}
ARCH_COLORS = {"logreg": "tab:blue", "mlp": "tab:orange"}
BASELINE_FAMILIES = [
    "ExpGrad LogReg (Demographic Parity)",
    "ExpGrad LogReg (Equalized Odds)",
    "ThresholdOptimizer LogReg (Demographic Parity)",
    "Reweighing LogReg (Kamiran-Calders)",
    "Random Forest (no fairness)",
]


def pareto_mask(acc, dpd):
    """True where no other point has >= accuracy AND <= DPD (one strict)."""
    acc, dpd = np.asarray(acc), np.asarray(dpd)
    mask = np.ones(len(acc), dtype=bool)
    for i in range(len(acc)):
        dominated = ((acc >= acc[i]) & (dpd <= dpd[i])
                     & ((acc > acc[i]) | (dpd < dpd[i])))
        if dominated.any():
            mask[i] = False
    return mask


def plot_pareto(dataset, out_dir, results_root=RESULTS):
    sweep = load_seed_csvs(dataset, "sweep_results", results_root)
    comp = load_seed_csvs(dataset, "model_comparison", results_root)
    mean_sweep = (sweep.groupby(["Arch", "Alpha", "Beta"], as_index=False)
                  [["Accuracy", DPD2]].mean())

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for arch, g in mean_sweep.groupby("Arch"):
        ax.scatter(g[DPD2], g["Accuracy"], s=16, alpha=0.35,
                   color=ARCH_COLORS[arch], label=f"{ARCH_LABELS[arch]} sweep (seed mean)")
        front = g[pareto_mask(g["Accuracy"].to_numpy(), g[DPD2].to_numpy())].sort_values(DPD2)
        ax.plot(front[DPD2], front["Accuracy"], "-o", lw=2, ms=4,
                color=ARCH_COLORS[arch], label=f"{ARCH_LABELS[arch]} Pareto front")

    agg = comp.groupby("Family")[["Accuracy", DPD2]].agg(["mean", "std"])
    markers = ["D", "s", "^", "v", "P"]
    for fam, mk in zip(BASELINE_FAMILIES, markers):
        if fam not in agg.index:
            continue
        ax.errorbar(agg.loc[fam, (DPD2, "mean")], agg.loc[fam, ("Accuracy", "mean")],
                    xerr=agg.loc[fam, (DPD2, "std")], yerr=agg.loc[fam, ("Accuracy", "std")],
                    fmt=mk, ms=7, capsize=3, label=fam)

    ax.set_xlabel(f"{DPD2}  (lower = fairer)")
    ax.set_ylabel("Accuracy (test)")
    ax.set_title(f"Accuracy–fairness Pareto frontier — {dataset} (mean over seeds)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out = Path(out_dir) / "fig1_pareto.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest new_experiment/tests/ -v`
Expected: 16 passed

- [ ] **Step 5: Verify against real data** (requires Task 6 output for at least German)

```bash
.venv/bin/python -c "
from pathlib import Path
import sys; sys.path[:0] = ['new_experiment', '.']
from analysis.figures import plot_pareto
out = Path('new_experiment/RESULTS/paper/german'); out.mkdir(parents=True, exist_ok=True)
print(plot_pareto('german', out))"
```

Expected: prints the PNG path; open it and confirm a frontier line plus labeled baseline points with error bars.

- [ ] **Step 6: Commit**

```bash
git add new_experiment/analysis/figures.py new_experiment/tests/test_figures.py
git commit -m "feat: Pareto frontier figure (Fig 1)"
```

---

### Task 9: β cross-effect + DPD-vs-Theil figures (Fig 2, the core contribution)

**Files:**
- Modify: `new_experiment/analysis/figures.py` (append two functions)
- Test: `new_experiment/tests/test_figures.py` (append)

**Interfaces:**
- Produces: `plot_beta_cross_effect(dataset, out_dir, results_root=RESULTS, alphas=(0.25, 0.5, 0.75)) -> Path` (`fig2_beta_cross_effect.png`) and `plot_dpd_vs_theil(dataset, out_dir, results_root=RESULTS) -> Path` (`fig2b_dpd_vs_theil.png`).

- [ ] **Step 1: Write the failing smoke test** (synthetic seed dirs; append to `test_figures.py`)

```python
import itertools

import pandas as pd

from analysis.stats import DPD2, METRICS


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest new_experiment/tests/test_figures.py -v`
Expected: 2 new tests FAIL with `ImportError`

- [ ] **Step 3: Implement** (append to `figures.py`)

```python
def plot_beta_cross_effect(dataset, out_dir, results_root=RESULTS, alphas=(0.25, 0.5, 0.75)):
    """The paper's core figure: group (DPD) and individual (Theil) fairness
    as β moves from individual-only (0) to group-only (1), per arch and α."""
    sweep = load_seed_csvs(dataset, "sweep_results", results_root)
    fig, axes = plt.subplots(2, len(alphas), figsize=(4.2 * len(alphas), 7.2),
                             sharex=True, squeeze=False)
    series = [(DPD2, "tab:red", "group fairness (DPD, largest 2)"),
              ("Theil Index", "tab:purple", "individual fairness (Theil)")]
    for r, arch in enumerate(["logreg", "mlp"]):
        for c, a in enumerate(alphas):
            ax = axes[r][c]
            g = sweep[(sweep["Arch"] == arch) & (sweep["Alpha"] == a)]
            st = g.groupby("Beta")[[DPD2, "Theil Index"]].agg(["mean", "std"])
            for metric, color, label in series:
                m = st[(metric, "mean")]
                s = st[(metric, "std")].fillna(0)
                ax.plot(st.index, m, "-o", ms=4, color=color,
                        label=label if (r == 0 and c == 0) else None)
                ax.fill_between(st.index, m - s, m + s, color=color, alpha=0.2)
            ax.set_title(f"{ARCH_LABELS[arch]}, α={a}", fontsize=10)
            if r == 1:
                ax.set_xlabel("β  (0 = individual only, 1 = group only)")
            if c == 0:
                ax.set_ylabel("metric value (test)")
    fig.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Group vs. individual fairness as β varies — {dataset} "
                 f"(mean ± std over seeds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(out_dir) / "fig2_beta_cross_effect.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_dpd_vs_theil(dataset, out_dir, results_root=RESULTS):
    sweep = load_seed_csvs(dataset, "sweep_results", results_root)
    fair = sweep[sweep["Alpha"] < 1]
    mean = fair.groupby(["Arch", "Alpha", "Beta"], as_index=False)[[DPD2, "Theil Index"]].mean()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(mean[DPD2], mean["Theil Index"], c=mean["Beta"],
                    cmap="viridis", s=32)
    fig.colorbar(sc, label="β (group-fairness weight)")
    ax.set_xlabel(DPD2)
    ax.set_ylabel("Theil Index")
    ax.set_title(f"Group vs. individual fairness trade-off — {dataset} (seed means)")
    fig.tight_layout()
    out = Path(out_dir) / "fig2b_dpd_vs_theil.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest new_experiment/tests/ -v`
Expected: 18 passed

- [ ] **Step 5: Commit**

```bash
git add new_experiment/analysis/figures.py new_experiment/tests/test_figures.py
git commit -m "feat: beta cross-effect and DPD-vs-Theil figures (Fig 2)"
```

---

### Task 10: Fair-vs-baseline bars with significance (Fig 3) + Tables 1–2

**Files:**
- Modify: `new_experiment/analysis/figures.py` (append `stars`, `plot_fair_vs_baseline`, `master_table`, `ablation_table`)
- Test: `new_experiment/tests/test_figures.py` (append)

**Interfaces:**
- Consumes: `significance_vs_baseline` (Task 7). (The leaky-features COMPAS numbers archived in Task 2 at `RESULTS/compas/legacy_leaky/` are referenced in the paper text, not drawn in Fig 3 — they are single-seed and not comparable to the 10-seed bars.)
- Produces: `stars(p: float) -> str` (`***`/`**`/`*`/`ns`), `plot_fair_vs_baseline(dataset, out_dir, results_root=RESULTS) -> Path` (`fig3_fair_vs_baseline.png`), `master_table(dataset, out_dir, results_root=RESULTS) -> pd.DataFrame` (`table1_master_comparison.csv/.tex`), `ablation_table(dataset, out_dir, results_root=RESULTS) -> pd.DataFrame` (`table2_ablation.csv/.tex`).

- [ ] **Step 1: Write the failing tests** (append to `test_figures.py`)

```python
def test_stars_thresholds():
    from analysis.figures import stars
    assert stars(0.0005) == "***"
    assert stars(0.005) == "**"
    assert stars(0.03) == "*"
    assert stars(0.2) == "ns"


def _fake_comparison_dir(tmp_path, n_seeds=6):
    import numpy as np
    rng = np.random.default_rng(1)
    fams = ["Logistic Regression baseline", "Fair Logistic Regression",
            "MLP (64-64) baseline", "Fair MLP (64-64)",
            "Reweighing LogReg (Kamiran-Calders)"]
    for seed in range(n_seeds):
        rows = []
        for fam in fams:
            row = {"Model": fam, "Family": fam, "Seed": seed,
                   "Alpha": 0.5, "Beta": 0.5}
            for m in METRICS:
                shift = -0.04 if fam.startswith("Fair") else 0.0
                row[m] = 0.5 + shift + rng.normal(0, 0.004)
            rows.append(row)
        d = tmp_path / "german" / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(d / "model_comparison.csv", index=False)
    return tmp_path


def test_fig3_and_tables_write_outputs(tmp_path):
    from analysis.figures import ablation_table, master_table, plot_fair_vs_baseline
    root = _fake_comparison_dir(tmp_path)
    _fake_sweep_dir(tmp_path, n_seeds=6)
    assert plot_fair_vs_baseline("german", tmp_path, results_root=root).exists()
    t1 = master_table("german", tmp_path, results_root=root)
    assert "±" in t1[METRICS[0]].iloc[0]
    t2 = ablation_table("german", tmp_path, results_root=root)
    assert len(t2) == 8            # 2 archs x 4 variants
    assert (tmp_path / "table1_master_comparison.tex").exists()
    assert (tmp_path / "table2_ablation.tex").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest new_experiment/tests/test_figures.py -v`
Expected: new tests FAIL with `ImportError`

- [ ] **Step 3: Implement** (append to `figures.py`)

```python
STAR_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
FIG3_METRICS = ["Accuracy", DPD2, "Equalized Odds Difference", "Theil Index"]
PAIRS = [("Logistic Regression baseline", "Fair Logistic Regression"),
         ("MLP (64-64) baseline", "Fair MLP (64-64)")]


def stars(p):
    for thr, s in STAR_LEVELS:
        if p < thr:
            return s
    return "ns"


def plot_fair_vs_baseline(dataset, out_dir, results_root=RESULTS):
    comp = load_seed_csvs(dataset, "model_comparison", results_root)
    # one Holm family per (baseline, fair) pair, across the four plotted metrics
    sig = {ff: significance_vs_baseline(comp, ff, bf, metrics=FIG3_METRICS)
                 .set_index("Metric")["p_holm"]
           for bf, ff in PAIRS}

    fig, axes = plt.subplots(1, len(FIG3_METRICS), figsize=(4.4 * len(FIG3_METRICS), 4.6))
    x = np.arange(len(PAIRS))
    width = 0.38
    for ax, metric in zip(axes, FIG3_METRICS):
        for k, kind in enumerate(["baseline", "fair"]):
            fams = [p[k] for p in PAIRS]
            mean = [comp.loc[comp["Family"] == f, metric].mean() for f in fams]
            std = [comp.loc[comp["Family"] == f, metric].std() for f in fams]
            ax.bar(x + (k - 0.5) * width, mean, width, yerr=std, capsize=3,
                   label=kind if metric == FIG3_METRICS[0] else None)
        for i, (bf, ff) in enumerate(PAIRS):
            top = comp.loc[comp["Family"].isin([bf, ff]), metric].max()
            ax.text(i, top * 1.04, stars(sig[ff][metric]), ha="center", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(["LogReg", "MLP"])
        ax.set_title(metric, fontsize=10)
    fig.legend(loc="upper right")
    fig.suptitle(f"Fairness loss vs. baseline — {dataset} "
                 f"(mean ± std over seeds; Wilcoxon, Holm-corrected)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = Path(out_dir) / "fig3_fair_vs_baseline.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def master_table(dataset, out_dir, results_root=RESULTS):
    comp = load_seed_csvs(dataset, "model_comparison", results_root)
    rows = []
    for fam, g in comp.groupby("Family", sort=False):
        row = {"Model": fam, "Seeds": len(g)}
        for m in METRICS:
            row[m] = f"{g[m].mean():.3f} ± {g[m].std():.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "table1_master_comparison.csv", index=False)
    (Path(out_dir) / "table1_master_comparison.tex").write_text(df.to_latex(index=False))
    return df


def ablation_table(dataset, out_dir, results_root=RESULTS):
    """No-fairness / SoftGE-only / SoftDP-only / blend, at the seed-mean-best α.
    Best (α, β) picked on validation seed means (same utopia rule as training)."""
    sweep = load_seed_csvs(dataset, "sweep_results", results_root)
    rows = []
    for arch in ["logreg", "mlp"]:
        g = sweep[sweep["Arch"] == arch]
        mean = g.groupby(["Alpha", "Beta"], as_index=False).mean(numeric_only=True)
        fair = mean[(mean["Alpha"] < 1)
                    & mean["Val Positive Rate"].between(0.05, 0.95)].copy()
        fair["dist"] = np.sqrt((1 - fair["Val Accuracy"]) ** 2
                               + fair["Val DPD (Largest 2 Groups)"] ** 2)
        best = fair.sort_values("dist").iloc[0]
        a_star, b_star = best["Alpha"], best["Beta"]
        variants = [("No fairness (α=1)", 1, 0),
                    (f"SoftGE only (α={a_star}, β=0)", a_star, 0),
                    (f"SoftDP only (α={a_star}, β=1)", a_star, 1),
                    (f"Blend (α={a_star}, β={b_star})", a_star, b_star)]
        for name, a, b in variants:
            sel = g[(g["Alpha"] == a) & (g["Beta"] == b)]
            row = {"Arch": ARCH_LABELS[arch], "Variant": name}
            for m in METRICS:
                row[m] = f"{sel[m].mean():.3f} ± {sel[m].std():.3f}"
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "table2_ablation.csv", index=False)
    (Path(out_dir) / "table2_ablation.tex").write_text(df.to_latex(index=False))
    return df
```

Note: the synthetic sweep in `_fake_sweep_dir` includes all `METRICS` columns plus the `Val *` columns, so `ablation_table` works against it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest new_experiment/tests/ -v`
Expected: 20 passed

- [ ] **Step 5: Commit**

```bash
git add new_experiment/analysis/figures.py new_experiment/tests/test_figures.py
git commit -m "feat: fair-vs-baseline significance figure (Fig 3) + master/ablation tables"
```

---

### Task 11: Supplementary assets + run_analysis orchestrator

**Files:**
- Modify: `new_experiment/analysis/figures.py` (append `seed_mean_heatmaps`, `surrogate_scatter`)
- Create: `new_experiment/analysis/run_analysis.py`
- Test: `new_experiment/tests/test_figures.py` (append)

**Interfaces:**
- Produces: `seed_mean_heatmaps(dataset, out_dir, results_root=RESULTS) -> list[Path]`, `surrogate_scatter(dataset, out_dir, results_root=RESULTS) -> Path` (`supp_surrogate_validation.png`), and the CLI `python new_experiment/analysis/run_analysis.py --dataset all` which writes every asset for each dataset into `new_experiment/RESULTS/paper/<dataset>/`.

- [ ] **Step 1: Write the failing test** (append to `test_figures.py`)

```python
def test_supplementary_assets_write(tmp_path):
    from analysis.figures import seed_mean_heatmaps, surrogate_scatter
    root = _fake_sweep_dir(tmp_path)
    paths = seed_mean_heatmaps("german", tmp_path, results_root=root)
    assert len(paths) == 6 and all(p.exists() for p in paths)   # 2 archs x 3 metrics
    assert surrogate_scatter("german", tmp_path, results_root=root).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest new_experiment/tests/test_figures.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement** (append to `figures.py`)

```python
HEATMAP_METRICS = ["Accuracy", DPD2, "Theil Index"]


def seed_mean_heatmaps(dataset, out_dir, results_root=RESULTS):
    sweep = load_seed_csvs(dataset, "sweep_results", results_root)
    paths = []
    for arch in ["logreg", "mlp"]:
        g = sweep[sweep["Arch"] == arch]
        for metric in HEATMAP_METRICS:
            pv = g.pivot_table(index="Alpha", columns="Beta", values=metric, aggfunc="mean")
            plt.figure(figsize=(8, 6))
            sns.heatmap(pv, annot=True, fmt=".3f",
                        cmap="viridis" if metric == "Accuracy" else "coolwarm",
                        cbar_kws={"label": f"{metric} (test, seed mean)"})
            plt.title(f"{metric} (mean over seeds) — {ARCH_LABELS[arch]} — {dataset}")
            plt.xlabel("β")
            plt.ylabel("α")
            plt.tight_layout()
            out = Path(out_dir) / f"heatmap_{arch}_{metric.replace(' ', '_')}.png"
            plt.savefig(out, dpi=150)
            plt.close()
            paths.append(out)
    return paths


def surrogate_scatter(dataset, out_dir, results_root=RESULTS):
    """Does the differentiable training surrogate track the hard eval metric?"""
    sweep = load_seed_csvs(dataset, "sweep_results", results_root)
    pairs = [("Soft DP", DPD2), ("Soft GE", "Theil Index")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (soft, hard) in zip(axes, pairs):
        ax.scatter(sweep[soft], sweep[hard], s=10, alpha=0.35)
        r = np.corrcoef(sweep[soft], sweep[hard])[0, 1]
        ax.set_xlabel(f"{soft} (training surrogate)")
        ax.set_ylabel(f"{hard} (evaluation metric)")
        ax.set_title(f"Pearson r = {r:.3f}")
    fig.suptitle(f"Surrogate vs. hard metric — {dataset} (all sweep points, all seeds)")
    fig.tight_layout()
    out = Path(out_dir) / "supp_surrogate_validation.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
```

`new_experiment/analysis/run_analysis.py`:

```python
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
from analysis.stats import RESULTS


def main():
    parser = argparse.ArgumentParser(description="Build paper figures and tables")
    parser.add_argument("--dataset", choices=["compas", "german", "adult", "all"],
                        default="all")
    args = parser.parse_args()
    datasets = (["compas", "german", "adult"] if args.dataset == "all"
                else [args.dataset])

    for ds in datasets:
        out = RESULTS / "paper" / ds
        out.mkdir(parents=True, exist_ok=True)
        print(f"== {ds} -> {out}")
        for fn in (plot_pareto, plot_beta_cross_effect, plot_dpd_vs_theil,
                   plot_fair_vs_baseline, master_table, ablation_table,
                   seed_mean_heatmaps, surrogate_scatter):
            result = fn(ds, out)
            print(f"   {fn.__name__}: {result if isinstance(result, Path) else 'ok'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest new_experiment/tests/ -v`
Expected: 21 passed

- [ ] **Step 5: Generate all real paper assets** (requires Task 6 complete)

Run: `.venv/bin/python new_experiment/analysis/run_analysis.py --dataset all`
Expected: prints per-dataset asset paths; verify:

```bash
ls new_experiment/RESULTS/paper/compas new_experiment/RESULTS/paper/german new_experiment/RESULTS/paper/adult
```

Each contains `fig1_pareto.png`, `fig2_beta_cross_effect.png`, `fig2b_dpd_vs_theil.png`, `fig3_fair_vs_baseline.png`, `table1_master_comparison.{csv,tex}`, `table2_ablation.{csv,tex}`, 6 heatmaps, `supp_surrogate_validation.png`. Eyeball each figure for sanity (frontiers slope the right way; stars appear; heatmaps not flat across β — the loss is differentiable now, so β must matter).

- [ ] **Step 6: Commit**

```bash
git add new_experiment/analysis new_experiment/tests/test_figures.py new_experiment/RESULTS/paper
git commit -m "feat: supplementary assets + run_analysis orchestrator; paper assets generated"
```

---

### Task 12: Legacy archival + documentation

**Files:**
- Move: `Compas.py`, `German.py`, `Loss.py`, `Models.py` → `legacy/`
- Create: `legacy/README.md`
- Modify: `CLAUDE.md`, `new_experiment/README.md`

**Interfaces:**
- Consumes: nothing new. `GroupFairness.py` and `IndividualFairness.py` MUST stay at repo root (imported by `run_experiment.py`).

- [ ] **Step 1: Move legacy scripts**

```bash
mkdir -p legacy
git mv Compas.py German.py Loss.py Models.py legacy/
```

`legacy/README.md`:

```markdown
# Legacy pipeline (frozen)

The original experiment scripts, kept for historical reference only. They
are NOT runnable from this directory (imports assume the repo root) and
contain two known training bugs fixed in `new_experiment/`:

1. Fairness terms computed via `.round().detach().numpy()` — no gradients,
   so only BCE trained the model and β had no effect.
2. Double sigmoid (in `forward` and again in the loss).

The COMPAS variant here also uses the leaking `duration = end - start`
feature (corr −0.78 with the label). The canonical pipeline for the paper
is `new_experiment/`; see `new_experiment/README.md`.
```

- [ ] **Step 2: Verify nothing at the root imports the moved files**

```bash
grep -rn "from Loss import\|from Models import\|import Compas\|import German" \
    --include="*.py" . | grep -v legacy/ | grep -v .venv
.venv/bin/pytest new_experiment/tests/ -v
```

Expected: no matches outside `legacy/`; all tests still pass.

- [ ] **Step 3: Update CLAUDE.md**

Replace the Commands and Architecture sections so they describe the canonical pipeline:

```markdown
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
`RESULTS/<dataset>/seed<k>/`, paper assets via `analysis/` into
`RESULTS/paper/<dataset>/`.

Evaluation metrics are imported from the repo root: `GroupFairness.py`,
`IndividualFairness.py` — keep them there.

`legacy/` holds the frozen original scripts (known training bugs — see
`legacy/README.md`). Do not extend them.

Design spec: `docs/superpowers/specs/2026-07-11-fairml-paper-design.md`.
```

- [ ] **Step 4: Update `new_experiment/README.md`** — replace the "How to run" and "Outputs" sections with the multi-seed usage:

```markdown
## How to run

From the repo root (`FairML/`):

```bash
# Full paper run: all three datasets, seeds 0-9 (~1-3 h)
.venv/bin/python new_experiment/run_experiment.py --dataset all --seeds 0-9

# Single dataset / ad-hoc seed (also produces inline plots)
.venv/bin/python new_experiment/run_experiment.py --dataset german --seeds 42

# Paper figures + tables from the per-seed CSVs
.venv/bin/python new_experiment/analysis/run_analysis.py --dataset all
```

## Outputs

```
MODELS/<dataset>/seed<k>/            cached .pth checkpoints
RESULTS/<dataset>/seed<k>/           sweep_results.csv, model_comparison.csv
RESULTS/paper/<dataset>/             fig1_pareto, fig2_beta_cross_effect,
                                     fig2b_dpd_vs_theil, fig3_fair_vs_baseline,
                                     table1/table2 (csv + tex), heatmaps,
                                     supp_surrogate_validation
```
```

- [ ] **Step 5: Commit**

```bash
git add -A legacy CLAUDE.md new_experiment/README.md
git commit -m "chore: archive legacy pipeline, document canonical paper pipeline"
```

---

### Task 13 (OPTIONAL — only if time permits, per spec): XGBoost accuracy reference

**Files:**
- Modify: `new_experiment/baselines.py`, `new_experiment/run_experiment.py`
- Test: `new_experiment/tests/test_baselines.py` (append)

- [ ] **Step 1: Install and pin**

```bash
.venv/bin/pip install xgboost
```

Append the installed version to `requirements.txt` (e.g. `xgboost==2.1.4`).

- [ ] **Step 2: Add wrapper to `baselines.py`**

```python
def xgboost_reference(X_train, y_train, X_test, seed):
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        random_state=seed, eval_metric="logloss")
    clf.fit(X_train, y_train)
    return clf.predict(X_test)
```

Test (append to `test_baselines.py`):

```python
def test_xgboost_reference_predicts_binary():
    import numpy as np
    from baselines import xgboost_reference
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int)
    pred = xgboost_reference(X[:150], y[:150], X[150:], seed=0)
    assert set(np.unique(pred)) <= {0, 1}
    assert (pred == y[150:]).mean() > 0.8
```

- [ ] **Step 3: Wire into `run_fairlearn_baselines`** (after the Random Forest block)

```python
    add("XGBoost (no fairness)", xgboost_reference(X_tr, y_tr, X_te, seed))
```

(also extend the import line: `from baselines import reweighing_logreg, xgboost_reference`).

If added AFTER the full Task-6 run, re-run only the baselines is not supported — either accept XGBoost being absent from seeds already computed, or re-run `--dataset all --seeds 0-9` (checkpoints are cached, so only baselines recompute).

- [ ] **Step 4: Run tests, then commit**

Run: `.venv/bin/pytest new_experiment/tests/ -v` — expected: all pass.

```bash
git add new_experiment/baselines.py new_experiment/run_experiment.py \
        new_experiment/tests/test_baselines.py requirements.txt
git commit -m "feat: optional XGBoost accuracy reference"
```
