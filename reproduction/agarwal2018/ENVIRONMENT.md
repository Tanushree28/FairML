# Environment

- Git commit: `14ec9f5b5c5f7fa04df873632cf9b64d86e360fb`
- Git dirty status before protected-path run: `M new_experiment/analysis/figures.py
 M new_experiment/analysis/run_analysis.py
 M new_experiment/dashboard/app.py
 M new_experiment/data_loading.py
 M new_experiment/run_experiment.py
 M new_experiment/tests/test_baselines.py
 M new_experiment/tests/test_data_loading.py
 M requirements.txt
?? data/acs_employment_ca2018.csv
?? data/acs_pubcov_ca2018.csv
?? data/folktables/
?? data/taiwan_credit.csv
?? new_experiment/MODELS/acs_employment/
?? new_experiment/MODELS/acs_pubcov/
?? new_experiment/MODELS/taiwan/
?? new_experiment/MODELS/taiwan_edu/
?? new_experiment/RESULTS/acs_employment/
?? new_experiment/RESULTS/acs_pubcov/
?? new_experiment/RESULTS/paper/acs_employment/
?? new_experiment/RESULTS/paper/acs_pubcov/
?? new_experiment/RESULTS/paper/cross_dataset/
?? new_experiment/RESULTS/paper/taiwan/
?? new_experiment/RESULTS/taiwan/
?? new_experiment/RESULTS/taiwan_edu/
?? new_experiment/analysis/cross_dataset.py
?? new_experiment/tests/test_cross_dataset.py`
- Git dirty status after protected-path run: `M new_experiment/analysis/figures.py
 M new_experiment/analysis/run_analysis.py
 M new_experiment/dashboard/app.py
 M new_experiment/data_loading.py
 M new_experiment/run_experiment.py
 M new_experiment/tests/test_baselines.py
 M new_experiment/tests/test_data_loading.py
 M requirements.txt
?? data/acs_employment_ca2018.csv
?? data/acs_pubcov_ca2018.csv
?? data/folktables/
?? data/taiwan_credit.csv
?? new_experiment/MODELS/acs_employment/
?? new_experiment/MODELS/acs_pubcov/
?? new_experiment/MODELS/taiwan/
?? new_experiment/MODELS/taiwan_edu/
?? new_experiment/RESULTS/acs_employment/
?? new_experiment/RESULTS/acs_pubcov/
?? new_experiment/RESULTS/paper/acs_employment/
?? new_experiment/RESULTS/paper/acs_pubcov/
?? new_experiment/RESULTS/paper/cross_dataset/
?? new_experiment/RESULTS/paper/taiwan/
?? new_experiment/RESULTS/taiwan/
?? new_experiment/RESULTS/taiwan_edu/
?? new_experiment/analysis/cross_dataset.py
?? new_experiment/tests/test_cross_dataset.py`
- Historical Fairlearn commit: `aeaa92d536406b54354a6c2db0d0ac5d14897782`
- Modern Fairlearn package: `fairlearn==0.11.0`; upstream tag target recorded during planning: `ea33211bd9c0c8d2102bdd1e4f5cc17c37a97796`
- Historical code path: `reproduction/agarwal2018/third_party/fairlearn_v0_2_0`
- Adult data path: `data/adult.csv`
- Adult data SHA256: `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc`

## Actual run environment

- python: `3.13.9`
- numpy: `2.1.3`
- pandas: `2.2.3`
- scipy: `1.17.1`
- sklearn: `1.5.2`
- fairlearn: `0.11.0`
- matplotlib: `3.9.2`

## Historical compatibility finding

The public Fairlearn v0.2.0 setup.py lists unversioned `numpy`, `scipy`, and `pandas` dependencies and no scikit-learn dependency, and the paper does not publish a lockfile. The current host uses Python 3.13.9, where historically plausible 2018 packages such as pandas 0.23.x and scikit-learn 0.19.x are not installable. The historical algorithm was therefore run from the pinned source with compatibility shims outside the algorithm for removed NumPy/pandas APIs (`np.PINF`, `Index.contains`, `Series.sum(level=...)`, and old `DataFrameGroupBy.mean` behavior).

See `requirements-historical-approx.txt` for a documented approximate 2018-era environment. It is not claimed as an established authors' environment.
