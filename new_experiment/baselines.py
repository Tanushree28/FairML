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
from xgboost import XGBClassifier


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


def xgboost_reference(X_train, y_train, X_test, seed):
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        random_state=seed, eval_metric="logloss")
    clf.fit(X_train, y_train)
    return clf.predict(X_test)
