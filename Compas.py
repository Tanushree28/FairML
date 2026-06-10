import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from GroupFairness import equal_opportunity_difference, demographic_parity_difference, disparate_impact_difference, equalized_odds_difference
from IndividualFairness import theil_index, generalized_entropy_index, atkinson_index, gini_coefficient
from sklearn.metrics import accuracy_score
from Loss import custom_loss_function, binary_cross_entropy

from Models import LogisticRegressionModel
 
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)

model_dir = "NEW_MODEL/"
results_dir = "NEW_RESULTS/"
plots_dir = f"{results_dir}PLOTS/"
results_csv = f"{results_dir}results.csv"

Path(model_dir).mkdir(parents=True, exist_ok=True)
Path(plots_dir).mkdir(parents=True, exist_ok=True)

url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
data = pd.read_csv(url)

print("Shape:", data.shape)
print("\nFirst 10 rows:")
print(data.head(10))

print("\nColumns:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

data["duration"] = data["end"] - data["start"]
features = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'race', 'sex', 'duration']
target = 'two_year_recid'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

sensitive_feature_X_train = X_train["race"]
sensitive_feature_X_test = X_test["race"]
sensitive_feature_X_val = X_val["race"]

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
X_val = preprocessor.transform(X_val)

X_train_tensorized = torch.tensor(X_train, dtype=torch.float32)
X_test_tensorized = torch.tensor(X_test, dtype=torch.float32)
X_val_tensorized = torch.tensor(X_val, dtype=torch.float32)

y_train_tensorized = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensorized = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
y_val_tensorized = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

input_dim = X_train.shape[1]

alpha_list = [0, 0.005, 0.05, 0.25, 0.5, 0.75, 1]
beta_list = [0, 0.005, 0.05, 0.25, 0.5, 0.75, 1]
group_fairness_list = [demographic_parity_difference, equalized_odds_difference, equal_opportunity_difference, disparate_impact_difference]
individual_fairness_list = [theil_index, generalized_entropy_index, atkinson_index, gini_coefficient]

combs = [(group_fairness, individual_fairness, alpha, beta) for alpha in alpha_list for beta in beta_list for group_fairness in group_fairness_list for individual_fairness in individual_fairness_list]

results = []
sweep_start = time.perf_counter()
for comb in combs:
    print(f"Group Fairness: {comb[0].__name__}, Individual Fairness: {comb[1].__name__}, Alpha: {comb[2]}, Beta: {comb[3]}")

    model_name = f"model_alpha_{comb[2]}_beta_{comb[3]}_group_{comb[0].__name__}_individual_{comb[1].__name__}.pth"

    combo_start = time.perf_counter()
    if Path(model_dir + model_name).exists():
        logreg_model = LogisticRegressionModel(input_dim)
        logreg_model.load_state_dict(torch.load(model_dir + model_name, weights_only=True))
        trained_this_run = False
    else:
        logreg_model = LogisticRegressionModel(input_dim)
        logreg_model.train_model(custom_loss_function, X_train_tensorized, y_train_tensorized, sensitive_features = sensitive_feature_X_train, group_fairness = comb[0], individual_fairness= comb[1], alpha = comb[2], beta = comb[3])
        logreg_model.save_model(model_dir + model_name)
        trained_this_run = True
    combo_time = time.perf_counter() - combo_start
    print(f"  -> {'trained' if trained_this_run else 'loaded'} in {combo_time:.2f}s")

    logreg_model.evaluate(X = X_test_tensorized, y = y_test_tensorized)

    y_pred = logreg_model(X_test_tensorized).round().detach().numpy().ravel()

    accuracy = accuracy_score(y_test, y_pred)
    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_feature_X_test)
    eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_feature_X_test)
    theil = theil_index(y_test, y_pred)
    gini = gini_coefficient(y_test, y_pred)

    results.append({
        'Alpha': comb[2],
        'Beta': comb[3],
        'Group Fairness': comb[0].__name__,
        'Individual Fairness': comb[1].__name__,
        'Accuracy': round(accuracy, 2),
        'Demographic Parity Difference': round(dpd, 2),
        'Equalized Odds Difference': round(eod, 2),
        'Theil Index': round(theil, 2),
        'Gini Coefficient': round(gini, 2),
        'Train Time (s)': round(combo_time, 3),
        'Trained This Run': trained_this_run
    })

total_sweep_time = time.perf_counter() - sweep_start
print(f"\nFull sweep over {len(combs)} combinations: {total_sweep_time:.2f}s ({total_sweep_time/60:.2f} min)")
trained_times = [r['Train Time (s)'] for r in results if r['Trained This Run']]
if trained_times:
    print(f"Trained {len(trained_times)} models | mean: {np.mean(trained_times):.2f}s | "
          f"min: {np.min(trained_times):.2f}s | max: {np.max(trained_times):.2f}s")

if Path(results_csv).exists():
    results_df = pd.read_csv(results_csv)
else:
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv, index=False)

metrics = [
    "Accuracy",
    "Demographic Parity Difference",
    "Equalized Odds Difference",
    "Theil Index",
    "Gini Coefficient"
]

for metric in metrics:
    heatmap_data = results_df.pivot_table(index="Alpha", columns="Beta", values=metric)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis" if metric == "Accuracy" else "coolwarm")
    plt.title(f"Heatmap of {metric} for Different Alpha and Beta Values")
    plt.xlabel("Beta")
    plt.ylabel("Alpha")
    plt.savefig(f"{plots_dir}heatmap_{metric}.png")