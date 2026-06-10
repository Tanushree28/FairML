from fairlearn.metrics import MetricFrame
from fairlearn.metrics import true_positive_rate, false_positive_rate, selection_rate
from fairlearn.metrics import demographic_parity_difference as dpd_metric
from fairlearn.metrics import equalized_odds_difference as eod_metric

def equal_opportunity_difference(y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None) -> float:
    tpr = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"sample_weight": sample_weight} if sample_weight is not None else None
    )

    result = tpr.difference(method=method)
    return result

def disparate_impact_difference(y_true, y_pred, *, sensitive_features, sample_weight=None) -> float:
    selection_rates = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"sample_weight": sample_weight} if sample_weight is not None else None
    )
    
    min_rate = selection_rates.group_min()
    max_rate = selection_rates.group_max()
    
    disparate_impact_diff = max_rate - min_rate
    return disparate_impact_diff

def average_odds_difference(y_true, y_pred, *, sensitive_features, sample_weight=None) -> float:
    tpr = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"sample_weight": sample_weight} if sample_weight is not None else None
    )
    
    fpr = MetricFrame(
        metrics=false_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"sample_weight": sample_weight} if sample_weight is not None else None
    )
    
    tpr_difference = tpr.difference(method="between_groups")
    fpr_difference = fpr.difference(method="between_groups")
    
    avg_odds_diff = (tpr_difference + fpr_difference) / 2
    return avg_odds_diff

def demographic_parity_difference(y_true, y_pred, *, sensitive_features, sample_weight=None) -> float:
    return dpd_metric(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)

def equalized_odds_difference(y_true, y_pred, *, sensitive_features, sample_weight=None) -> float:
    return eod_metric(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)