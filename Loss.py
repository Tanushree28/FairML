import torch
import torch.nn.functional as F
from fairlearn.metrics import demographic_parity_difference
from IndividualFairness import theil_index

def binary_cross_entropy(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)
    loss = F.binary_cross_entropy(y_pred, y_true)
    
    return loss

def custom_loss_function(y_true, y_pred, sensitive_features, group_fairness, individual_fairness, alpha, beta):
    bce_loss = binary_cross_entropy(y_pred, y_true)

    group_fairness = group_fairness(y_true=y_true.numpy(), y_pred=y_pred.round().detach().numpy(), sensitive_features=sensitive_features)
    individual_fairness = individual_fairness(y_true=y_true.numpy(), y_pred=y_pred.round().detach().numpy())

    fairness_loss = beta * group_fairness + (1 - beta) * individual_fairness
    total_loss = alpha * bce_loss + (1 - alpha) * fairness_loss
    
    return total_loss

