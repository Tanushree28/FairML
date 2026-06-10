"""Differentiable composite fairness loss.

This fixes the two training bugs in the original Loss.py / Models.py:

1. The original fairness terms were computed with fairlearn/aif360 on
   `y_pred.round().detach().numpy()`, which cuts them out of the autograd
   graph — only BCE ever produced gradients, so alpha/beta had almost no
   effect on the learned model. Here every term is computed in torch on the
   raw predicted probabilities, so gradients flow through all components.

2. The original model applied sigmoid in `forward` and the loss applied
   sigmoid again (double sigmoid). Here models return logits and the loss
   uses `binary_cross_entropy_with_logits`.

Loss composition (same weighting scheme as the original):

    total = alpha * BCE + (1 - alpha) * (beta * group + (1 - beta) * individual)

- group term:      soft demographic parity (largest gap in mean predicted
                   probability between any two sensitive groups)
- individual term: soft generalized entropy index (Speicher et al. 2018) on
                   the benefit vector b_i = 1 + p_i - y_i, with ge_alpha = 2,
                   matching the Theil-style index used for evaluation.
"""

import torch
import torch.nn.functional as F


def soft_demographic_parity(probs, group_ids):
    """Differentiable demographic-parity gap.

    Largest difference in mean predicted probability between any two
    sensitive groups. Equals the hard demographic_parity_difference when
    probabilities saturate to 0/1.
    """
    group_means = torch.stack([probs[group_ids == g].mean() for g in torch.unique(group_ids)])
    return group_means.max() - group_means.min()


def soft_generalized_entropy(probs, y_true, ge_alpha=2.0):
    """Differentiable generalized entropy index on the benefit vector.

    b_i = 1 + p_i - y_i (Speicher et al. 2018). ge_alpha=2 corresponds to the
    half squared coefficient of variation, the same family as the evaluation
    Theil index (aif360 generalized_entropy_error with alpha=2).
    """
    b = 1.0 + probs - y_true
    mu = b.mean()
    return (((b / mu) ** ge_alpha - 1.0).mean()) / (ge_alpha * (ge_alpha - 1.0))


def composite_loss(logits, y_true, group_ids, alpha, beta):
    """alpha * BCE + (1 - alpha) * (beta * soft_DP + (1 - beta) * soft_GE)."""
    bce = F.binary_cross_entropy_with_logits(logits, y_true)
    probs = torch.sigmoid(logits)
    group_term = soft_demographic_parity(probs, group_ids)
    individual_term = soft_generalized_entropy(probs, y_true)
    fairness = beta * group_term + (1.0 - beta) * individual_term
    return alpha * bce + (1.0 - alpha) * fairness
