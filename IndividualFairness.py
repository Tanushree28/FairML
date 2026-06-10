from aif360.sklearn.metrics import generalized_entropy_error
import numpy as np

def theil_index(y_true, y_pred):
    return generalized_entropy_error(y_true, y_pred, alpha = 2)

def generalized_entropy_index(y_true, y_pred):
    return generalized_entropy_error(y_true, y_pred, alpha = 1)

def atkinson_index(y_true, y_pred, epsilon=0.5):
    """ 
    Calculates the Atkinson Index for inequality in model outcomes.
    
    Parameters:
    - y_true: True labels (numpy array)
    - y_pred: Model predictions (numpy array)
    - epsilon: Inequality aversion parameter (0 < epsilon < 1)
    
    Returns:
    - Atkinson Index value
    """
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("Epsilon must be between 0 and 1.")

    n = len(y_pred)
    mean_pred = np.mean(y_pred)
    if mean_pred == 0:
        return 0  # Avoid division by zero

    atkinson_sum = np.mean((y_pred / mean_pred) ** (1 - epsilon))
    atkinson_value = 1 - atkinson_sum ** (1 / (1 - epsilon))
    return atkinson_value

def gini_coefficient(y_true, y_pred):
    sorted_pred = np.sort(y_pred)
    n = len(y_pred)

    lorenz_curve = np.cumsum(sorted_pred) / np.sum(sorted_pred)
    lorenz_curve = np.insert(lorenz_curve, 0, 0)

    gini_value = 1 - 2 * np.trapz(lorenz_curve, np.linspace(0, 1, n + 1))
    
    return gini_value
