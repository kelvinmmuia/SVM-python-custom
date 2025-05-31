# kernels.py

import numpy as np

def linear(X1, X2=None):
    """
    Linear kernel: K(x, x') = x · x'
    If X2 is None, computes X1 · X1T (useful for Gram matrices).
    """
    # Original code had X1 @ X1.T for X2 is None,
    # but for consistency in _kernel, X2 will usually be X1 for Gram matrix.
    # If X2 is truly None, it implies X1 vs X1.
    if X2 is None:
        X2 = X1
    return X1 @ X2.T

def rbf(X1, X2, gamma=None):
    """
    RBF (Gaussian) kernel: exp(-gamma ||x - x'||²).
    gamma = 1 / n_features by default if None.
    """
    if X2 is None: # Should not happen if _kernel prepares X2_use
        X2 = X1
    if gamma is None:
        gamma = 1.0 / X1.shape[1]
    # pairwise squared distances
    XX = np.sum(X1**2, axis=1)[:, None]
    YY = np.sum(X2**2, axis=1)[None, :]
    distances = XX + YY - 2 * (X1 @ X2.T)
    # Clip distances to be non-negative to avoid issues with floating point inaccuracies
    distances = np.maximum(distances, 0) 
    return np.exp(-gamma * distances)

def polynomial(X1, X2, degree=3, gamma=None, coef0=1):
    """
    Polynomial kernel: (gamma * (x · x') + coef0)^degree
    gamma = 1 / n_features by default if None.
    """
    if X2 is None:
        X2 = X1
    if gamma is None:
        gamma = 1.0 / X1.shape[1]
    return (gamma * (X1 @ X2.T) + coef0) ** degree

def sigmoid(X1, X2, gamma=None, coef0=1):
    """
    Sigmoid kernel: tanh(gamma * (x · x') + coef0)
    gamma = 1 / n_features by default if None.
    """
    if X2 is None:
        X2 = X1
    if gamma is None:
        gamma = 1.0 / X1.shape[1]
    return np.tanh(gamma * (X1 @ X2.T) + coef0)