"""
minilearn/regression.py
=======================
Regression models implemented from scratch using NumPy.

In the context of this SER project, regression is used in Week 6 to predict
continuous-valued targets from audio features — for example, predicting the
emotional intensity label (1 = normal, 2 = strong) as a real number, or any
other continuous audio property you choose to explore.

Classes
-------
LinearRegression
    Ordinary Least Squares regression solved analytically via the normal
    equations: w = (X^T X)^{-1} X^T y.
    Also supports a gradient-descent solver for large feature spaces.

RidgeRegression
    L2-regularized least squares: adds lambda * ||w||^2 to the loss to
    prevent overfitting on high-dimensional audio features.

Usage
-----
    from minilearn.regression import LinearRegression, RidgeRegression

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    from minilearn.metrics import mse, r2_score
    print(mse(y_test, y_pred))
    print(r2_score(y_test, y_pred))
"""

import numpy as np


class LinearRegression:
    """
    Ordinary Least Squares linear regression.

    Parameters
    ----------
    solver : str, 'analytic' (normal equations) or 'gradient_descent'
    lr     : float, learning rate (only used when solver='gradient_descent')
    n_iter : int,   number of iterations (gradient descent only)
    """

    def __init__(self, solver="analytic", lr=0.01, n_iter=1000):
        self.solver = solver
        self.lr = lr
        self.n_iter = n_iter
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        """
        Fit model to training data.

        Analytic: w = (X^T X)^{-1} X^T y  (use np.linalg.pinv for stability)
        Gradient descent: iteratively update w and b using MSE gradients.
        """
        # TODO: implement both solver branches
        raise NotImplementedError

    def predict(self, X):
        """Return continuous predictions: X @ w + b."""
        # TODO: implement
        raise NotImplementedError

    def score(self, X, y):
        """Return R² coefficient of determination."""
        # TODO: implement using r2_score helper or inline formula
        raise NotImplementedError


class RidgeRegression:
    """
    L2-regularized linear regression (Ridge).

    Adds alpha * ||w||^2 to the MSE loss.
    Analytic solution: w = (X^T X + alpha * I)^{-1} X^T y

    Parameters
    ----------
    alpha : float, regularization strength (default 1.0)
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        """Fit using the Ridge normal equations."""
        # TODO: implement
        raise NotImplementedError

    def predict(self, X):
        # TODO: implement
        raise NotImplementedError

    def score(self, X, y):
        # TODO: R² score
        raise NotImplementedError


# ── Regression-specific metrics (also importable from minilearn.metrics) ──────

def mse(y_true, y_pred):
    """Mean Squared Error."""
    # TODO: np.mean((y_true - y_pred) ** 2)
    raise NotImplementedError


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    # TODO: np.sqrt(mse(y_true, y_pred))
    raise NotImplementedError


def r2_score(y_true, y_pred):
    """
    Coefficient of determination R².
    R² = 1 - SS_res / SS_tot
    """
    # TODO: implement
    raise NotImplementedError
