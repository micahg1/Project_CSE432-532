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
        n_samples, n_features = X.shape

        if self.solver == "analytic":
            # Augment X with a bias column of ones
            X_b = np.hstack([np.ones((n_samples, 1)), X])
            # w = pinv(X_b^T X_b) X_b^T y
            params = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            self.bias_ = params[0]
            self.weights_ = params[1:]

        elif self.solver == "gradient_descent":
            self.weights_ = np.zeros(n_features)
            self.bias_ = 0.0

            for _ in range(self.n_iter):
                y_pred = X @ self.weights_ + self.bias_
                error = y_pred - y
                grad_w = (2 / n_samples) * (X.T @ error)
                grad_b = (2 / n_samples) * np.sum(error)
                self.weights_ -= self.lr * grad_w
                self.bias_ -= self.lr * grad_b

        else:
            raise ValueError(f"Unknown solver '{self.solver}'. Use 'analytic' or 'gradient_descent'.")

    def predict(self, X):
        """Return continuous predictions: X @ w + b."""
        return X @ self.weights_ + self.bias_

    def score(self, X, y):
        """Return R² coefficient of determination."""
        return r2_score(y, self.predict(X))


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
        n_features = X.shape[1]

        # Center y and X to absorb the bias (avoids regularizing the intercept)
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()
        Xc = X - self.x_mean_
        yc = y - self.y_mean_

        # w = (Xc^T Xc + alpha * I)^{-1} Xc^T yc
        I = np.eye(n_features)
        self.weights_ = np.linalg.solve(Xc.T @ Xc + self.alpha * I, Xc.T @ yc)
        self.bias_ = self.y_mean_ - self.x_mean_ @ self.weights_

    def predict(self, X):
        return X @ self.weights_ + self.bias_

    def score(self, X, y):
        return r2_score(y, self.predict(X))


# ── Regression-specific metrics (also importable from minilearn.metrics) ──────

def mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    """
    Coefficient of determination R².
    R² = 1 - SS_res / SS_tot
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot
