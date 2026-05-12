"""
minilearn/decomposition.py
==========================
Dimensionality reduction implemented from scratch using NumPy.

Classes
-------
PCA (Principal Component Analysis)
    Computes the top-n_components principal axes via eigendecomposition of the
    covariance matrix (or equivalently SVD of the centered data matrix).

    Two common uses in this project:
    1. Feature compression — reduce high-dimensional audio features before
       passing them to classifiers to speed up training and reduce overfitting.
    2. 2-D visualization — project down to 2 components so clusters/classes
       can be plotted in a scatter plot.

Usage
-----
    from minilearn.decomposition import PCA

    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)   # fit on train, transform train
    X_test_pca  = pca.transform(X_test)        # transform test with same axes

    # Explained variance
    print(pca.explained_variance_ratio_)       # fraction per component
    print(pca.explained_variance_ratio_.cumsum())  # cumulative

    # 2-D scatter for visualization
    pca2d = PCA(n_components=2)
    X_2d = pca2d.fit_transform(X_scaled)
"""

import numpy as np


class PCA:
    """
    Principal Component Analysis via SVD of the centered data matrix.

    Using SVD on the (n_samples × n_features) centered matrix is numerically
    more stable than eigendecomposing the covariance matrix directly, and is
    faster when n_samples < n_features.

    Parameters
    ----------
    n_components : int or None
        Number of components to keep.  If None, keep all components.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None           # shape (n_components, n_features)
        self.explained_variance_ = None   # variance captured by each component
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """
        Compute principal components from training data X.

        Steps:
        1. Center X by subtracting the column mean.
        2. Run full SVD on the centered matrix: X_c = U S Vt.
        3. Principal axes  = rows of Vt  (right singular vectors).
        4. Explained variance per component = s² / (n_samples - 1).
        5. Keep only the top n_components axes.
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Full SVD — Vt rows are the principal axes, sorted by descending singular value
        _, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Explained variance: eigenvalues of the covariance matrix = s² / (n-1)
        explained_var = (s ** 2) / (n_samples - 1)
        total_var     = explained_var.sum()

        # Determine number of components to keep
        n_components = self.n_components if self.n_components is not None else n_features
        n_components = min(n_components, n_features, n_samples)

        self.components_               = Vt[:n_components]
        self.explained_variance_       = explained_var[:n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        return self

    def transform(self, X):
        """Project X onto the stored principal axes."""
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        """Fit on X then project X (use for training data only)."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced):
        """Reconstruct approximate original-space data from reduced representation."""
        return np.asarray(X_reduced, dtype=float) @ self.components_ + self.mean_
