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
    Principal Component Analysis via eigendecomposition of the covariance matrix.

    Parameters
    ----------
    n_components : int or None
        Number of components to keep.  If None, keep all components.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None           # shape (n_components, n_features)
        self.explained_variance_ = None   # eigenvalues of top components
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """
        Compute principal components from training data X.

        Steps:
        1. Center X by subtracting the column mean.
        2. Compute the covariance matrix (or use np.linalg.svd on centered X).
        3. Sort eigenvectors by descending eigenvalue.
        4. Store top n_components axes in self.components_.
        """
        # TODO: implement using np.cov + np.linalg.eigh  (or np.linalg.svd)
        raise NotImplementedError

    def transform(self, X):
        """Project X onto the stored principal axes."""
        # TODO: center X using self.mean_, then dot with self.components_.T
        raise NotImplementedError

    def fit_transform(self, X):
        """Fit on X then project X (use for training data only)."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced):
        """Reconstruct approximate original-space data from reduced representation."""
        # TODO: X_reduced @ self.components_ + self.mean_
        raise NotImplementedError
