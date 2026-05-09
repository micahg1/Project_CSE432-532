"""
minilearn/preprocessing.py
===========================
Data preprocessing utilities implemented from scratch using NumPy.

Classes
-------
StandardScaler
    Fits mean/std on training data and applies z-score normalization.
    IMPORTANT: always fit on training data only, then transform train + test
    separately to avoid data leakage.

Functions
---------
train_test_split(X, y, test_size, random_state)
    Randomly splits arrays into train/test subsets while preserving class
    proportions when stratify=True.

Usage
-----
    from minilearn.preprocessing import StandardScaler, train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit AND transform train
    X_test_scaled  = scaler.transform(X_test)        # transform only — no fit!
"""

import numpy as np


class StandardScaler:
    """Z-score normalization: (x - mean) / std, computed per feature column."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Compute per-column mean and std from training data X."""
        # TODO: implement using np.mean and np.std along axis=0
        raise NotImplementedError

    def transform(self, X):
        """Apply stored mean/std to X.  Must call fit() first."""
        # TODO: subtract mean_, divide by std_ (handle zero-std columns)
        raise NotImplementedError

    def fit_transform(self, X):
        """Convenience: fit then transform in one call (training data only)."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Reverse the scaling: X * std_ + mean_"""
        # TODO: implement
        raise NotImplementedError


def train_test_split(X, y, test_size=0.2, random_state=None, stratify=False):
    """
    Split X and y into random train/test subsets.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    test_size : float, fraction of samples for test set (default 0.2)
    random_state : int or None, seed for reproducibility
    stratify : bool, if True maintain class proportions in each split

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    # TODO: implement — handle both stratified and random splits
    raise NotImplementedError
