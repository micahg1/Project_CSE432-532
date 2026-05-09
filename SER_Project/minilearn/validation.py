"""
minilearn/validation.py
=======================
Cross-validation and hyperparameter tuning utilities.

Classes
-------
StratifiedKFold
    Splits data into k folds while preserving class proportions in each fold.
    Use this for all classical model evaluation in the project.

GridSearchCV
    Exhaustive search over a parameter grid.  Evaluates each combination
    with StratifiedKFold and records the best parameters + best score.

Functions
---------
cross_val_score(estimator, X, y, cv, scoring)
    Convenience wrapper: fits estimator on each fold and returns an array
    of scores, one per fold.

Usage
-----
    from minilearn.validation import StratifiedKFold, cross_val_score

    kfold = StratifiedKFold(n_splits=5, random_state=42)
    for train_idx, val_idx in kfold.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        ...

    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(scores.mean(), scores.std())
"""

import numpy as np


class StratifiedKFold:
    """
    k-Fold cross-validator that keeps class proportions in each split.

    Parameters
    ----------
    n_splits     : int, number of folds (default 5)
    shuffle      : bool, whether to shuffle each class's samples before splitting
    random_state : int or None
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        """
        Generate (train_indices, val_indices) pairs for each fold.

        Yields
        ------
        train_idx : np.ndarray of int
        val_idx   : np.ndarray of int
        """
        # TODO: group sample indices by class, split each group into n_splits
        #       chunks, then yield union of all-but-one chunk as train, one as val
        raise NotImplementedError

    def get_n_splits(self):
        return self.n_splits


class GridSearchCV:
    """
    Exhaustive hyperparameter search using cross-validation.

    Parameters
    ----------
    estimator   : object with .fit() and .score()
    param_grid  : dict, e.g. {'lr': [0.001, 0.01], 'n_iter': [500, 1000]}
    cv          : int, number of folds (uses StratifiedKFold internally)
    scoring     : str, 'accuracy' (only option for now)
    """

    def __init__(self, estimator, param_grid, cv=5, scoring="accuracy"):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None

    def fit(self, X, y):
        """
        Search over all parameter combinations, evaluate each via k-fold,
        store the best parameters in self.best_params_.
        """
        # TODO: use itertools.product to enumerate all combinations,
        #       instantiate a fresh copy of estimator with each combo,
        #       run cross_val_score, track best mean score
        raise NotImplementedError

    def best_estimator(self):
        """Return a new estimator instance fitted with best_params_ on full data."""
        # TODO: implement
        raise NotImplementedError


def cross_val_score(estimator, X, y, cv=5, scoring="accuracy"):
    """
    Evaluate estimator performance using stratified k-fold cross-validation.

    Returns
    -------
    scores : np.ndarray, shape (cv,) — one score per fold
    """
    # TODO: instantiate StratifiedKFold, iterate folds, fit + score each time
    raise NotImplementedError
