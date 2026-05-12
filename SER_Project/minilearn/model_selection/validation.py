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

import copy
import itertools
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
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        classes = np.unique(y)

        # Group sample indices by class, optionally shuffle within each group
        class_indices = []
        for c in classes:
            idx = np.where(y == c)[0]
            if self.shuffle:
                idx = rng.permutation(idx)
            class_indices.append(idx)

        # Split each class's indices into n_splits chunks
        class_folds = [np.array_split(idx, self.n_splits) for idx in class_indices]

        for fold in range(self.n_splits):
            val_idx   = np.concatenate([cf[fold] for cf in class_folds])
            train_idx = np.concatenate([
                np.concatenate([cf[j] for j in range(self.n_splits) if j != fold])
                for cf in class_folds
            ])
            yield train_idx.astype(int), val_idx.astype(int)

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

    def _param_combinations(self):
        """Yield all dicts from the cartesian product of param_grid values."""
        keys   = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def fit(self, X, y):
        """
        Search over all parameter combinations, evaluate each via k-fold,
        store the best parameters in self.best_params_.
        """
        X, y = np.asarray(X), np.asarray(y)
        results = []

        for params in self._param_combinations():
            clf = copy.deepcopy(self.estimator)
            clf.__dict__.update(params)
            fold_scores = cross_val_score(clf, X, y, cv=self.cv, scoring=self.scoring)
            mean_score  = float(fold_scores.mean())
            std_score   = float(fold_scores.std())
            results.append({
                'params':     params,
                'mean_score': mean_score,
                'std_score':  std_score,
                'fold_scores': fold_scores,
            })

        self.cv_results_ = results
        best = max(results, key=lambda r: r['mean_score'])
        self.best_params_ = best['params']
        self.best_score_  = best['mean_score']
        return self

    def best_estimator(self, X=None, y=None):
        """
        Return a new estimator fitted with best_params_.
        If X and y are provided, fits it on that data; otherwise returns unfitted.
        """
        clf = copy.deepcopy(self.estimator)
        clf.__dict__.update(self.best_params_)
        if X is not None and y is not None:
            clf.fit(np.asarray(X), np.asarray(y))
        return clf

    def results_dataframe(self):
        """Return cv_results_ as a sorted pandas DataFrame (requires pandas)."""
        import pandas as pd
        rows = [
            {**r['params'], 'mean_score': r['mean_score'], 'std_score': r['std_score']}
            for r in self.cv_results_
        ]
        return pd.DataFrame(rows).sort_values('mean_score', ascending=False).reset_index(drop=True)


def cross_val_score(estimator, X, y, cv=5, scoring="accuracy"):
    """
    Evaluate estimator performance using stratified k-fold cross-validation.

    Returns
    -------
    scores : np.ndarray, shape (cv,) — one score per fold
    """
    X, y = np.asarray(X), np.asarray(y)
    kfold  = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    scores = []

    for train_idx, val_idx in kfold.split(X, y):
        clf = copy.deepcopy(estimator)
        clf.fit(X[train_idx], y[train_idx])
        if scoring == "accuracy":
            score = clf.score(X[val_idx], y[val_idx])
        else:
            raise ValueError(f"Unsupported scoring '{scoring}'. Use 'accuracy'.")
        scores.append(score)

    return np.array(scores)
