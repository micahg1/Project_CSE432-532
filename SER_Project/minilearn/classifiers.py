"""
minilearn/classifiers.py
========================
Supervised classification algorithms implemented from scratch using NumPy.

Each class follows the scikit-learn API:
    .fit(X, y)        — train the model
    .predict(X)       — return predicted class labels
    .predict_proba(X) — return class probabilities (where applicable)
    .score(X, y)      — return accuracy

Classes
-------
LogisticRegression
    Multi-class logistic regression via softmax + gradient descent (or
    scipy.optimize.minimize for the optimization step).

KNearestNeighbors
    Lazy learner: stores training data and classifies by majority vote
    among the k nearest neighbors (Euclidean distance by default).

GaussianNaiveBayes
    Probabilistic classifier using Bayes' theorem with the Gaussian
    likelihood assumption per feature.

LinearSVM
    Simplified linear SVM using a hinge-loss subgradient update rule.
    For full kernel support, wrap sklearn's SVC in notebooks.

DecisionTree
    Binary CART decision tree that splits on the feature/threshold
    minimizing Gini impurity (or entropy — your choice).

Usage
-----
    from minilearn.classifiers import LogisticRegression

    clf = LogisticRegression(lr=0.01, n_iter=500)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(clf.score(X_test, y_test))
"""

import numpy as np


class LogisticRegression:
    """
    Softmax (multi-class) logistic regression with gradient descent.

    Parameters
    ----------
    lr       : float, learning rate
    n_iter   : int,   number of gradient-descent iterations
    tol      : float, convergence tolerance on loss change
    """

    def __init__(self, lr=0.01, n_iter=1000, tol=1e-4):
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.weights_ = None
        self.bias_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Train using softmax cross-entropy loss + gradient descent.
        Tip: you may use scipy.optimize.minimize for the optimizer, but
        you must write the loss function and gradient yourself.
        """
        # TODO: implement one-hot encoding, softmax, cross-entropy loss, gradients
        raise NotImplementedError

    def predict_proba(self, X):
        """Return softmax probabilities, shape (n_samples, n_classes)."""
        # TODO: implement softmax(X @ W + b)
        raise NotImplementedError

    def predict(self, X):
        """Return argmax of predicted probabilities."""
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class KNearestNeighbors:
    """
    k-Nearest Neighbors classifier.

    Parameters
    ----------
    k        : int,    number of neighbors
    metric   : str,    'euclidean' or 'manhattan'
    """

    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        """Store training data (KNN is a lazy learner — no actual training)."""
        # TODO: store X and y
        raise NotImplementedError

    def predict(self, X):
        """
        For each test sample, find k nearest training neighbors and
        return the majority class label.
        """
        # TODO: compute pairwise distances, find k smallest, majority vote
        raise NotImplementedError

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes: models each feature as Gaussian given the class.

    Stores per-class mean, variance, and prior during fit.
    """

    def __init__(self):
        self.class_priors_ = None
        self.means_ = None
        self.variances_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Estimate per-class Gaussian parameters and class priors."""
        # TODO: for each class, compute mean and variance of each feature
        raise NotImplementedError

    def predict_proba(self, X):
        """Return log-posterior probabilities for numerical stability."""
        # TODO: sum log-likelihoods + log-prior per class
        raise NotImplementedError

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class LinearSVM:
    """
    Simplified linear SVM using subgradient descent on hinge loss.
    Only supports binary classification; for multi-class use one-vs-rest.

    Parameters
    ----------
    C        : float, regularization parameter
    lr       : float, learning rate
    n_iter   : int,   number of iterations
    """

    def __init__(self, C=1.0, lr=0.001, n_iter=1000):
        self.C = C
        self.lr = lr
        self.n_iter = n_iter
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        """
        Train with subgradient updates.
        Loss = 0.5 * ||w||^2 + C * sum(max(0, 1 - y_i*(w·x_i + b)))
        """
        # TODO: implement subgradient descent
        raise NotImplementedError

    def predict(self, X):
        """Return sign of decision function."""
        # TODO: return np.sign(X @ self.weights_ + self.bias_)
        raise NotImplementedError

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class DecisionTree:
    """
    Binary CART decision tree using Gini impurity for splits.

    Parameters
    ----------
    max_depth   : int or None, maximum tree depth
    min_samples : int, minimum samples required to split a node
    criterion   : str, 'gini' or 'entropy'
    """

    def __init__(self, max_depth=None, min_samples=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.criterion = criterion
        self.root_ = None

    def fit(self, X, y):
        """
        Recursively build the tree by finding the best (feature, threshold)
        split at each node using the CART algorithm.
        """
        # TODO: implement _build_tree, _best_split, _gini/_entropy helpers
        raise NotImplementedError

    def predict(self, X):
        """Traverse the tree for each sample and return leaf class."""
        # TODO: implement _traverse helper
        raise NotImplementedError

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
