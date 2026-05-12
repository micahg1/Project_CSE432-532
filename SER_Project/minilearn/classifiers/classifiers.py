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

    @staticmethod
    def _softmax(logits):
        # Subtract max per row for numerical stability
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        """Train using softmax cross-entropy loss + gradient descent."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Map labels to 0-based indices
        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_to_idx[c] for c in y])

        # One-hot encode targets
        Y = np.zeros((n_samples, n_classes))
        Y[np.arange(n_samples), y_idx] = 1.0

        self.weights_ = np.zeros((n_features, n_classes))
        self.bias_    = np.zeros(n_classes)

        prev_loss = np.inf
        for _ in range(self.n_iter):
            logits = X @ self.weights_ + self.bias_
            probs  = self._softmax(logits)

            # Cross-entropy loss
            loss = -np.mean(np.sum(Y * np.log(probs + 1e-15), axis=1))

            # Gradients
            delta = (probs - Y) / n_samples
            self.weights_ -= self.lr * (X.T @ delta)
            self.bias_    -= self.lr * delta.sum(axis=0)

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

    def predict_proba(self, X):
        """Return softmax probabilities, shape (n_samples, n_classes)."""
        logits = np.asarray(X, dtype=float) @ self.weights_ + self.bias_
        return self._softmax(logits)

    def predict(self, X):
        """Return argmax of predicted probabilities."""
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))


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
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)

    def _distances(self, X):
        if self.metric == "manhattan":
            # ||x - x'||_1  via broadcasting
            return np.sum(np.abs(X[:, None, :] - self.X_train_[None, :, :]), axis=2)
        # Default: Euclidean — computed via ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b^T
        X2  = np.sum(X ** 2, axis=1, keepdims=True)
        Xt2 = np.sum(self.X_train_ ** 2, axis=1)
        return np.sqrt(np.maximum(X2 + Xt2 - 2 * (X @ self.X_train_.T), 0.0))

    def predict(self, X):
        """Find k nearest training neighbors and return the majority class label."""
        X = np.asarray(X, dtype=float)
        dists = self._distances(X)                          # (n_test, n_train)
        k_idx = np.argpartition(dists, self.k, axis=1)[:, :self.k]
        neighbors = self.y_train_[k_idx]                    # (n_test, k)
        # Majority vote — np.unique works for any label type
        def majority(row):
            vals, counts = np.unique(row, return_counts=True)
            return vals[np.argmax(counts)]
        return np.array([majority(row) for row in neighbors])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))


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
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]

        self.means_         = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        # Add small epsilon to avoid zero-variance features
        self.variances_     = np.array([X[y == c].var(axis=0) + 1e-9 for c in self.classes_])
        self.class_priors_  = np.array([np.sum(y == c) / n_samples for c in self.classes_])

    def predict_proba(self, X):
        """Return log-posterior scores for numerical stability."""
        X = np.asarray(X, dtype=float)
        log_probs = np.zeros((X.shape[0], len(self.classes_)))
        for i, (mean, var, prior) in enumerate(
            zip(self.means_, self.variances_, self.class_priors_)
        ):
            # Log of Gaussian PDF: -0.5 * log(2π σ²) - (x-μ)²/(2σ²)
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * var) + ((X - mean) ** 2) / var, axis=1
            )
            log_probs[:, i] = log_likelihood + np.log(prior)
        return log_probs

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))


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
        X = np.asarray(X, dtype=float)
        # Map labels to {-1, +1}
        classes = np.unique(y)
        y_pm = np.where(np.asarray(y) == classes[1], 1.0, -1.0)

        n_features = X.shape[1]
        self.weights_ = np.zeros(n_features)
        self.bias_    = 0.0
        self._classes = classes

        for _ in range(self.n_iter):
            margins = y_pm * (X @ self.weights_ + self.bias_)
            mask    = margins < 1                               # hinge is active

            grad_w = self.weights_ - self.C * (y_pm[mask, None] * X[mask]).sum(axis=0)
            grad_b = -self.C * y_pm[mask].sum()

            self.weights_ -= self.lr * grad_w
            self.bias_    -= self.lr * grad_b

    def predict(self, X):
        """Return class labels based on sign of decision function."""
        scores = np.asarray(X, dtype=float) @ self.weights_ + self.bias_
        return np.where(scores >= 0, self._classes[1], self._classes[0])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))


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

    # ── Impurity helpers ──────────────────────────────────────────────────

    def _impurity(self, y):
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        if self.criterion == "entropy":
            return -np.sum(probs * np.log2(probs + 1e-15))
        # Gini
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -np.inf, None, None
        parent_imp = self._impurity(y)
        n = len(y)

        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left  = y[X[:, feat] <= thresh]
                right = y[X[:, feat] >  thresh]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent_imp - (
                    len(left)  / n * self._impurity(left) +
                    len(right) / n * self._impurity(right)
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        return best_feat, best_thresh

    def _build_tree(self, X, y, depth):
        # Stopping conditions → leaf node
        if (
            len(y) < self.min_samples
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(np.unique(y)) == 1
        ):
            return {"leaf": True, "label": np.bincount(y.astype(int)).argmax()}

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return {"leaf": True, "label": np.bincount(y.astype(int)).argmax()}

        mask = X[:, feat] <= thresh
        return {
            "leaf":   False,
            "feat":   feat,
            "thresh": thresh,
            "left":   self._build_tree(X[mask],  y[mask],  depth + 1),
            "right":  self._build_tree(X[~mask], y[~mask], depth + 1),
        }

    def fit(self, X, y):
        """Recursively build the CART tree."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        # Re-map to 0-based ints for bincount
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_map[c] for c in y])
        self.root_ = self._build_tree(X, y_idx, depth=0)

    def _traverse(self, node, x):
        if node["leaf"]:
            return node["label"]
        if x[node["feat"]] <= node["thresh"]:
            return self._traverse(node["left"], x)
        return self._traverse(node["right"], x)

    def predict(self, X):
        """Traverse the tree for each sample and return leaf class."""
        X = np.asarray(X, dtype=float)
        idx = np.array([self._traverse(self.root_, row) for row in X])
        return self.classes_[idx]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))
