"""
minilearn/clustering.py
=======================
Unsupervised clustering algorithms implemented from scratch.

Classes
-------
KMeans
    Lloyd's algorithm: randomly initialize k centroids, then alternate between
    assigning each point to its nearest centroid and recomputing centroids until
    convergence (or max_iter is reached).

    Evaluation is done externally with minilearn.metrics or sklearn's ARI/NMI
    because clustering is unsupervised — there are no "correct" labels to score
    against during training.

Usage
-----
    from minilearn.clustering import KMeans

    km = KMeans(k=8, max_iter=300, random_state=42)
    km.fit(X_pca)                        # fit on PCA-reduced features
    labels = km.labels_                  # cluster assignment per sample
    centers = km.cluster_centers_        # final centroid coordinates

    # Evaluate against ground-truth emotion labels
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    print(adjusted_rand_score(y_true, labels))
    print(normalized_mutual_info_score(y_true, labels))
"""

import numpy as np


class KMeans:
    """
    K-Means clustering using Lloyd's algorithm.

    Parameters
    ----------
    k            : int, number of clusters (set to 8 to match RAVDESS emotions)
    max_iter     : int, maximum number of Lloyd iterations
    tol          : float, convergence tolerance (centroid shift)
    n_init       : int, number of random restarts — keep the best (lowest inertia)
    random_state : int or None
    """

    def __init__(self, k=8, max_iter=300, tol=1e-4, n_init=10, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        """
        Run Lloyd's algorithm n_init times and keep the run with lowest inertia.

        Steps per run:
        1. Randomly pick k samples as initial centroids.
        2. Assign each point to nearest centroid (Euclidean distance).
        3. Recompute centroids as mean of assigned points.
        4. Repeat 2–3 until centroid shift < tol or max_iter reached.
        """
        # TODO: implement _single_run helper; loop n_init times; store best result
        raise NotImplementedError

    def predict(self, X):
        """Assign each sample in X to its nearest stored centroid."""
        # TODO: compute distances to self.cluster_centers_, return argmin
        raise NotImplementedError

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
