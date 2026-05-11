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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _distances_to_centers(self, X, centers):
        """Return (n_samples, k) matrix of squared Euclidean distances."""
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c^T  (faster than broadcasting)
        X2 = np.sum(X ** 2, axis=1, keepdims=True)       # (n, 1)
        C2 = np.sum(centers ** 2, axis=1)                 # (k,)
        return X2 + C2 - 2.0 * (X @ centers.T)           # (n, k)

    def _assign(self, X, centers):
        """Return cluster labels: argmin distance to each centroid."""
        return np.argmin(self._distances_to_centers(X, centers), axis=1)

    def _inertia(self, X, labels, centers):
        """Sum of squared distances from each point to its assigned centroid."""
        dists = self._distances_to_centers(X, centers)
        return float(np.sum(dists[np.arange(len(X)), labels]))

    def _single_run(self, X, rng):
        """One full Lloyd's run from a random initialisation."""
        n_samples = X.shape[0]

        # Step 1 — initialise: pick k distinct samples as centroids
        init_idx = rng.choice(n_samples, size=self.k, replace=False)
        centers  = X[init_idx].copy()

        for _ in range(self.max_iter):
            # Step 2 — assign
            labels = self._assign(X, centers)

            # Step 3 — recompute centroids (handle empty clusters by keeping old center)
            new_centers = centers.copy()
            for c in range(self.k):
                members = X[labels == c]
                if len(members) > 0:
                    new_centers[c] = members.mean(axis=0)

            # Step 4 — check convergence
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break

        labels  = self._assign(X, centers)
        inertia = self._inertia(X, labels, centers)
        return centers, labels, inertia

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X):
        """
        Run Lloyd's algorithm n_init times and keep the run with lowest inertia.
        """
        X   = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        best_centers = best_labels = best_inertia = None

        for _ in range(self.n_init):
            centers, labels, inertia = self._single_run(X, rng)
            if best_inertia is None or inertia < best_inertia:
                best_centers = centers
                best_labels  = labels
                best_inertia = inertia

        self.cluster_centers_ = best_centers
        self.labels_          = best_labels
        self.inertia_         = best_inertia
        return self

    def predict(self, X):
        """Assign each sample in X to its nearest stored centroid."""
        X = np.asarray(X, dtype=float)
        return self._assign(X, self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
