"""
MiniLearn — A lightweight ML library built from scratch for CSE432/532.

This package mirrors the scikit-learn API (fit / predict / score) and implements
every required algorithm using only NumPy/SciPy.  Import examples:

    from minilearn.classifiers import LogisticRegression, KNearestNeighbors
    from minilearn.preprocessing import StandardScaler, train_test_split
    from minilearn.metrics import accuracy, confusion_matrix
    from minilearn.validation import StratifiedKFold
    from minilearn.clustering import KMeans
    from minilearn.decomposition import PCA
    from minilearn.ann import MLP
"""

from . import (
    preprocessing,
    classifiers,
    metrics,
    validation,
    clustering,
    decomposition,
    regression,
    ann,
)

__version__ = "0.1.0"
__all__ = [
    "preprocessing",
    "classifiers",
    "metrics",
    "validation",
    "clustering",
    "decomposition",
    "regression",
    "ann",
]
