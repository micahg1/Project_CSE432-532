"""
minilearn/metrics.py
====================
Evaluation metrics implemented from scratch using NumPy.

All functions accept 1-D arrays of true labels (y_true) and predicted
labels (y_pred).  Multi-class support is handled via macro/weighted
averaging where applicable.

Functions
---------
accuracy(y_true, y_pred)
    Fraction of correctly classified samples.

precision(y_true, y_pred, average)
    TP / (TP + FP) per class; averaged with 'macro' or 'weighted'.

recall(y_true, y_pred, average)
    TP / (TP + FN) per class; averaged with 'macro' or 'weighted'.

f1_score(y_true, y_pred, average)
    Harmonic mean of precision and recall.

confusion_matrix(y_true, y_pred)
    Returns an (n_classes × n_classes) integer array where entry [i, j]
    is the count of samples with true label i predicted as label j.

roc_auc_score(y_true, y_score)
    Area under the ROC curve (One-vs-Rest for multi-class).

Usage
-----
    from minilearn.metrics import accuracy, f1_score, confusion_matrix

    print(accuracy(y_test, y_pred))
    print(f1_score(y_test, y_pred, average='macro'))
    cm = confusion_matrix(y_test, y_pred)
"""

import numpy as np


def accuracy(y_true, y_pred):
    """Return fraction of predictions that match the true labels."""
    # TODO: implement
    raise NotImplementedError


def confusion_matrix(y_true, y_pred):
    """
    Build a (C x C) confusion matrix where C = number of unique classes.
    Row = true label, Column = predicted label.
    """
    # TODO: implement — use np.unique to discover classes
    raise NotImplementedError


def precision(y_true, y_pred, average="macro"):
    """
    Compute precision per class then aggregate.

    Parameters
    ----------
    average : 'macro' (unweighted mean) or 'weighted' (weighted by support)
    """
    # TODO: iterate over classes, compute TP/(TP+FP) per class, then average
    raise NotImplementedError


def recall(y_true, y_pred, average="macro"):
    """
    Compute recall per class then aggregate.

    Parameters
    ----------
    average : 'macro' or 'weighted'
    """
    # TODO: iterate over classes, compute TP/(TP+FN) per class, then average
    raise NotImplementedError


def f1_score(y_true, y_pred, average="macro"):
    """
    Harmonic mean of precision and recall: 2 * P * R / (P + R).

    Parameters
    ----------
    average : 'macro' or 'weighted'
    """
    # TODO: call precision() and recall() then combine
    raise NotImplementedError


def roc_auc_score(y_true, y_score):
    """
    Compute the macro-averaged AUC-ROC using a One-vs-Rest strategy.

    Parameters
    ----------
    y_true  : 1-D array of integer class labels
    y_score : 2-D array of shape (n_samples, n_classes) — probability estimates

    Returns
    -------
    float : macro-averaged AUC
    """
    # TODO: for each class binarize y_true, compute trapezoidal AUC, then average
    raise NotImplementedError


def classification_report(y_true, y_pred, class_names=None):
    """
    Print a formatted table of per-class precision, recall, F1, and support —
    similar to sklearn.metrics.classification_report.
    """
    # TODO: build string table using precision(), recall(), f1_score() per class
    raise NotImplementedError
