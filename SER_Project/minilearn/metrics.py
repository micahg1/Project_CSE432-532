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
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))


def confusion_matrix(y_true, y_pred):
    """
    Build a (C x C) confusion matrix where C = number of unique classes.
    Row = true label, Column = predicted label.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    idx = {c: i for i, c in enumerate(classes)}
    C = len(classes)
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def precision(y_true, y_pred, average="macro"):
    """
    Compute precision per class then aggregate.

    Parameters
    ----------
    average : 'macro' (unweighted mean) or 'weighted' (weighted by support)
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true)
    per_class, support = [], []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        per_class.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        support.append(np.sum(y_true == c))
    per_class, support = np.array(per_class), np.array(support)
    if average == "weighted":
        return np.average(per_class, weights=support)
    return per_class.mean()


def recall(y_true, y_pred, average="macro"):
    """
    Compute recall per class then aggregate.

    Parameters
    ----------
    average : 'macro' or 'weighted'
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true)
    per_class, support = [], []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        per_class.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        support.append(np.sum(y_true == c))
    per_class, support = np.array(per_class), np.array(support)
    if average == "weighted":
        return np.average(per_class, weights=support)
    return per_class.mean()


def f1_score(y_true, y_pred, average="macro"):
    """
    Harmonic mean of precision and recall: 2 * P * R / (P + R).

    Parameters
    ----------
    average : 'macro' or 'weighted'
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true)
    per_class, support = [], []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        support.append(np.sum(y_true == c))
    per_class, support = np.array(per_class), np.array(support)
    if average == "weighted":
        return np.average(per_class, weights=support)
    return per_class.mean()


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
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    classes = np.unique(y_true)
    aucs = []
    for i, c in enumerate(classes):
        binary = (y_true == c).astype(int)
        scores = y_score[:, i]
        # Sort by descending score to walk the ROC curve
        order = np.argsort(-scores)
        binary_sorted = binary[order]
        tps = np.cumsum(binary_sorted)
        fps = np.cumsum(1 - binary_sorted)
        tpr = tps / tps[-1] if tps[-1] > 0 else tps * 0.0
        fpr = fps / fps[-1] if fps[-1] > 0 else fps * 0.0
        # Prepend origin and compute trapezoidal area
        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])
        aucs.append(np.trapezoid(tpr, fpr))
    return float(np.mean(aucs))


def classification_report(y_true, y_pred, class_names=None):
    """
    Print a formatted table of per-class precision, recall, F1, and support —
    similar to sklearn.metrics.classification_report.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true)
    if class_names is None:
        class_names = [str(c) for c in classes]

    header = f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    lines = [header, "-" * len(header)]

    total_support = 0
    weighted_p, weighted_r, weighted_f = 0.0, 0.0, 0.0

    for c, name in zip(classes, class_names):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        sup = int(np.sum(y_true == c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        lines.append(f"{name:<15} {p:>10.4f} {r:>10.4f} {f:>10.4f} {sup:>10}")
        weighted_p += p * sup
        weighted_r += r * sup
        weighted_f += f * sup
        total_support += sup

    lines.append("-" * len(header))
    wp = weighted_p / total_support if total_support > 0 else 0.0
    wr = weighted_r / total_support if total_support > 0 else 0.0
    wf = weighted_f / total_support if total_support > 0 else 0.0
    lines.append(f"{'weighted avg':<15} {wp:>10.4f} {wr:>10.4f} {wf:>10.4f} {total_support:>10}")

    report = "\n".join(lines)
    print(report)
    return report
