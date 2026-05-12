# MiniLearn — API Reference

A from-scratch ML library that mirrors the scikit-learn API, built for CSE432/532.
All algorithms are implemented using only NumPy.

## Installation

MiniLearn is a local package — no pip install needed. Just ensure `SER_Project/` is on
your path (each notebook does this via `sys.path.insert(0, os.path.abspath('..'))`).

## Subpackages

### `minilearn.classifiers`

```python
from minilearn.classifiers import (
    LogisticRegression,   # Softmax multi-class via gradient descent
    KNearestNeighbors,    # Lazy learner, Euclidean or Manhattan distance
    GaussianNaiveBayes,   # Gaussian likelihood + Bayes' theorem
    LinearSVM,            # Hinge-loss subgradient, binary classification
    DecisionTree,         # CART with Gini or entropy splitting
)
```

All classifiers share the same interface:

```python
clf.fit(X_train, y_train)
clf.predict(X_test)          # class labels
clf.predict_proba(X_test)    # probability estimates (where supported)
clf.score(X_test, y_test)    # accuracy
```

### `minilearn.preprocessing`

```python
from minilearn.preprocessing import StandardScaler, train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=True
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only!
X_test_scaled  = scaler.transform(X_test)        # never re-fit on test
```

### `minilearn.metrics`

```python
from minilearn.metrics import (
    accuracy,               # fraction correct
    precision,              # TP / (TP + FP), macro or weighted
    recall,                 # TP / (TP + FN), macro or weighted
    f1_score,               # harmonic mean of precision and recall
    confusion_matrix,       # (C x C) array, row=true / col=predicted
    roc_auc_score,          # macro-averaged AUC, One-vs-Rest
    classification_report,  # formatted per-class table
)
```

### `minilearn.model_selection`

```python
from minilearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

# Manual k-fold loop
kfold = StratifiedKFold(n_splits=5, random_state=42)
for train_idx, val_idx in kfold.split(X, y):
    ...

# Convenience wrapper
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(scores.mean(), '±', scores.std())

# Hyperparameter search
gs = GridSearchCV(clf, param_grid={'lr': [0.001, 0.01], 'n_iter': [500, 1000]}, cv=5)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
```

### `minilearn.clustering`

```python
from minilearn.clustering import KMeans

km = KMeans(k=8, max_iter=300, n_init=10, random_state=42)
km.fit(X_pca)
labels  = km.labels_           # cluster assignment per sample
centers = km.cluster_centers_  # final centroid coordinates
```

### `minilearn.decomposition`

```python
from minilearn.decomposition import PCA

pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)   # fit on train, transform train
X_test_pca  = pca.transform(X_test)        # transform test with same axes

print(pca.explained_variance_ratio_.cumsum())  # cumulative variance explained
```

### `minilearn.regression`

```python
from minilearn.regression import LinearRegression, RidgeRegression, mse, r2_score

reg = LinearRegression(solver='analytic')   # or 'gradient_descent'
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(r2_score(y_test, y_pred))

ridge = RidgeRegression(alpha=1.0)
ridge.fit(X_train, y_train)
```

### `minilearn.ann`

```python
from minilearn.ann import MLP

mlp = MLP(
    layer_sizes=[X_train.shape[1], 256, 128, 8],
    activation='relu',
    lr=0.001,
    n_epochs=50,
    batch_size=64,
    l2_lambda=1e-4,
)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print(mlp.score(X_test, y_test))

import matplotlib.pyplot as plt
plt.plot(mlp.loss_history_)   # training loss curve
```

## Design Notes

- **No data leakage:** always `fit` scalers and PCA on training data only, then `transform` test data separately.
- **sklearn compatibility:** all estimators follow the `fit` / `predict` / `score` convention so they can be swapped with sklearn equivalents for comparison.
- **Reproducibility:** pass `random_state=42` consistently across all estimators and splits.
