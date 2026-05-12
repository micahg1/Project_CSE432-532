"""
minilearn/ann.py
================
Artificial Neural Network implemented from scratch using NumPy.

Classes
-------
MLP (Multi-Layer Perceptron)
    Fully-connected feedforward network with configurable hidden layers,
    activation functions, and a softmax output layer for multi-class
    classification.  Trained via mini-batch SGD + backpropagation.

Usage
-----
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

    plt.plot(mlp.loss_history_)
"""

import numpy as np


class MLP:
    """
    Multi-Layer Perceptron for multi-class classification.

    Parameters
    ----------
    layer_sizes  : list of int — sizes of each layer including input and output.
                   E.g. [112, 256, 128, 8] for 112-dim input, 2 hidden layers, 8 classes.
    activation   : str, 'relu', 'tanh', or 'sigmoid' (applied to hidden layers)
    lr           : float, learning rate
    n_epochs     : int, number of full passes over training data
    batch_size   : int, mini-batch size
    l2_lambda    : float, L2 regularization coefficient
    random_state : int or None
    """

    def __init__(
        self,
        layer_sizes,
        activation="relu",
        lr=0.001,
        n_epochs=50,
        batch_size=64,
        l2_lambda=1e-4,
        random_state=None,
    ):
        self.layer_sizes  = layer_sizes
        self.activation   = activation
        self.lr           = lr
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.l2_lambda    = l2_lambda
        self.random_state = random_state

        self.weights_      = []
        self.biases_       = []
        self.loss_history_ = []
        self.classes_      = None

    # ── Activation functions ──────────────────────────────────────────────────

    def _activation(self, Z):
        if self.activation == "relu":
            return np.maximum(0.0, Z)
        elif self.activation == "tanh":
            return np.tanh(Z)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
        raise ValueError(f"Unknown activation '{self.activation}'")

    def _activation_grad(self, Z):
        if self.activation == "relu":
            return (Z > 0).astype(float)
        elif self.activation == "tanh":
            return 1.0 - np.tanh(Z) ** 2
        elif self.activation == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
            return s * (1.0 - s)
        raise ValueError(f"Unknown activation '{self.activation}'")

    @staticmethod
    def _softmax(Z):
        shifted = Z - Z.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    # ── Weight initialization ─────────────────────────────────────────────────

    def _initialize_weights(self):
        """
        He initialization for ReLU (scale = sqrt(2/n_in)).
        Xavier for tanh/sigmoid (scale = sqrt(1/n_in)).
        """
        rng = np.random.default_rng(self.random_state)
        self.weights_, self.biases_ = [], []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            scale = np.sqrt(2.0 / n_in) if self.activation == "relu" else np.sqrt(1.0 / n_in)
            self.weights_.append(rng.normal(0.0, scale, (n_in, n_out)))
            self.biases_.append(np.zeros(n_out))

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _forward(self, X):
        """
        Propagate X through the network.

        Returns
        -------
        activations     : list[ndarray] — one per layer including input (length = n_layers+1)
        pre_activations : list[ndarray] — Z = A·W + b, one per weight layer
        """
        activations     = [X]
        pre_activations = []
        A = X
        n_hidden = len(self.weights_) - 1
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            Z = A @ W + b
            pre_activations.append(Z)
            A = self._activation(Z) if i < n_hidden else self._softmax(Z)
            activations.append(A)
        return activations, pre_activations

    # ── Backward pass ─────────────────────────────────────────────────────────

    def _backward(self, activations, pre_activations, Y_one_hot):
        """
        Backpropagate cross-entropy + softmax gradients.

        Returns grad_W and grad_b as lists (same order as self.weights_).
        """
        n = Y_one_hot.shape[0]
        n_layers = len(self.weights_)
        grad_W = [None] * n_layers
        grad_b = [None] * n_layers

        # Output delta: derivative of cross-entropy through softmax = probs - y
        delta = activations[-1] - Y_one_hot          # (n, n_out)

        for i in reversed(range(n_layers)):
            A_prev = activations[i]                  # (n, n_in)
            grad_W[i] = (A_prev.T @ delta) / n + self.l2_lambda * self.weights_[i]
            grad_b[i] = delta.mean(axis=0)
            if i > 0:
                delta = (delta @ self.weights_[i].T) * self._activation_grad(pre_activations[i - 1])

        return grad_W, grad_b

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Train via mini-batch SGD with cross-entropy loss."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Map arbitrary labels → 0-based indices
        self.classes_    = np.unique(y)
        label_to_idx     = {c: i for i, c in enumerate(self.classes_)}
        y_idx            = np.array([label_to_idx[c] for c in y])
        n_samples, n_cls = X.shape[0], len(self.classes_)

        # One-hot targets
        Y = np.zeros((n_samples, n_cls))
        Y[np.arange(n_samples), y_idx] = 1.0

        self._initialize_weights()
        self.loss_history_ = []
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_epochs):
            perm   = rng.permutation(n_samples)
            X_s, Y_s = X[perm], Y[perm]
            batch_losses = []

            for start in range(0, n_samples, self.batch_size):
                Xb = X_s[start : start + self.batch_size]
                Yb = Y_s[start : start + self.batch_size]

                acts, pre_acts = self._forward(Xb)
                loss = -np.mean(np.sum(Yb * np.log(acts[-1] + 1e-15), axis=1))
                batch_losses.append(loss)

                grad_W, grad_b = self._backward(acts, pre_acts, Yb)
                for i in range(len(self.weights_)):
                    self.weights_[i] -= self.lr * grad_W[i]
                    self.biases_[i]  -= self.lr * grad_b[i]

            self.loss_history_.append(float(np.mean(batch_losses)))

        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X):
        """Return softmax probabilities, shape (n_samples, n_classes)."""
        acts, _ = self._forward(np.asarray(X, dtype=float))
        return acts[-1]

    def predict(self, X):
        """Return predicted class labels (mapped back to original label space)."""
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
