"""
minilearn/ann.py
================
Artificial Neural Network implemented from scratch using NumPy.

This module satisfies the Week 13–14 MiniLearn requirement: a from-scratch
ANN with at least one hidden layer trained via backpropagation.

Classes
-------
MLP (Multi-Layer Perceptron)
    A fully-connected feedforward network with configurable hidden layers,
    activation functions, and a softmax output layer for multi-class
    classification.

    Architecture example (for 8-class SER):
        Input (n_features) → Dense(256, ReLU) → Dense(128, ReLU) → Softmax(8)

    Training uses mini-batch stochastic gradient descent with cross-entropy loss
    and optional L2 weight decay.

Usage
-----
    from minilearn.ann import MLP

    # Build a 2-hidden-layer network
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

    # Plot training loss curve
    import matplotlib.pyplot as plt
    plt.plot(mlp.loss_history_)
    plt.xlabel('Epoch'); plt.ylabel('Cross-entropy loss')
    plt.title('MLP Training Curve')
    plt.show()

Notes
-----
- For more powerful DL models (1D-CNN, LSTM) use TensorFlow/Keras or PyTorch
  in the notebooks — those do NOT need to be re-implemented from scratch.
- This MLP is the minimum required scratch implementation.
"""

import numpy as np


class MLP:
    """
    Multi-Layer Perceptron for multi-class classification.

    Parameters
    ----------
    layer_sizes : list of int
        Sizes of each layer including input and output.
        E.g. [128, 256, 64, 8] for input_dim=128, two hidden layers, 8 classes.
    activation  : str, 'relu', 'tanh', or 'sigmoid' (applied to hidden layers)
    lr          : float, learning rate
    n_epochs    : int, number of full passes over training data
    batch_size  : int, mini-batch size
    l2_lambda   : float, L2 regularization coefficient (0 = no regularization)
    random_state: int or None
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
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.random_state = random_state

        self.weights_ = []    # list of weight matrices per layer
        self.biases_ = []     # list of bias vectors per layer
        self.loss_history_ = []   # cross-entropy loss recorded each epoch

    # ── Activation functions ────────────────────────────────────────────────

    def _activation(self, Z):
        """Apply the chosen hidden-layer activation to pre-activation Z."""
        # TODO: implement relu, tanh, sigmoid branches
        raise NotImplementedError

    def _activation_grad(self, Z):
        """Derivative of the activation function at Z (for backprop)."""
        # TODO: implement derivative for each activation
        raise NotImplementedError

    @staticmethod
    def _softmax(Z):
        """Numerically stable softmax: subtract row-max before exp."""
        # TODO: implement
        raise NotImplementedError

    # ── Weight initialization ───────────────────────────────────────────────

    def _initialize_weights(self):
        """
        He initialization for ReLU (or Xavier for tanh/sigmoid).
        Populates self.weights_ and self.biases_.
        """
        # TODO: for each consecutive pair in layer_sizes, create W and b
        raise NotImplementedError

    # ── Forward pass ────────────────────────────────────────────────────────

    def _forward(self, X):
        """
        Compute activations for all layers.

        Returns
        -------
        activations : list of np.ndarray, one per layer (including input)
        pre_activations : list of np.ndarray Z = W·A + b (one per hidden/output layer)
        """
        # TODO: propagate X through each layer, store A and Z for backprop
        raise NotImplementedError

    # ── Backward pass ───────────────────────────────────────────────────────

    def _backward(self, activations, pre_activations, y_one_hot):
        """
        Backpropagate gradients using the chain rule.

        Returns
        -------
        grad_W : list of weight gradients (same shape as self.weights_)
        grad_b : list of bias gradients (same shape as self.biases_)
        """
        # TODO: compute output delta (softmax - y_one_hot), then propagate back
        #       through each layer using stored Z and W.
        raise NotImplementedError

    # ── Training ────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Train the MLP using mini-batch SGD with cross-entropy loss.

        Steps each epoch:
        1. Shuffle training data.
        2. Split into mini-batches.
        3. Forward pass → compute loss.
        4. Backward pass → compute gradients.
        5. Update weights: W -= lr * (grad_W + l2_lambda * W)
        6. Record mean epoch loss in self.loss_history_.
        """
        # TODO: implement full training loop
        raise NotImplementedError

    # ── Inference ───────────────────────────────────────────────────────────

    def predict_proba(self, X):
        """Return softmax output probabilities, shape (n_samples, n_classes)."""
        # TODO: run forward pass, return final activation
        raise NotImplementedError

    def predict(self, X):
        """Return predicted class index (argmax of probabilities)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
