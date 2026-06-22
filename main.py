"""
NLayerNN — A feedforward neural network of arbitrary depth, built from
scratch with NumPy only (no autograd, no frameworks).

Generalizes a fixed two-layer network into N layers, with:
  - per-layer activation functions (e.g. relu on hidden layers, sigmoid
    on the output layer)
  - full-batch vectorized forward/backward passes (not per-sample loops)
  - standard He/Xavier-style weight init
"""

import numpy as np


# ───────────────────────── activations ─────────────────────────
def sigmoid(z):
    z = np.clip(z, -500, 500)          # avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    return np.maximum(0, z)


def drelu(z):
    return (z > 0).astype(z.dtype)


def tanh(z):
    return np.tanh(z)


def dtanh(z):
    return 1 - np.tanh(z) ** 2


def linear(z):
    return z


def dlinear(z):
    return np.ones_like(z)


ACTIVATIONS = {
    "sigmoid": (sigmoid, dsigmoid),
    "relu": (relu, drelu),
    "tanh": (tanh, dtanh),
    "linear": (linear, dlinear),
}


# ───────────────────────── the network ─────────────────────────
class NLayerNN:
    """
    Feedforward neural network with an arbitrary number of layers.

    Parameters
    ----------
    layer_sizes : list[int]
        e.g. [2, 4, 4, 1] = 2 inputs -> hidden(4) -> hidden(4) -> 1 output.
        len(layer_sizes) - 1 is the number of weight matrices (layers).
    activations : list[str] or None
        One activation name per layer (length == len(layer_sizes) - 1).
        Choices: "relu", "sigmoid", "tanh", "linear".
        Defaults to relu on every hidden layer and sigmoid on the output.
    seed : int or None
        Optional RNG seed for reproducible weight init.
    """

    def __init__(self, layer_sizes, activations=None, seed=None):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least an input and an output layer.")

        self.layer_sizes = list(layer_sizes)
        self.n_layers = len(layer_sizes) - 1  # number of weight matrices

        if activations is None:
            activations = ["relu"] * (self.n_layers - 1) + ["sigmoid"]
        if len(activations) != self.n_layers:
            raise ValueError(
                f"Need {self.n_layers} activations, got {len(activations)}."
            )
        self.activation_names = activations
        self.acts = [ACTIVATIONS[a][0] for a in activations]
        self.dacts = [ACTIVATIONS[a][1] for a in activations]

        rng = np.random.default_rng(seed)
        self.weights = []   # W[l] : (layer_sizes[l+1], layer_sizes[l])
        self.biases = []    # b[l] : (layer_sizes[l+1],)

        for l in range(self.n_layers):
            n_in, n_out = self.layer_sizes[l], self.layer_sizes[l + 1]
            # He init for relu, Xavier-ish otherwise — both are fine
            # general-purpose defaults and avoid vanishing/exploding
            # activations as depth grows.
            if activations[l] == "relu":
                scale = np.sqrt(2.0 / n_in)
            else:
                scale = np.sqrt(1.0 / n_in)
            self.weights.append(rng.standard_normal((n_out, n_in)) * scale)
            self.biases.append(np.zeros(n_out))

        # caches populated during forward(), used by backward()
        self._z = [None] * self.n_layers   # pre-activations
        self._a = [None] * self.n_layers   # post-activations

    # ───────────────────── forward pass ─────────────────────
    def forward(self, X):
        """
        X : (n_samples, n_features)
        Returns the network output, shape (n_samples, n_output).
        Caches every z (pre-activation) and a (post-activation) for backprop.
        """
        a = X
        self._a0 = X  # input "activation" for layer 0's weight gradient
        for l in range(self.n_layers):
            z = a @ self.weights[l].T + self.biases[l]
            a = self.acts[l](z)
            self._z[l] = z
            self._a[l] = a
        return a

    # ───────────────────── backward pass ─────────────────────
    def backward(self, X, y, y_pred, lr):
        """
        One full-batch gradient-descent update using mean-squared error
        loss: L = mean((y_pred - y)^2).

        X      : (n_samples, n_features)
        y      : (n_samples, n_output)
        y_pred : (n_samples, n_output)  — output of forward(X)
        """
        n_samples = X.shape[0]

        # dL/da at the output layer
        delta = (2.0 / n_samples) * (y_pred - y)  # (n_samples, n_output)

        for l in reversed(range(self.n_layers)):
            z = self._z[l]
            delta = delta * self.dacts[l](z)        # dL/dz, shape (n_samples, n_out_l)

            a_prev = self._a0 if l == 0 else self._a[l - 1]  # (n_samples, n_in_l)

            dW = delta.T @ a_prev                   # (n_out_l, n_in_l)
            db = delta.sum(axis=0)                   # (n_out_l,)

            # propagate error to the previous layer BEFORE overwriting
            # this layer's weights with the update
            if l > 0:
                delta = delta @ self.weights[l]      # (n_samples, n_in_l)

            self.weights[l] -= lr * dW
            self.biases[l] -= lr * db

    # ───────────────────── training loop ─────────────────────
    def fit(self, X, y, lr=0.01, epochs=1000, verbose_every=100):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]  # ensure (n_samples, n_output)

        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = float(np.mean((y_pred - y) ** 2))
            losses.append(loss)

            self.backward(X, y, y_pred, lr)

            if verbose_every and epoch % verbose_every == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

        return losses

    # ───────────────────── inference ─────────────────────
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.forward(X)


# ───────────────────────── one-layer ANN ─────────────────────────
class OneLayerNN(NLayerNN):
    """
    Single-layer network (no hidden layer): input -> output.
    Equivalent to logistic/linear regression depending on activation.

    Parameters
    ----------
    n_features : int
    n_output   : int, default 1
    activation : str, default "sigmoid"
    seed       : int or None
    """

    def __init__(self, n_features, n_output=1, activation="sigmoid", seed=None):
        super().__init__(
            layer_sizes=[n_features, n_output],
            activations=[activation],
            seed=seed,
        )


# ───────────────────────── two-layer ANN ─────────────────────────
class TwoLayerNN(NLayerNN):
    """
    Two-layer network: input -> hidden -> output.
    The classic shallow MLP (this is what the original TwoNN generalizes).

    Parameters
    ----------
    n_features        : int
    n_hidden           : int
    n_output           : int, default 1
    hidden_activation  : str, default "relu"
    output_activation  : str, default "sigmoid"
    seed                : int or None
    """

    def __init__(
        self,
        n_features,
        n_hidden,
        n_output=1,
        hidden_activation="relu",
        output_activation="sigmoid",
        seed=None,
    ):
        super().__init__(
            layer_sizes=[n_features, n_hidden, n_output],
            activations=[hidden_activation, output_activation],
            seed=seed,
        )


# ───────────────────────── demo: XOR ─────────────────────────
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)

    # ── One-layer ANN (no hidden layer) ──────────────────────────
    # XOR is NOT linearly separable, so a single layer (no hidden
    # units) cannot solve it — this is expected to fail/plateau.
    # Included to show the limitation, not as a working XOR solver.
    print("=" * 50)
    print("OneLayerNN (input -> output, no hidden layer)")
    print("=" * 50)
    one_layer = OneLayerNN(n_features=2, n_output=1, activation="sigmoid", seed=42)
    one_layer.fit(X, y, lr=0.5, epochs=5000, verbose_every=1000)
    print("\nPredictions:", one_layer.predict(X).ravel())
    print("Rounded:    ", np.round(one_layer.predict(X).ravel()))
    print("(expected to NOT match [0 1 1 0] — XOR needs a hidden layer)\n")

    # ── Two-layer ANN (one hidden layer) ─────────────────────────
    print("=" * 50)
    print("TwoLayerNN (input -> hidden -> output)")
    print("=" * 50)
    two_layer = TwoLayerNN(
        n_features=2,
        n_hidden=6,
        n_output=1,
        hidden_activation="relu",
        output_activation="sigmoid",
        seed=42,
    )
    two_layer.fit(X, y, lr=0.1, epochs=5000, verbose_every=1000)
    print("\nPredictions:", two_layer.predict(X).ravel())
    print("Rounded:    ", np.round(two_layer.predict(X).ravel()))
    print()

    # ── N-layer ANN (two hidden layers) ──────────────────────────
    print("=" * 50)
    print("NLayerNN (input -> hidden -> hidden -> output)")
    print("=" * 50)
    net = NLayerNN(
        layer_sizes=[2, 4, 4, 1],
        activations=["relu", "relu", "sigmoid"],
        seed=42,
    )
    net.fit(X, y, lr=0.1, epochs=5000, verbose_every=1000)
    print("\nPredictions:", net.predict(X).ravel())
    print("Rounded:    ", np.round(net.predict(X).ravel()))
