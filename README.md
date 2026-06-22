# NLayerNN — Feedforward Neural Network from Scratch (NumPy)

A feedforward neural network of **arbitrary depth**, built using only
`numpy` — no autograd, no frameworks. Forward pass, backpropagation, and
gradient descent are all written out by hand. Includes `OneLayerNN` and
`TwoLayerNN` as convenience subclasses for the common shallow cases.

## Architecture

```
X  →  [W₁, b₁] → act₁  →  [W₂, b₂] → act₂  →  ...  →  [Wₙ, bₙ] → actₙ  →  output
```

- `layer_sizes` defines the shape, e.g. `[2, 4, 4, 1]` = 2 inputs → hidden(4)
  → hidden(4) → 1 output. `len(layer_sizes) - 1` is the number of weight
  matrices.
- Each layer has its **own activation function** — e.g. `relu` on every
  hidden layer, `sigmoid` on the output (the default), rather than one
  global activation forced on every layer.
- The whole batch is processed at once (`X` is `(n_samples, n_features)`)
  — no per-sample Python loop.

**Forward pass** (vectorized over the batch):

```
Z[l] = A[l-1] @ W[l].T + b[l]
A[l] = activation[l](Z[l])
```

**Backward pass** (mean-squared-error loss, full-batch gradient descent):

```
delta = (2/n) * (y_pred - y)                      # at the output layer
for l from last layer to first:
    delta   = delta * dactivation[l](Z[l])
    dW[l]   = delta.T @ A[l-1]
    db[l]   = sum(delta, axis=0)
    delta   = delta @ W[l]                         # propagate to layer l-1
    W[l]   -= lr * dW[l]
    b[l]   -= lr * db[l]
```

Weights are initialized with He scaling (`sqrt(2/n_in)`) for relu layers
and Xavier-style scaling (`sqrt(1/n_in)`) otherwise, which keeps
activations well-behaved as depth increases.

## Classes

### `NLayerNN(layer_sizes, activations=None, seed=None)`
The general N-layer network. `activations` is a list of strings, one per
layer, chosen from `"relu"`, `"sigmoid"`, `"tanh"`, `"linear"`. If omitted,
defaults to relu on all hidden layers and sigmoid on the output.

```python
net = NLayerNN(layer_sizes=[2, 4, 4, 1], activations=["relu", "relu", "sigmoid"], seed=42)
net.fit(X, y, lr=0.1, epochs=5000)
net.predict(X)
```

### `OneLayerNN(n_features, n_output=1, activation="sigmoid", seed=None)`
No hidden layer — input straight to output. Equivalent to logistic/linear
regression depending on the activation. A thin subclass of `NLayerNN`
with `layer_sizes=[n_features, n_output]`.

```python
net = OneLayerNN(n_features=2, activation="sigmoid", seed=42)
net.fit(X, y, lr=0.5, epochs=5000)
```

**Note:** a one-layer network has no hidden units, so it can only learn a
linear decision boundary. It **cannot** solve XOR — see "Demo output"
below for what that failure actually looks like.

### `TwoLayerNN(n_features, n_hidden, n_output=1, hidden_activation="relu", output_activation="sigmoid", seed=None)`
The classic shallow MLP: input → hidden → output. A thin subclass of
`NLayerNN` with `layer_sizes=[n_features, n_hidden, n_output]`.

```python
net = TwoLayerNN(n_features=2, n_hidden=6, seed=42)
net.fit(X, y, lr=0.1, epochs=5000)
```

### Shared methods (all three classes)
- **`.fit(X, y, lr=0.01, epochs=1000, verbose_every=100)`** — trains with
  full-batch gradient descent. Prints loss every `verbose_every` epochs
  (set to `0`/`None` to silence). Returns the per-epoch loss history.
- **`.predict(X)`** — runs the forward pass, returns predictions shaped
  `(n_samples, n_output)`.

## Usage

```bash
python main.py
```

Running the file directly trains all three (`OneLayerNN`, `TwoLayerNN`,
`NLayerNN`) on the XOR problem and prints results for each.

## Demo output (XOR)

```
==================================================
OneLayerNN (input -> output, no hidden layer)
==================================================
...
Predictions: [0.5 0.5 0.5 0.5]
Rounded:     [0. 0. 0. 0.]
(expected to NOT match [0 1 1 0] — XOR needs a hidden layer)

==================================================
TwoLayerNN (input -> hidden -> output)
==================================================
...
Predictions: [0.066 0.964 0.973 0.027]
Rounded:     [0. 1. 1. 0.]

==================================================
NLayerNN (input -> hidden -> hidden -> output)
==================================================
...
Predictions: [0.013 0.961 0.961 0.013]
Rounded:     [0. 1. 1. 0.]
```

`OneLayerNN` plateauing at `0.5` for every input is the expected, correct
behavior — not a bug. XOR isn't linearly separable, so a network with no
hidden layer mathematically cannot represent its decision boundary,
regardless of training time. It's kept in the demo specifically to
illustrate that limitation.

## Notes & limitations

- **Full-batch gradient descent only** — no mini-batching, no momentum,
  no Adam/RMSProp. Good for understanding the mechanics; for larger
  datasets you'd want a library (PyTorch/TensorFlow) or at least
  mini-batch SGD with a modern optimizer.
- **Shallow nets can hit bad local minima.** With `relu`, a `TwoLayerNN`
  with too few hidden units (e.g. 4, on XOR) can converge to a plateau
  where two of the four XOR points collapse to the same prediction. Use
  `verbose_every` to watch the loss curve, and widen the hidden layer or
  try a different seed if training stalls above the expected loss.
- **No regularization, dropout, or batch norm** — this is a from-scratch
  teaching implementation, not a production model.
- **MSE loss only**, regardless of whether the task is regression or
  classification. For classification, MSE with a sigmoid output still
  works, but cross-entropy is the more standard choice and converges
  faster in practice.
