# TwoNN — A Two-Layer Neural Network from Scratch (NumPy)

A minimal feedforward neural network with one hidden layer, implemented using
only `numpy`. No frameworks, no autograd — forward pass, backward pass, and
gradient descent are all written out by hand. The included example trains it
to solve XOR.

## How it works

**Architecture**

```
input x  →  hidden layer (wx, bx)  →  activation  →  output layer (wy, by)  →  activation  →  ay
```

- `wx`, `bx` — weights/bias for the hidden layer (`n_hidden x n_features`)
- `wy`, `by` — weights/bias for the output layer (`1 x n_hidden`)
- `activation` — applied after both layers (same function for both, by design)
- `dact` — derivative of the activation, used in backprop. If not supplied,
  it's estimated numerically with a finite difference.

**Forward pass** (`forward`)

```
zx = wx @ x + bx        # hidden pre-activation
ax = activation(zx)     # hidden activation
zy = wy @ ax + by        # output pre-activation
ay = activation(zy)      # output activation
```

**Backward pass** (`fit`)

Per-sample SGD with squared error loss `(y_pred - y)^2`:

1. `delta_out = 2 * err * dact(zy)` — output layer gradient
2. Update `wy`, `by` using `delta_out` and the hidden activation `ax`
3. Backpropagate: `d_hidden = (wy^T @ delta_out) * dact(zx)`
4. Update `wx`, `bx` using `d_hidden` and the raw input `x`

Weights are initialized as small random values (`* 0.1`), not identity —
identity init would make the hidden layer redundant at the start of training.

## Bug fixed

The original `forward()` computed `ax` from `zx` *before* `zx` was assigned:

```python
self.ax = self.activation(self.zx)   # zx doesn't exist yet
self.zx = self.wx @ x + self.bx
```

This raised `AttributeError: 'TwoNN' object has no attribute 'zx'` on the very
first call. Fixed by reordering so `zx` is computed first:

```python
self.zx = self.wx @ x + self.bx
self.ax = self.activation(self.zx)
self.zy = self.wy @ self.ax + self.by
self.ay = self.activation(self.zy)
```

## Usage

```python
import numpy as np
from main import TwoNN

# z**2 activation with analytical derivative 2z — works for XOR here
# because it's nonlinear; note it's an unusual choice (not bounded,
# not monotonic) — sigmoid/tanh/relu would be more typical picks.
model = TwoNN(activation=lambda z: z**2, dact=lambda z: 2*z)

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model.fit(x, y, lr=0.01, epochs=10000)
print(model.predict(x))
# ≈ [0, 1, 1, 0]
```

Run directly:

```bash
python main.py
```

Expected output (abbreviated):

```
Epoch    0 | Loss: 0.2...
...
Epoch 9900 | Loss: 0.000000
[0.0012 0.9999 0.9999 0.0006]
```

## API

### `TwoNN(activation=lambda x: x, dact=None)`
- `activation` — function applied after each linear layer.
- `dact` — derivative of `activation`. If omitted, estimated numerically
  via `(activation(z + 1e-4) - activation(z)) / 1e-4`. Supplying the exact
  derivative is faster and more numerically stable.

### `.fit(x, y, lr=0.001, epochs=1000)`
Trains with plain (non-batched) SGD — one sample at a time, weights updated
every sample. Hidden layer size is fixed to `n_features` (no separate
hyperparameter for hidden width). Prints loss every 100 epochs.

### `.predict(x)`
Runs the forward pass on each row of `x`, returns predictions as an array.

## Notes & limitations

- **Hidden size is tied to input size.** `n_hidden = n_features`, so the
  hidden layer can't be made wider/narrower independently.
- **No bias-free option, no batching, no validation split** — this is a
  from-scratch teaching/demo implementation, not a production model.
- **Same activation for both layers.** Using `z**2` for the output layer
  means predictions are always non-negative — fine for this 0/1 XOR target,
  but worth knowing if you swap in a different `y`.
- **Learning rate is sensitive** to the activation choice. The comment in
  the original code (`lower lr for z**2`) is a reminder that `z**2`'s
  growing derivative can blow up updates at higher `lr`.
