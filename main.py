import numpy as np

class TwoNN:
    def __init__(self, activation=lambda x: x, dact=None):
        self.activation = activation
        self.dact = dact          
        self.wx = None
        self.bx = None
        self.wy = None
        self.by = None

    def forward(self, x):
        self.zx = self.wx @ x + self.bx
        self.ax = self.activation(self.zx)
        self.zy = self.wy @ self.ax + self.by
        self.ay = self.activation(self.zy)
        return self.ay[0]                     

    def _dact(self, z):
        """Use analytical dact if given, else numerical"""
        if self.dact:
            return self.dact(z)
        return (self.activation(z + 1e-4) - self.activation(z)) / 1e-4

    def fit(self, x, y, lr=0.001, epochs=1000):  # ← lower lr for z**2
        n_samples, n_features = x.shape
        n_hidden = n_features

        # ── weight init: small random, NOT eye ──────────────────────
        self.wx = np.random.randn(n_hidden, n_features) * 0.1  
        self.bx = np.zeros(n_hidden)
        self.wy = np.random.randn(1, n_hidden) * 0.1     
        self.by = np.zeros(1)

        for epoch in range(epochs):
            total_loss = 0
            for i in range(n_samples):
                y_ = self.forward(x[i])
                err = y_ - y[i]                      
                total_loss += err ** 2

                # ── output layer ──
                d_out    = self._dact(self.zy)             
                delta_out = 2 * err * d_out                

                dwy = delta_out[:, None] * self.ax[None, :]
                self.wy -= lr * dwy
                self.by -= lr * delta_out

                # ── hidden layer ───
                d_hid    = self._dact(self.zx)             
                d_hidden = (self.wy.T @ delta_out) * d_hid 

                dwx = d_hidden[:, None] * x[i][None, :]   
                self.wx -= lr * dwx
                self.bx -= lr * d_hidden

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {total_loss/n_samples:.6f}")

    def predict(self, x):
        return np.array([self.forward(xi) for xi in x])
