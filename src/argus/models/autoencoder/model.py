"""Autoencoder / VAE trained on BENIGN traffic only. Score = reconstruction error.

PyTorch implementation when torch is available; otherwise an sklearn MLPRegressor autoencoder
(no VAE) so the pipeline, tests and CI still run end-to-end.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = nn = None
    HAS_TORCH = False


if HAS_TORCH:
    class _Net(nn.Module):
        def __init__(self, d: int, hidden: list[int], variational: bool):
            super().__init__()
            dims = [d, *hidden]
            enc = []
            for i in range(len(dims) - 2):
                enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
            self.enc = nn.Sequential(*enc)
            self.variational = variational
            pre, z = dims[-2], dims[-1]
            if variational:
                self.mu, self.logvar = nn.Linear(pre, z), nn.Linear(pre, z)
            else:
                self.to_z = nn.Linear(pre, z)
            rd = list(reversed(dims))
            dec = []
            for i in range(len(rd) - 1):
                dec.append(nn.Linear(rd[i], rd[i + 1]))
                if i < len(rd) - 2:
                    dec.append(nn.ReLU())
            self.dec = nn.Sequential(*dec)

        def forward(self, x):
            h = self.enc(x)
            if self.variational:
                mu, lv = self.mu(h), self.logvar(h)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
                return self.dec(z), mu, lv
            return self.dec(self.to_z(h)), None, None


class TabularAutoencoder:
    name = "autoencoder"

    def __init__(self, hidden=(32, 16, 8), epochs: int = 30, lr: float = 1e-3,
                 batch_size: int = 256, variational: bool = False, beta: float = 0.1,
                 seed: int = 0, backend: str = "auto"):
        self.hidden, self.epochs, self.lr = list(hidden), epochs, lr
        self.batch_size, self.variational, self.beta = batch_size, variational, beta
        self.seed, self.backend = seed, backend
        self.scaler = StandardScaler()
        self.backend_used_ = ""
        self._net = None
        self._mlp = None

    def fit(self, X) -> TabularAutoencoder:
        Z = self.scaler.fit_transform(np.asarray(X, dtype=np.float64)).astype(np.float32)
        if self.backend == "torch" and not HAS_TORCH:
            raise ImportError("torch not installed")
        if HAS_TORCH and self.backend in ("auto", "torch"):
            self._fit_torch(Z)
            self.backend_used_ = "torch"
        else:
            if self.variational:
                warnings.warn("VAE requires torch; falling back to a plain sklearn autoencoder.")
            self._fit_sklearn(Z)
            self.backend_used_ = "sklearn"
        return self

    def _fit_torch(self, Z):
        torch.manual_seed(self.seed)
        net = _Net(Z.shape[1], self.hidden, self.variational)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        data = torch.from_numpy(Z)
        n = len(data)
        g = torch.Generator().manual_seed(self.seed)
        net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                xb = data[perm[i:i + self.batch_size]]
                rec, mu, lv = net(xb)
                loss = ((rec - xb) ** 2).mean()
                if mu is not None:
                    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                    loss = loss + self.beta * kl
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        self._net = net

    def _fit_sklearn(self, Z):
        layers = tuple(self.hidden) + tuple(reversed(self.hidden[:-1]))
        self._mlp = MLPRegressor(hidden_layer_sizes=layers, max_iter=max(60, self.epochs * 4),
                                 batch_size=min(self.batch_size, len(Z)),
                                 learning_rate_init=max(self.lr, 1e-3), random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._mlp.fit(Z, Z)

    def _reconstruct(self, Z: np.ndarray) -> np.ndarray:
        if self._net is not None:
            with torch.no_grad():
                return self._net(torch.from_numpy(Z.astype(np.float32)))[0].numpy()
        return self._mlp.predict(Z)

    def feature_errors(self, X) -> np.ndarray:
        Z = self.scaler.transform(np.asarray(X, dtype=np.float64))
        return (Z - self._reconstruct(Z)) ** 2

    def score(self, X) -> np.ndarray:
        return self.feature_errors(X).mean(1)
