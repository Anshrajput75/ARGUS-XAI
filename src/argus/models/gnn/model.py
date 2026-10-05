"""Graph anomaly model: GraphSAGE-style encoder + edge decoder, trained on BENIGN edges only.

Each edge (flow) is reconstructed from the embeddings of its two endpoint nodes, which summarise
2-hop neighbourhood behaviour. A flow whose features cannot be explained by its network context
gets a high error. Pure-PyTorch (no PyG dependency). Without torch we fall back to a
"context autoencoder": neighbourhood statistics are concatenated to each flow's features and a
tabular autoencoder is trained on them (same idea, no message passing).
"""
from __future__ import annotations

import numpy as np

from argus.graph import FlowGraph, edge_context_features, node_features
from argus.models.autoencoder.model import HAS_TORCH, TabularAutoencoder

if HAS_TORCH:
    import torch
    from torch import nn

    class _SAGE(nn.Module):
        def __init__(self, in_dim: int, hidden: int, edge_dim: int):
            super().__init__()
            self.s1, self.n1 = nn.Linear(in_dim, hidden), nn.Linear(in_dim, hidden, bias=False)
            self.s2, self.n2 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden, bias=False)
            self.dec = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, edge_dim))

        @staticmethod
        def _mean_agg(h, s, d):
            agg = torch.zeros_like(h).index_add_(0, d, h[s])
            deg = torch.zeros(h.size(0)).index_add_(0, d, torch.ones(len(d))).clamp(min=1)
            return agg / deg[:, None]

        def forward(self, x, src, dst, e_src, e_dst):
            s, d = torch.cat([src, dst]), torch.cat([dst, src])  # undirected message passing
            h = torch.relu(self.s1(x) + self.n1(self._mean_agg(x, s, d)))
            h = torch.relu(self.s2(h) + self.n2(self._mean_agg(h, s, d)))
            return self.dec(torch.cat([h[e_src], h[e_dst]], dim=1))


class GraphAnomalyModel:
    name = "graph"

    def __init__(self, hidden: int = 32, epochs: int = 80, lr: float = 5e-3, seed: int = 0,
                 backend: str = "auto"):
        self.hidden, self.epochs, self.lr, self.seed, self.backend = hidden, epochs, lr, seed, backend
        self.backend_used_ = ""
        self._net = None
        self._ctx_ae: TabularAutoencoder | None = None

    # ---- shared preprocessing -------------------------------------------------------------
    def _node_matrix(self, g: FlowGraph) -> np.ndarray:
        nf = node_features(g)
        return np.concatenate([nf["out_mean"], nf["in_mean"], nf["stats"]], axis=1)

    def fit(self, g: FlowGraph, train_idx: np.ndarray) -> GraphAnomalyModel:
        if self.backend == "torch" and not HAS_TORCH:
            raise ImportError("torch not installed")
        if HAS_TORCH and self.backend in ("auto", "torch"):
            self._fit_torch(g, train_idx)
            self.backend_used_ = "torch-sage"
        else:
            ctx = edge_context_features(g)
            self._ctx_ae = TabularAutoencoder(hidden=(48, 24, 8), epochs=30, seed=self.seed,
                                              backend="sklearn").fit(ctx[train_idx])
            self.backend_used_ = "context-ae"
        return self

    def _fit_torch(self, g: FlowGraph, train_idx: np.ndarray):
        torch.manual_seed(self.seed)
        nodes = self._node_matrix(g)
        touched = np.unique(np.concatenate([g.src[train_idx], g.dst[train_idx]]))
        self.n_mu, self.n_sd = nodes[touched].mean(0), nodes[touched].std(0) + 1e-6
        self.e_mu, self.e_sd = g.edge_x[train_idx].mean(0), g.edge_x[train_idx].std(0) + 1e-6
        x, ex, src, dst, tr = self._tensors(g, nodes, train_idx)
        net = _SAGE(x.size(1), self.hidden, ex.size(1))
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)
        net.train()
        for _ in range(self.epochs):
            rec = net(x, src, dst, src, dst)
            loss = ((rec[tr] - ex[tr]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        self._net = net

    def _tensors(self, g, nodes, idx=None):
        x = torch.from_numpy(((nodes - self.n_mu) / self.n_sd).astype(np.float32))
        ex = torch.from_numpy(((g.edge_x - self.e_mu) / self.e_sd).astype(np.float32))
        src, dst = torch.from_numpy(g.src).long(), torch.from_numpy(g.dst).long()
        tr = torch.from_numpy(idx).long() if idx is not None else None
        return x, ex, src, dst, tr

    # ---- scoring --------------------------------------------------------------------------
    def feature_errors(self, g: FlowGraph) -> np.ndarray:
        """Per-edge squared error on the (standardised) flow features: shape (E, d)."""
        if self._net is not None:
            x, ex, src, dst, _ = self._tensors(g, self._node_matrix(g))
            with torch.no_grad():
                rec = self._net(x, src, dst, src, dst)
            return ((rec - ex) ** 2).numpy()
        return self._ctx_ae.feature_errors(edge_context_features(g))[:, : g.edge_x.shape[1]]

    def score(self, g: FlowGraph) -> np.ndarray:
        if self._net is not None:
            return self.feature_errors(g).mean(1)
        return self._ctx_ae.score(edge_context_features(g))  # full-context error
