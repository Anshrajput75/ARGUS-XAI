"""Temporal host-communication graph: nodes = (host, time-window), edges = flows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FlowGraph:
    src: np.ndarray            # (E,) node index of source
    dst: np.ndarray            # (E,) node index of destination
    n_nodes: int
    edge_x: np.ndarray         # (E, d) edge features == flow features; edge i <-> row i
    node_keys: list            # [(ip, window_idx)]
    window: float

    @property
    def n_edges(self) -> int:
        return len(self.src)

    def node_ips(self) -> np.ndarray:
        return np.array([k[0] for k in self.node_keys])


def build_flow_graph(df: pd.DataFrame, X: pd.DataFrame | np.ndarray, window: float = 60.0) -> FlowGraph:
    w = np.floor(df["timestamp"].to_numpy() / window).astype(int)
    s_key = df["src_ip"].astype(str).to_numpy() + "|" + w.astype(str)
    d_key = df["dst_ip"].astype(str).to_numpy() + "|" + w.astype(str)
    codes, uniques = pd.factorize(np.concatenate([s_key, d_key]))
    n = len(df)
    keys = []
    for u in uniques:
        ip, wi = u.rsplit("|", 1)
        keys.append((ip, int(wi)))
    ex = X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
    return FlowGraph(codes[:n], codes[n:], len(uniques), ex, keys, window)


def node_features(g: FlowGraph) -> dict[str, np.ndarray]:
    """Label-free structural context per node."""
    n, d = g.n_nodes, g.edge_x.shape[1]
    out_deg = np.bincount(g.src, minlength=n).astype(float)
    in_deg = np.bincount(g.dst, minlength=n).astype(float)
    out_sum = np.zeros((n, d)); in_sum = np.zeros((n, d))
    for j in range(d):
        out_sum[:, j] = np.bincount(g.src, weights=g.edge_x[:, j], minlength=n)
        in_sum[:, j] = np.bincount(g.dst, weights=g.edge_x[:, j], minlength=n)
    out_mean = out_sum / np.maximum(out_deg, 1)[:, None]
    in_mean = in_sum / np.maximum(in_deg, 1)[:, None]
    pairs = pd.DataFrame({"s": g.src, "d": g.dst}).drop_duplicates()
    uo = pairs.groupby("s").size().reindex(range(n), fill_value=0).to_numpy(float)
    ui = pairs.groupby("d").size().reindex(range(n), fill_value=0).to_numpy(float)
    stats = np.log1p(np.stack([out_deg, in_deg, uo, ui], axis=1))
    return {"out_mean": out_mean, "in_mean": in_mean, "stats": stats}


def edge_context_features(g: FlowGraph) -> np.ndarray:
    """Per-edge vector: own features + source/destination neighbourhood summaries."""
    nf = node_features(g)
    return np.concatenate([
        g.edge_x, nf["out_mean"][g.src], nf["in_mean"][g.dst],
        nf["stats"][g.src][:, [0, 2]], nf["stats"][g.dst][:, [1, 3]]], axis=1).astype(np.float32)
