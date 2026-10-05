"""End-to-end orchestration used by the CLI, API, dashboard and experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from argus.config import load_config
from argus.features import FeatureBuilder
from argus.graph import FlowGraph, build_flow_graph
from argus.models import GraphAnomalyModel, SupervisedDetector, TabularAutoencoder
from argus.models.hybrid import HybridDetector
from argus.preprocessing import load_dataset, split_indices


@dataclass
class Context:
    cfg: dict
    seed: int
    df: pd.DataFrame
    fb: FeatureBuilder
    X: np.ndarray
    y: np.ndarray
    graph: FlowGraph
    idx: dict[str, np.ndarray]


def make_context(cfg: dict | None = None, seed: int | None = None, df: pd.DataFrame | None = None,
                 known: list[str] | None = None, unknown: list[str] | None = None) -> Context:
    cfg = cfg or load_config()
    seed = cfg["seed"] if seed is None else seed
    df = load_dataset(cfg, seed) if df is None else df
    f = cfg["features"]
    fb = FeatureBuilder(f["window"], f["baseline_frac"], f["use_novelty"]).fit(df)
    X = fb.transform(df).to_numpy()
    graph = build_flow_graph(df, X, f["window"])
    d = cfg["data"]
    y = df["label"].to_numpy()
    idx = split_indices(y, known or d["known_attacks"], unknown or d["unknown_attacks"], seed,
                        d["test_size"], d["val_size"])
    return Context(cfg, seed, df, fb, X, y, graph, idx)


def make_supervised(cfg: dict, seed: int, backend: str | None = None) -> SupervisedDetector:
    m = dict(cfg["models"]["supervised"])
    if backend:
        m["backend"] = backend
    return SupervisedDetector(seed=seed, **m)


def make_autoencoder(cfg: dict, seed: int, variational: bool | None = None) -> TabularAutoencoder:
    m = dict(cfg["models"]["autoencoder"])
    if variational is not None:
        m["variational"] = variational
    return TabularAutoencoder(seed=seed, **m)


def make_graph_model(cfg: dict, seed: int) -> GraphAnomalyModel:
    return GraphAnomalyModel(seed=seed, **cfg["models"]["graph"])


def train_detector(ctx: Context) -> HybridDetector:
    cfg, seed, tr, va = ctx.cfg, ctx.seed, ctx.idx["train"], ctx.idx["val"]
    sup = make_supervised(cfg, seed).fit(ctx.X[tr], ctx.y[tr])
    ben_tr = tr[ctx.y[tr] == "BENIGN"]
    ae = make_autoencoder(cfg, seed).fit(ctx.X[ben_tr])
    gnn = make_graph_model(cfg, seed).fit(ctx.graph, ben_tr)
    h = cfg["hybrid"]
    det = HybridDetector(sup, ae, gnn, h["fusion"], h["weights"], h["target_fpr"],
                         h["known_class_confidence"], h["z_offset"])
    ben_va = va[ctx.y[va] == "BENIGN"]
    det.calibrate(ctx.X[ben_va], ctx.graph, ben_va)
    return det


def evaluate_detector(ctx: Context, det: HybridDetector, known: list[str] | None = None,
                      unknown: list[str] | None = None) -> dict:
    """Open-set evaluation on the test split."""
    from argus.evaluation.metrics import (auroc_vs_benign, binary_detection_metrics,
                                          per_class_flag_rate)
    d = ctx.cfg["data"]
    known, unknown = known or d["known_attacks"], unknown or d["unknown_attacks"]
    te = ctx.idx["test"]
    y = ctx.y[te]
    sc = det.component_scores(ctx.X, ctx.graph, te)
    dec = det.decide(sc)
    risk = dec["risk"]
    out = {"threshold": det.threshold, "target_fpr": det.target_fpr}
    out["all_attacks"] = binary_detection_metrics(y != "BENIGN", risk, det.threshold)
    out["auroc_known"] = auroc_vs_benign(risk, y, known)
    out["auroc_unseen"] = auroc_vs_benign(risk, y, unknown)
    out["flag_rate_by_class"] = per_class_flag_rate(y, dec["is_anomaly"])
    unk_mask = np.isin(y, unknown)
    out["unseen_flagged"] = float(dec["is_anomaly"][unk_mask].mean()) if unk_mask.any() else float("nan")
    out["unseen_labelled_unknown"] = (float((dec["verdict"][unk_mask] == "Anomalous - previously unseen behavior").mean())
                                      if unk_mask.any() else float("nan"))
    kn_mask = np.isin(y, known)
    out["known_correct_class"] = (float((dec["pred_class"][kn_mask] == y[kn_mask]).mean())
                                  if kn_mask.any() else float("nan"))
    out["benign_fpr"] = float(dec["is_anomaly"][y == "BENIGN"].mean())
    out["backends"] = {"supervised": det.supervised.backend_used_,
                       "autoencoder": det.autoencoder.backend_used_,
                       "graph": det.graph_model.backend_used_}
    return out
