"""Hybrid anomaly decision engine: supervised + autoencoder + graph, fused into one risk score."""
from __future__ import annotations

import copy

import numpy as np

from argus.evaluation.metrics import threshold_at_fpr

UNKNOWN = "Anomalous - previously unseen behavior"


class ScoreCalibrator:
    """Map a raw reconstruction error to (0,1) using BENIGN validation statistics.

    z = (log(err) - mu) / sd ; score = sigmoid(z - offset). Monotone, no saturation, and the
    benign mass sits near 0 while errors far above benign noise approach 1.
    """

    def __init__(self, z_offset: float = 2.0):
        self.z_offset = z_offset

    def fit(self, raw: np.ndarray) -> ScoreCalibrator:
        lg = np.log(np.asarray(raw) + 1e-9)
        self.mu, self.sd = float(lg.mean()), float(lg.std() + 1e-9)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        z = (np.log(np.asarray(raw) + 1e-9) - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-np.clip(z - self.z_offset, -30, 30)))


def fuse(scores: dict[str, np.ndarray], weights: dict[str, float], how: str) -> np.ndarray:
    names = [k for k in scores if weights.get(k, 0) > 0]
    if not names:
        raise ValueError("all components disabled")
    S = np.stack([weights[k] * scores[k] for k in names], axis=1)
    if how == "max":
        return S.max(1)
    if how == "noisy_or":
        return 1.0 - np.prod(1.0 - np.clip(S, 0, 1), axis=1)
    if how == "mean":
        return S.sum(1) / sum(weights[k] for k in names)
    raise ValueError(how)


class HybridDetector:
    COMPONENTS = ("supervised", "autoencoder", "graph")

    def __init__(self, supervised, autoencoder, graph_model, fusion: str = "max",
                 weights: dict | None = None, target_fpr: float = 0.05,
                 known_class_confidence: float = 0.6, z_offset: float = 2.0):
        self.supervised, self.autoencoder, self.graph_model = supervised, autoencoder, graph_model
        self.fusion = fusion
        self.weights = {**{k: 1.0 for k in self.COMPONENTS}, **(weights or {})}
        self.target_fpr, self.known_conf = target_fpr, known_class_confidence
        self.cal_ae, self.cal_graph = ScoreCalibrator(z_offset), ScoreCalibrator(z_offset)
        self.threshold = 0.5
        self.threshold_tabular = 0.5
        self._val = None

    # ---- calibration ----------------------------------------------------------------------
    def calibrate(self, X_val_benign: np.ndarray, graph, val_benign_idx: np.ndarray) -> None:
        """Fit calibrators and the alert threshold on BENIGN validation flows only."""
        ae_raw = self.autoencoder.score(X_val_benign)
        g_raw = self.graph_model.score(graph)[val_benign_idx]
        self.cal_ae.fit(ae_raw); self.cal_graph.fit(g_raw)
        self._val = {"X": X_val_benign, "ae_raw": ae_raw, "g_raw": g_raw}
        self.recalibrate_threshold()

    def recalibrate_threshold(self) -> None:
        v = self._val
        comp = {"supervised": self.supervised.attack_score(v["X"]),
                "autoencoder": self.cal_ae.transform(v["ae_raw"]),
                "graph": self.cal_graph.transform(v["g_raw"])}
        self.threshold = threshold_at_fpr(fuse(comp, self.weights, self.fusion), self.target_fpr)
        wt = {**self.weights, "graph": 0.0}
        self.threshold_tabular = threshold_at_fpr(fuse(comp, wt, self.fusion), self.target_fpr)

    def replace_supervised(self, new_supervised) -> None:
        self.supervised = new_supervised
        self.recalibrate_threshold()

    def with_config(self, fusion: str | None = None, weights: dict | None = None) -> HybridDetector:
        """Shallow copy with a different fusion/weights (for ablations); threshold re-fit."""
        c = copy.copy(self)
        c.fusion = fusion or self.fusion
        c.weights = {**{k: 0.0 for k in self.COMPONENTS}, **weights} if weights else dict(self.weights)
        c.recalibrate_threshold()
        return c

    # ---- inference ------------------------------------------------------------------------
    def component_scores(self, X: np.ndarray, graph, idx: np.ndarray | None = None) -> dict:
        """X: feature matrix whose row i is edge i of `graph`. idx selects rows to return."""
        idx = np.arange(len(X)) if idx is None else np.asarray(idx)
        Xs = X[idx]
        proba = self.supervised.predict_proba(Xs)
        ae_raw, g_raw = self.autoencoder.score(Xs), self.graph_model.score(graph)[idx]
        return {"supervised": 1.0 - proba[:, self.supervised.benign_index],
                "autoencoder": self.cal_ae.transform(ae_raw),
                "graph": self.cal_graph.transform(g_raw),
                "proba": proba, "ae_raw": ae_raw, "graph_raw": g_raw}

    def tabular_risk(self, X: np.ndarray) -> np.ndarray:
        """Risk using only per-flow components (used for counterfactuals; graph is context)."""
        comp = {"supervised": self.supervised.attack_score(X),
                "autoencoder": self.cal_ae.transform(self.autoencoder.score(X))}
        return fuse(comp, {**self.weights, "graph": 0.0}, self.fusion)

    def decide(self, sc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        risk = fuse({k: sc[k] for k in self.COMPONENTS}, self.weights, self.fusion)
        flagged = risk >= self.threshold
        proba = sc["proba"].copy()
        proba[:, self.supervised.benign_index] = 0.0
        top = proba.argmax(1)
        top_p = proba[np.arange(len(top)), top]
        classes = np.array(self.supervised.classes_)
        known = flagged & (top_p >= self.known_conf)
        verdict = np.where(~flagged, "Normal", np.where(known, "Known attack", UNKNOWN))
        pred = np.where(known, classes[top], np.where(flagged, "unknown", "BENIGN"))
        trig = np.array(self.COMPONENTS)[np.stack(
            [self.weights[k] * sc[k] for k in self.COMPONENTS], 1).argmax(1)]
        trigger = np.where(known, "supervised", trig)
        return {"risk": risk, "is_anomaly": flagged, "verdict": verdict, "pred_class": pred,
                "class_confidence": top_p, "trigger": trigger}
