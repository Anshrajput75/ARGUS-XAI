"""ArgusEngine: the single object the API, dashboard and CLI talk to."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from argus import __version__
from argus.config import load_config
from argus.continual_learning import ContinualLearner
from argus.explainability import BaselineStats, FlowExplainer
from argus.graph import FlowGraph, build_flow_graph
from argus.pipeline import (evaluate_detector, make_context, make_supervised, train_detector)


@dataclass
class Analysis:
    flows: pd.DataFrame          # input flows + scores + verdicts (row order preserved)
    X: np.ndarray
    graph: FlowGraph
    scores: dict
    host_risk: pd.DataFrame


class ArgusEngine:
    def __init__(self, cfg, fb, detector, baseline, benign_mean_err, learner, meta,
                 benign_mean_err_graph=None):
        self.bme_graph = benign_mean_err_graph
        self.cfg, self.fb, self.detector = cfg, fb, detector
        self.baseline, self.benign_mean_err = baseline, benign_mean_err
        self.learner, self.meta = learner, meta

    # ---- training -------------------------------------------------------------------------
    @classmethod
    def train(cls, cfg: dict | None = None, seed: int | None = None):
        cfg = cfg or load_config()
        ctx = make_context(cfg, seed)
        det = train_detector(ctx)
        metrics = evaluate_detector(ctx, det)
        tr = ctx.idx["train"]
        ben = tr[ctx.y[tr] == "BENIGN"]
        base = BaselineStats.fit(ctx.X[ben], ctx.fb.feature_names_, seed=ctx.seed)
        bme = det.autoencoder.feature_errors(ctx.X[ben]).mean(0)
        bme_g = det.graph_model.feature_errors(ctx.graph)[ben].mean(0)
        learner = ContinualLearner("replay", cfg["continual"]["replay_capacity"], ctx.seed)
        learner.observe(ctx.X[tr], ctx.y[tr])
        meta = {"version": __version__, "seed": ctx.seed, "classes": det.supervised.classes_,
                "known_attacks": cfg["data"]["known_attacks"], "n_train": int(len(tr)),
                "threshold": det.threshold, "metrics": metrics, "updates": 0}
        return cls(cfg, ctx.fb, det, base, bme, learner, meta, bme_g), ctx, metrics

    # ---- inference ------------------------------------------------------------------------
    def analyze(self, flows: pd.DataFrame) -> Analysis:
        flows = flows.reset_index(drop=True)
        Xdf = self.fb.transform(flows)
        X = Xdf.to_numpy()
        g = build_flow_graph(flows, X, self.cfg["features"]["window"])
        sc = self.detector.component_scores(X, g)
        dec = self.detector.decide(sc)
        out = flows.copy()
        out["score_supervised"], out["score_autoencoder"], out["score_graph"] = (
            sc["supervised"], sc["autoencoder"], sc["graph"])
        for k, v in dec.items():
            out[k] = v
        thr = self.detector.threshold
        hi = thr + 0.5 * (1 - thr)
        out["risk_level"] = np.where(out["risk"] >= hi, "HIGH", np.where(out["risk"] >= thr, "MEDIUM", "LOW"))
        return Analysis(out, X, g, sc, self.host_risk(out))

    @staticmethod
    def host_risk(res: pd.DataFrame) -> pd.DataFrame:
        g = res.groupby("src_ip")
        hr = pd.DataFrame({
            "flows": g.size(), "anomalous_flows": g["is_anomaly"].sum(),
            "max_risk": g["risk"].max(), "mean_risk": g["risk"].mean(),
            "dominant_verdict": g["pred_class"].agg(
                lambda s: s[s != "BENIGN"].mode().iat[0] if (s != "BENIGN").any() else "BENIGN"),
        }).reset_index()
        return hr.sort_values(["max_risk", "anomalous_flows"], ascending=False).reset_index(drop=True)

    def explain(self, an: Analysis, pos: int, method: str = "auto", top_k: int = 5) -> dict:
        row = an.flows.iloc[pos]
        ex = FlowExplainer(self.detector, self.baseline, self.benign_mean_err, self.bme_graph)
        g_err = (self.detector.graph_model.feature_errors(an.graph)[pos]
                 if row["trigger"] == "graph" else None)
        out = ex.explain(an.X[pos], row["trigger"], float(row["risk"]), row["verdict"], method,
                         top_k, graph_err=g_err)
        out.update({"flow_id": int(row.get("flow_id", pos)), "verdict": row["verdict"],
                    "risk": float(row["risk"]), "pred_class": row["pred_class"],
                    "components": {k: float(row[f"score_{k}"]) for k in self.detector.COMPONENTS}})
        return out

    # ---- continual learning ---------------------------------------------------------------
    def update_with_feedback(self, X_new: np.ndarray, y_new: np.ndarray, strategy: str = "replay") -> dict:
        """Analyst-labelled flows -> refreshed supervised head; threshold re-fit on benign val."""
        before = list(self.detector.supervised.classes_)
        self.learner.strategy = strategy
        sup_cfg = self.cfg
        new_sup = self.learner.update(lambda: make_supervised(sup_cfg, self.meta["seed"]), X_new, y_new)
        self.detector.replace_supervised(new_sup)
        self.meta["updates"] += 1
        self.meta["classes"] = list(new_sup.classes_)
        self.meta["threshold"] = self.detector.threshold
        return {"classes_before": before, "classes_after": list(new_sup.classes_),
                "threshold": self.detector.threshold, "strategy": strategy}

    # ---- persistence ----------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p / "engine.joblib")
        (p / "metadata.json").write_text(json.dumps(self.meta, indent=2, default=float))

    @classmethod
    def load(cls, path: str | Path) -> ArgusEngine:
        """SECURITY: joblib/pickle executes code on load. Only load artifacts you trained."""
        return joblib.load(Path(path) / "engine.joblib")
