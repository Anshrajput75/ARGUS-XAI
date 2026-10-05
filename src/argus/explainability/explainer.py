"""Turn model internals into analyst-readable explanations."""
from __future__ import annotations

import numpy as np

from argus.explainability.attribution import (ae_attribution, lime_attribution,
                                               perturbation_attribution, shap_attribution)
from argus.explainability.baseline import BaselineStats
from argus.explainability.counterfactual import greedy_counterfactual
from argus.features.engineering import FEATURE_DESCRIPTIONS


BINARY = ("is_", "proto_", "dst_unseen")


def human_value(name: str, v: float) -> str:
    if name.startswith(BINARY):
        return "yes" if v >= 0.5 else "no"
    if name.startswith("log_") and "ratio" not in name:
        raw = np.expm1(v)
        return f"{raw:,.0f}" if raw >= 10 else f"{raw:.2f}"
    return f"{v:.2f}"


class FlowExplainer:
    """method: auto | shap | lime | perturbation (auto: shap if usable, else perturbation)."""

    def __init__(self, detector, baseline: BaselineStats, benign_mean_err: np.ndarray,
                 benign_mean_err_graph: np.ndarray | None = None):
        self.det, self.base, self.bme = detector, baseline, benign_mean_err
        self.bme_g = benign_mean_err_graph

    def _attr(self, x: np.ndarray, trigger: str, method: str, graph_err=None) -> tuple[np.ndarray, str]:
        det = self.det
        if trigger == "autoencoder":
            return ae_attribution(det.autoencoder, x[None], self.bme)[0], "autoencoder reconstruction error"
        if trigger == "graph" and graph_err is not None and self.bme_g is not None:
            e = graph_err / (self.bme_g + 1e-9)
            return e / max(e.sum(), 1e-12), "graph-context reconstruction error"
        if trigger == "graph":
            return ae_attribution(det.autoencoder, x[None], self.bme)[0], "autoencoder reconstruction error"
        fn = det.supervised.attack_score
        if method in ("auto", "shap"):
            try:
                return shap_attribution(det.supervised, x[None])[0], "TreeSHAP"
            except Exception:
                if method == "shap":
                    raise
        if method == "lime":
            return lime_attribution(fn, x, self.base.sample, self.base.names), "LIME"
        return perturbation_attribution(fn, x, self.base.median), "occlusion (reset-to-baseline)"

    def explain(self, x: np.ndarray, trigger: str, risk: float, verdict: str,
                method: str = "auto", top_k: int = 5, with_counterfactual: bool = True,
                graph_err: np.ndarray | None = None) -> dict:
        attr, used = self._attr(x, trigger, method, graph_err)
        order = np.argsort(-attr)[:top_k]
        z = self.base.z(x)
        top = []
        for j in order:
            name = self.base.names[j]
            if attr[j] <= 0 or abs(z[j]) < 0.5:
                continue
            direction = "higher" if z[j] > 0 else "lower"
            top.append({"feature": name, "description": FEATURE_DESCRIPTIONS.get(name, name),
                        "value": human_value(name, x[j]), "baseline": human_value(name, self.base.median[j]),
                        "z": float(z[j]), "direction": direction, "attribution": float(attr[j])})
        cf = None
        if with_counterfactual and verdict != "Normal":
            cf = greedy_counterfactual(self.det.tabular_risk, x, self.base.median,
                                       self.det.threshold_tabular, self.base.names)
            for c in cf["changes"]:
                c["from_h"] = human_value(c["feature"], c["from"])
                c["to_h"] = human_value(c["feature"], c["to"])
        return {"method": used, "top_features": top, "counterfactual": cf,
                "narrative": narrate(verdict, risk, top, cf)}


def narrate(verdict: str, risk: float, top: list[dict], cf: dict | None) -> str:
    if verdict == "Normal":
        return f"Flow is within learned baseline behaviour (risk {risk:.2f})."
    lines = [f"{verdict} (risk {risk:.2f}). Why:"]
    for t in top[:4]:
        lines.append(f"  - {t['description']}: {t['value']} ({t['direction']} than typical "
                     f"{t['baseline']}, {t['z']:+.1f} sd)")
    if cf and cf["changes"]:
        names = ", ".join(FEATURE_DESCRIPTIONS.get(c["feature"], c["feature"]) for c in cf["changes"])
        if cf["valid"]:
            lines.append(f"Counterfactual: returning [{names}] to baseline would drop this flow "
                         f"below the alert threshold ({cf['threshold']:.2f}).")
        else:
            lines.append(f"Counterfactual: even resetting [{names}] does not clear the alert; "
                         "the anomaly is spread across many features.")
    return "\n".join(lines)
