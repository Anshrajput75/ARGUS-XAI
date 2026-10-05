from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import ks_2samp


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


@dataclass
class DriftReport:
    flagged: bool
    mean_psi: float
    drifted_features: list[str] = field(default_factory=list)
    psi_by_feature: dict[str, float] = field(default_factory=dict)


class DriftDetector:
    def __init__(self, psi_threshold: float = 0.2, ks_alpha: float = 0.001, min_frac: float = 0.15):
        self.psi_threshold, self.ks_alpha, self.min_frac = psi_threshold, ks_alpha, min_frac

    def fit(self, X_ref: np.ndarray, names: list[str]) -> DriftDetector:
        self.ref, self.names = X_ref, names
        return self

    def check(self, X_new: np.ndarray) -> DriftReport:
        by, drifted = {}, []
        for j, n in enumerate(self.names):
            p = psi(self.ref[:, j], X_new[:, j])
            by[n] = p
            if p > self.psi_threshold and ks_2samp(self.ref[:, j], X_new[:, j]).pvalue < self.ks_alpha:
                drifted.append(n)
        return DriftReport(len(drifted) / len(self.names) >= self.min_frac,
                           float(np.mean(list(by.values()))), drifted, by)
