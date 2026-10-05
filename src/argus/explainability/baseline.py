from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineStats:
    names: list[str]
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    sample: np.ndarray       # small benign sample (used by LIME)

    @classmethod
    def fit(cls, X_benign: np.ndarray, names: list[str], n_sample: int = 2000, seed: int = 0):
        rng = np.random.default_rng(seed)
        take = rng.choice(len(X_benign), min(n_sample, len(X_benign)), replace=False)
        return cls(list(names), np.median(X_benign, 0), X_benign.mean(0),
                   X_benign.std(0) + 1e-9, X_benign[take])

    def z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std
