"""Update the supervised component from analyst feedback without (fully) forgetting.

Strategies
  naive  : retrain on the NEW data stream only (benign + newly labelled attack) -> forgetting baseline
  replay : new data + class-balanced replay buffer of past data (rehearsal)
  full   : retrain on all data ever seen (upper bound / oracle, not deployable at scale)
The unsupervised detectors (autoencoder, graph) are trained on benign traffic and are untouched,
which is why novel attacks can still be flagged between updates.
"""
from __future__ import annotations

import numpy as np

from argus.continual_learning.replay import ReplayBuffer


class ContinualLearner:
    def __init__(self, strategy: str = "replay", capacity: int = 4000, seed: int = 0):
        assert strategy in ("naive", "replay", "full")
        self.strategy, self.seed = strategy, seed
        self.buffer = ReplayBuffer(capacity, seed)
        self._all_X: list[np.ndarray] = []
        self._all_y: list[np.ndarray] = []

    def observe(self, X: np.ndarray, y: np.ndarray) -> None:
        """Register data the model has already been trained on (seeds the buffer)."""
        self.buffer.add(X, y)
        self._all_X.append(X); self._all_y.append(np.asarray(y))

    def update(self, make_detector, X_new: np.ndarray, y_new: np.ndarray):
        """make_detector() -> fresh unfitted SupervisedDetector. Returns the new fitted one."""
        y_new = np.asarray(y_new)
        if self.strategy == "naive":
            Xt, yt = X_new, y_new
        elif self.strategy == "replay":
            Xb, yb = self.buffer.arrays()
            Xt = np.concatenate([Xb, X_new]) if len(Xb) else X_new
            yt = np.concatenate([yb, y_new]) if len(Xb) else y_new
        else:
            Xt = np.concatenate([*self._all_X, X_new]); yt = np.concatenate([*self._all_y, y_new])
        det = make_detector().fit(Xt, yt)
        self.observe(X_new, y_new)
        return det
