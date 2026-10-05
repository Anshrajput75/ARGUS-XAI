from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Class-balanced reservoir: each class keeps at most capacity // n_classes examples."""

    def __init__(self, capacity: int = 4000, seed: int = 0):
        self.capacity, self.rng = capacity, np.random.default_rng(seed)
        self.X: dict[str, np.ndarray] = {}

    def add(self, X: np.ndarray, y: np.ndarray) -> None:
        y = np.asarray(y)
        for c in np.unique(y):
            new = X[y == c]
            self.X[c] = np.concatenate([self.X[c], new]) if c in self.X else new
        cap = max(self.capacity // max(len(self.X), 1), 1)
        for c, arr in self.X.items():
            if len(arr) > cap:
                self.X[c] = arr[self.rng.choice(len(arr), cap, replace=False)]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.X:
            return np.empty((0, 0)), np.empty(0, dtype=object)
        X = np.concatenate(list(self.X.values()))
        y = np.concatenate([[c] * len(a) for c, a in self.X.items()])
        return X, y

    def __len__(self) -> int:
        return sum(len(a) for a in self.X.values())
