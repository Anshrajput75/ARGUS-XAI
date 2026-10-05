"""Supervised multi-class detector. XGBoost when installed, otherwise sklearn fallbacks."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight


def _try_xgb():
    try:
        import xgboost
        return xgboost
    except Exception:
        return None


def _try_lgbm():
    try:
        import lightgbm
        return lightgbm
    except Exception:
        return None


class SupervisedDetector:
    """backend: auto | xgboost | lightgbm | random_forest | hgb"""

    def __init__(self, backend: str = "auto", n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.1, seed: int = 0):
        self.backend = backend
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.learning_rate, self.seed = learning_rate, seed
        self.classes_: list[str] = []
        self.model = None
        self.backend_used_: str = ""

    def _make(self):
        b = self.backend
        if b == "auto":
            b = "xgboost" if _try_xgb() else "hgb"
        if b == "xgboost":
            xgb = _try_xgb()
            if xgb is None:
                raise ImportError("xgboost not installed")
            m = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                  learning_rate=self.learning_rate, subsample=0.8,
                                  colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
                                  random_state=self.seed, verbosity=0)
        elif b == "lightgbm":
            lgb = _try_lgbm()
            if lgb is None:
                raise ImportError("lightgbm not installed")
            m = lgb.LGBMClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                   learning_rate=self.learning_rate, subsample=0.8,
                                   colsample_bytree=0.8, random_state=self.seed, verbose=-1)
        elif b == "random_forest":
            m = RandomForestClassifier(n_estimators=min(self.n_estimators, 300), n_jobs=-1,
                                       class_weight="balanced_subsample", random_state=self.seed)
        elif b == "hgb":
            m = HistGradientBoostingClassifier(max_iter=self.n_estimators, max_depth=self.max_depth,
                                               learning_rate=self.learning_rate,
                                               random_state=self.seed)
        else:
            raise ValueError(b)
        self.backend_used_ = b
        return m

    def fit(self, X, y) -> SupervisedDetector:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = sorted(set(y.tolist()))
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes (BENIGN + one attack)")
        idx = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([idx[c] for c in y])
        self.model = self._make()
        if self.backend_used_ == "random_forest":
            self.model.fit(X, yi)
        else:
            self.model.fit(X, yi, sample_weight=compute_sample_weight("balanced", yi))
        return self

    @property
    def benign_index(self) -> int:
        return self.classes_.index("BENIGN")

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(np.asarray(X, dtype=np.float32))

    def attack_score(self, X) -> np.ndarray:
        return 1.0 - self.predict_proba(X)[:, self.benign_index]

    def predict(self, X) -> np.ndarray:
        return np.array(self.classes_)[self.predict_proba(X).argmax(1)]
