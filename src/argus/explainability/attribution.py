"""Local feature attributions. Positive value = pushes the anomaly score UP."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def perturbation_attribution(score_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                             baseline: np.ndarray) -> np.ndarray:
    """Model-agnostic occlusion: score drop when ONE feature is reset to its benign median."""
    d = len(x)
    M = np.tile(x, (d, 1))
    M[np.arange(d), np.arange(d)] = baseline
    return float(score_fn(x[None])[0]) - score_fn(M)


def ae_attribution(autoencoder, X: np.ndarray, benign_mean_err: np.ndarray) -> np.ndarray:
    """Share of the anomaly attributable to each feature, from reconstruction error.

    err_j is normalised by the mean error that feature has on benign traffic, so noisy features
    do not dominate by default. Rows sum to 1. This explains UNSUPERVISED alerts, where
    supervised SHAP values are meaningless (the classifier never saw the attack).
    """
    e = autoencoder.feature_errors(X) / (benign_mean_err + 1e-9)
    return e / np.maximum(e.sum(1, keepdims=True), 1e-12)


def shap_attribution(supervised, X: np.ndarray) -> np.ndarray:
    """TreeSHAP contribution toward 'not BENIGN'. Raises if shap/tree model unavailable."""
    import shap
    sv = shap.TreeExplainer(supervised.model).shap_values(np.asarray(X, dtype=np.float32))
    b = supervised.benign_index
    if isinstance(sv, list):
        return -sv[b]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return -sv[:, :, b]
    return sv if b == 0 else -sv   # binary: values are for class 1


def lime_attribution(score_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                     baseline_sample: np.ndarray, names: list[str], n_samples: int = 1500) -> np.ndarray:
    from lime.lime_tabular import LimeTabularExplainer
    ex = LimeTabularExplainer(baseline_sample, feature_names=names, mode="classification",
                              discretize_continuous=True, random_state=0)

    def predict(M):
        s = np.clip(score_fn(M), 0, 1)
        return np.stack([1 - s, s], axis=1)

    exp = ex.explain_instance(x, predict, num_features=len(names), num_samples=n_samples,
                              labels=(1,))
    out = np.zeros(len(names))
    for j, w in exp.as_map()[1]:
        out[j] = w
    return out
