from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(benign_scores: np.ndarray, target_fpr: float) -> float:
    return float(np.quantile(benign_scores, 1.0 - target_fpr))


def auroc_vs_benign(score: np.ndarray, y: np.ndarray, attack_classes: list[str]) -> float:
    """AUROC of BENIGN vs the given attack classes only."""
    m = (y == "BENIGN") | np.isin(y, attack_classes)
    if not np.isin(y[m], attack_classes).any():
        return float("nan")
    return float(roc_auc_score(y[m] != "BENIGN", score[m]))


def binary_detection_metrics(is_attack: np.ndarray, score: np.ndarray, thr: float) -> dict:
    t = np.asarray(is_attack).astype(bool)
    p = np.asarray(score) >= thr
    tp, fp = int((t & p).sum()), int((~t & p).sum())
    fn, tn = int((t & ~p).sum()), int((~t & ~p).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    out = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
           "fpr": fp / max(fp + tn, 1)}
    if 0 < t.sum() < len(t):
        out["auroc"] = float(roc_auc_score(t, score))
        out["auprc"] = float(average_precision_score(t, score))
    else:
        out["auroc"] = out["auprc"] = float("nan")
    return out


def per_class_flag_rate(y: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    return {c: float(np.mean(flagged[y == c])) for c in sorted(set(y.tolist()))}
