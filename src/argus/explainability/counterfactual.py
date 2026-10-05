from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np


def greedy_counterfactual(score_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                          baseline: np.ndarray, threshold: float, names: list[str],
                          max_changes: int = 6, frozen: Iterable[str] = ()) -> dict:
    """Smallest greedy set of features that, moved back to benign-median, drop risk < threshold.

    Honest scope: this is a *behavioural* counterfactual over engineered features (e.g. "if the
    connection frequency returned to baseline"), not a realisable packet-level edit. Features are
    not constrained to be jointly feasible. Treat it as a diagnostic, not as attack evasion advice.
    """
    frozen = set(frozen)
    cur = x.astype(float).copy()
    s0 = s = float(score_fn(cur[None])[0])
    free = [j for j in range(len(x)) if names[j] not in frozen and cur[j] != baseline[j]]
    changes = []
    while s >= threshold and free and len(changes) < max_changes:
        cands = np.tile(cur, (len(free), 1))
        cands[np.arange(len(free)), free] = baseline[free]
        sc = score_fn(cands)
        k = int(np.argmin(sc))
        j = free.pop(k)
        changes.append({"feature": names[j], "from": float(cur[j]), "to": float(baseline[j]),
                        "risk_after": float(sc[k])})
        cur[j] = baseline[j]
        s = float(sc[k])
    return {"valid": s < threshold, "original_risk": s0, "final_risk": s, "threshold": threshold,
            "n_changed": len(changes), "changes": changes}
