from __future__ import annotations

import numpy as np


def forgetting_summary(history: list[dict[str, float]], phases_learned: list[list[str]]) -> dict:
    """history[t]: per-class metric (e.g. recall) measured after phase t.
    phases_learned[t]: classes that were *newly* introduced at phase t.

    Returns average metric over classes seen so far, per-class forgetting
    (best earlier value minus final value; >0 = forgetting) and backward transfer.
    """
    seen: list[str] = []
    avg = []
    for t, new in enumerate(phases_learned):
        seen += new
        avg.append(float(np.mean([history[t][c] for c in seen])))
    final = history[-1]
    forgetting, bwt = {}, []
    for t, new in enumerate(phases_learned[:-1]):
        for c in new:
            best = max(history[k][c] for k in range(t, len(history)))
            forgetting[c] = float(best - final[c])
            bwt.append(final[c] - history[t][c])
    return {"avg_metric_after_each_phase": avg, "forgetting": forgetting,
            "mean_forgetting": float(np.mean(list(forgetting.values()))) if forgetting else 0.0,
            "backward_transfer": float(np.mean(bwt)) if bwt else 0.0}
