"""Experiment 04 - Are the explanations faithful and actionable?

RQ4: Do the attributed features actually drive the model's risk score (fidelity), and how often can
a small, behaviourally-plausible change clear the alert (counterfactual validity / sparsity)?

Fidelity = drop in tabular risk after resetting the top-k attributed features to their benign
median, compared with resetting k random features. Methods unavailable in the environment
(shap needs a tree backend + shap; lime needs lime) are skipped and logged.

SCOPE NOTE: this measures faithfulness to the model, not usefulness to a human analyst. A user
study is required to claim the latter.
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from argus.config import load_config
from argus.evaluation import aggregate_seeds, write_results

from argus.explainability import (BaselineStats, ae_attribution, greedy_counterfactual,
                                  lime_attribution, perturbation_attribution, shap_attribution)
from argus.pipeline import make_context, train_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-flows", type=int, default=15000)
    ap.add_argument("--n-explain", type=int, default=60)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    rows, cf_rows = [], []
    for seed in a.seeds:
        cfg = load_config(a.config, {"data": {"n_flows": a.n_flows}})
        ctx = make_context(cfg, seed)
        det = train_detector(ctx)
        tr = ctx.idx["train"]; ben = tr[ctx.y[tr] == "BENIGN"]
        base = BaselineStats.fit(ctx.X[ben], ctx.fb.feature_names_, seed=seed)
        bme = det.autoencoder.feature_errors(ctx.X[ben]).mean(0)
        rng = np.random.default_rng(seed)
        known, unknown = cfg["data"]["known_attacks"], cfg["data"]["unknown_attacks"]
        te = ctx.idx["test"]
        dec = det.decide(det.component_scores(ctx.X, ctx.graph, te))
        for group, classes in (("known", known), ("unseen", unknown)):
            cand = te[np.isin(ctx.y[te], classes) & dec["is_anomaly"]]
            if len(cand) == 0:
                continue
            pick = rng.choice(cand, min(a.n_explain, len(cand)), replace=False)
            X = ctx.X[pick]
            risk0 = det.tabular_risk(X)
            attrs = {"random": rng.random(X.shape),
                     "ae_error_share": ae_attribution(det.autoencoder, X, bme),
                     "occlusion": np.stack([perturbation_attribution(det.tabular_risk, x, base.median) for x in X])}
            try:
                attrs["treeshap"] = shap_attribution(det.supervised, X)
            except Exception as e:
                print(f"[skip] treeshap: {type(e).__name__}")
            try:
                attrs["lime"] = np.stack([lime_attribution(det.tabular_risk, x, base.sample, base.names, 500) for x in X[:20]])
            except Exception as e:
                print(f"[skip] lime: {type(e).__name__}")
            for name, A in attrs.items():
                n = len(A)
                Xm = X[:n].copy()
                top = np.argsort(-A, axis=1)[:, : a.k]
                for i in range(n):
                    Xm[i, top[i]] = base.median[top[i]]
                drop = risk0[:n] - det.tabular_risk(Xm)
                rows.append({"seed": seed, "flows": group, "method": name, f"risk_drop_top{a.k}": drop.mean(),
                             f"alert_cleared_top{a.k}": float((det.tabular_risk(Xm) < det.threshold_tabular).mean())})
            valid, nch = [], []
            for x in X[:30]:
                cf = greedy_counterfactual(det.tabular_risk, x, base.median, det.threshold_tabular, base.names)
                valid.append(cf["valid"]); nch.append(cf["n_changed"])
            cf_rows.append({"seed": seed, "flows": group, "counterfactual_valid_rate": float(np.mean(valid)),
                            "mean_features_changed": float(np.mean(nch))})
    t1 = aggregate_seeds(rows, ["flows", "method"])
    t2 = aggregate_seeds(cf_rows, ["flows"])
    write_results(Path(__file__).parent, "attribution_fidelity", t1, rows,
                  "Higher risk_drop = more faithful attribution. 'random' is the null baseline.\n")
    write_results(Path(__file__).parent, "counterfactuals", t2, cf_rows,
                  "Greedy counterfactual (<=6 features reset to benign median). Behavioural, not packet-realisable.\n")


if __name__ == "__main__":
    main()
