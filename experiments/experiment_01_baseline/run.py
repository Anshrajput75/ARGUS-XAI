"""Experiment 01 - Classical ML vs deep learning vs hybrid (open-set protocol).

RQ1: Does adding unsupervised detectors to a supervised classifier improve detection of attacks
that were absent from training, without hurting known-attack detection?
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from argus.config import load_config
from argus.evaluation import aggregate_seeds, write_results

from argus.evaluation.metrics import auroc_vs_benign, threshold_at_fpr
from argus.models.autoencoder.model import HAS_TORCH
from argus.pipeline import (evaluate_detector, make_autoencoder, make_context, make_supervised,
                            train_detector)
from sklearn.metrics import f1_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-flows", type=int, default=20000)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    rows = []
    for seed in a.seeds:
        cfg = load_config(a.config, {"data": {"n_flows": a.n_flows}})
        ctx = make_context(cfg, seed)
        d = cfg["data"]; known, unknown = d["known_attacks"], d["unknown_attacks"]
        tr, va, te = ctx.idx["train"], ctx.idx["val"], ctx.idx["test"]
        ben_tr, ben_va = tr[ctx.y[tr] == "BENIGN"], va[ctx.y[va] == "BENIGN"]
        yte = ctx.y[te]

        def record(name, s_val, s_te, closed_f1=float("nan")):
            thr = threshold_at_fpr(s_val, 0.05)
            fl = s_te >= thr
            rows.append({"seed": seed, "model": name,
                         "auroc_known": auroc_vs_benign(s_te, yte, known),
                         "auroc_unseen": auroc_vs_benign(s_te, yte, unknown),
                         "recall_known@5%FPR": fl[np.isin(yte, known)].mean(),
                         "recall_unseen@5%FPR": fl[np.isin(yte, unknown)].mean(),
                         "fpr": fl[yte == "BENIGN"].mean(), "closed_set_macroF1": closed_f1})

        backends = ["random_forest", "xgboost" if _has("xgboost") else "hgb"]
        if _has("lightgbm"):
            backends.append("lightgbm")
        for b in backends:
            sup = make_supervised(cfg, seed, b).fit(ctx.X[tr], ctx.y[tr])
            km = np.isin(yte, ["BENIGN", *known])
            f1 = f1_score(yte[km], sup.predict(ctx.X[te][km]), average="macro")
            record(f"supervised/{sup.backend_used_}", sup.attack_score(ctx.X[ben_va]), sup.attack_score(ctx.X[te]), f1)
        for v in ([False, True] if HAS_TORCH else [False]):
            ae = make_autoencoder(cfg, seed, v).fit(ctx.X[ben_tr])
            record("vae" if v else f"autoencoder/{ae.backend_used_}", ae.score(ctx.X[ben_va]), ae.score(ctx.X[te]))
        det = train_detector(ctx)
        sc = det.component_scores(ctx.X, ctx.graph, te)
        record("HYBRID (sup+ae+graph)", _risk(det, ctx, ben_va), det.decide(sc)["risk"])
        ev = evaluate_detector(ctx, det)
        rows[-1]["closed_set_macroF1"] = ev["known_correct_class"]
    tbl = aggregate_seeds(rows, ["model"])
    write_results(Path(__file__).parent, "baseline_comparison", tbl, rows,
                  "Open-set protocol: unseen classes appear only at test time. Threshold = 5% FPR on benign validation.\n"
                  "Backends actually used are in the model names; torch models only when torch is installed.\n"
                  "closed_set_macroF1 for HYBRID is accuracy of the known-class label among known attacks.\n")


def _has(m):
    try:
        __import__(m); return True
    except Exception:
        return False


def _risk(det, ctx, ben_va):
    from argus.models.hybrid import fuse
    sc = det.component_scores(ctx.X, ctx.graph, ben_va)
    return fuse({k: sc[k] for k in det.COMPONENTS}, det.weights, det.fusion)


if __name__ == "__main__":
    main()
