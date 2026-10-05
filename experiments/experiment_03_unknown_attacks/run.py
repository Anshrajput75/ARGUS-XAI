"""Experiment 03 - Zero-day / unseen-attack detection, leave-one-attack-class-out.

RQ3: When an attack class is withheld from training, how often is it (a) flagged and (b) correctly
reported as 'previously unseen' instead of being forced into a known class or called benign?
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from argus.config import load_config
from argus.evaluation import aggregate_seeds, write_results

from argus.data.synthetic import ATTACK_CLASSES
from argus.evaluation.metrics import threshold_at_fpr
from argus.models.hybrid import UNKNOWN, fuse
from argus.pipeline import make_context, train_detector
from argus.preprocessing import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-flows", type=int, default=20000)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    rows = []
    for seed in a.seeds:
        cfg = load_config(a.config, {"data": {"n_flows": a.n_flows}})
        df = load_dataset(cfg, seed)
        for held in ATTACK_CLASSES:
            known = [c for c in ATTACK_CLASSES if c != held]
            ctx = make_context(cfg, seed, df=df, known=known, unknown=[held])
            det = train_detector(ctx)
            te, va = ctx.idx["test"], ctx.idx["val"]
            ben_va = va[ctx.y[va] == "BENIGN"]
            yte = ctx.y[te]
            sv = det.component_scores(ctx.X, ctx.graph, ben_va)
            st = det.component_scores(ctx.X, ctx.graph, te)
            dec = det.decide(st)
            m = yte == held
            systems = {"supervised only": {"supervised": 1.0}, "autoencoder only": {"autoencoder": 1.0},
                       "graph only": {"graph": 1.0}}
            for name, w in systems.items():
                thr = threshold_at_fpr(fuse({k: sv[k] for k in det.COMPONENTS}, w, "max"), 0.05)
                fl = fuse({k: st[k] for k in det.COMPONENTS}, w, "max") >= thr
                rows.append({"seed": seed, "held_out": held, "system": name,
                             "unseen_recall": fl[m].mean(), "fpr": fl[yte == "BENIGN"].mean(),
                             "labelled_unknown": float("nan")})
            for fusion in ("max", "noisy_or", "mean"):
                d2 = det.with_config(fusion=fusion)
                s2 = d2.decide(st)
                rows.append({"seed": seed, "held_out": held, "system": f"hybrid ({fusion})",
                             "unseen_recall": s2["is_anomaly"][m].mean(),
                             "fpr": s2["is_anomaly"][yte == "BENIGN"].mean(),
                             "labelled_unknown": (s2["verdict"][m] == UNKNOWN).mean()})
    tbl = aggregate_seeds(rows, ["held_out", "system"])
    write_results(Path(__file__).parent, "unseen_attack_detection", tbl, rows,
                  "unseen_recall: fraction of the withheld class flagged. labelled_unknown: flagged AND reported as\n"
                  "'previously unseen' rather than forced into a known class. 5% FPR operating point.\n")


if __name__ == "__main__":
    main()
