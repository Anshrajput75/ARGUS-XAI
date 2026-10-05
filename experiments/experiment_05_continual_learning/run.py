"""Experiment 05 - Static model vs continual learning with analyst feedback.

RQ5: After an analyst labels newly discovered attack classes, can the supervised head learn them
without catastrophically forgetting earlier classes?  Compare:
  static  (never updated) | naive (new data only) | replay (new + buffer) | full (all data, oracle)
Phases: P0 = BENIGN + DDoS/BruteForce/PortScan; P1 += WebAttack; P2 += Botnet.
Only a small analyst-labelled sample (continual.feedback_per_class) is available per new class.
Unsupervised components are frozen, so unseen attacks stay detectable between updates.
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from argus.config import load_config
from argus.evaluation import aggregate_seeds, write_results

import copy

from argus.continual_learning import ContinualLearner, DriftDetector, forgetting_summary
from argus.pipeline import make_context, make_supervised, train_detector

PHASES = [["DDoS", "BruteForce", "PortScan"], ["WebAttack"], ["Botnet"]]
ALL = [c for p in PHASES for c in p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-flows", type=int, default=20000)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    rows, drift_rows = [], []
    for seed in a.seeds:
        cfg = load_config(a.config, {"data": {"n_flows": a.n_flows}})
        ctx = make_context(cfg, seed, known=ALL, unknown=[])
        rng = np.random.default_rng(seed)
        tr, te = ctx.idx["train"], ctx.idx["test"]
        yte, Xte = ctx.y[te], ctx.X[te]
        n_fb = cfg["continual"]["feedback_per_class"]
        # P0 detector trained on the first phase only
        p0 = tr[np.isin(ctx.y[tr], ["BENIGN", *PHASES[0]])]
        ctx0 = copy.copy(ctx); ctx0.idx = {**ctx.idx, "train": p0}
        base_det = train_detector(ctx0)
        ben_tr = tr[ctx.y[tr] == "BENIGN"]
        # drift check: does the stream containing the new class drift from the benign reference?
        dd = DriftDetector(cfg["continual"]["drift_psi_threshold"]).fit(ctx.X[ben_tr], ctx.fb.feature_names_)
        for k, new in enumerate(PHASES[1:], 1):
            stream = ctx.X[np.isin(ctx.y, new)][:500]
            rep = dd.check(stream)
            drift_rows.append({"seed": seed, "new_class": "+".join(new), "mean_psi": rep.mean_psi,
                               "drifted_features": len(rep.drifted_features), "flagged": float(rep.flagged)})
        for strat in ["static", "naive", "replay", "full"]:
            det = copy.copy(base_det)
            learner = ContinualLearner("replay" if strat == "static" else strat,
                                       cfg["continual"]["replay_capacity"], seed)
            learner.observe(ctx.X[p0], ctx.y[p0])
            history = []
            for t in range(len(PHASES)):
                if t > 0 and strat != "static":
                    new = PHASES[t]
                    pool = tr[np.isin(ctx.y[tr], new)]
                    pick = rng.choice(pool, min(n_fb, len(pool)), replace=False)
                    ben_pick = rng.choice(ben_tr, min(3 * n_fb, len(ben_tr)), replace=False)
                    idx = np.concatenate([pick, ben_pick])
                    sup = learner.update(lambda: make_supervised(cfg, seed), ctx.X[idx], ctx.y[idx])
                    det.replace_supervised(sup)
                pred = det.supervised.predict(Xte)
                rec = {c: float((pred[yte == c] == c).mean()) for c in [*ALL, "BENIGN"]}
                history.append(rec)
                dec = det.decide(det.component_scores(ctx.X, ctx.graph, te))
                seen = [c for p in PHASES[: t + 1] for c in p]
                rows.append({"seed": seed, "strategy": strat, "phase": t,
                             "avg_recall_seen_classes": np.mean([rec[c] for c in seen]),
                             "recall_newest_class": np.mean([rec[c] for c in PHASES[t]]),
                             "benign_recall": rec["BENIGN"],
                             "hybrid_flag_rate_all_attacks": float(dec["is_anomaly"][yte != "BENIGN"].mean()),
                             "hybrid_benign_fpr": float(dec["is_anomaly"][yte == "BENIGN"].mean())})
            fs = forgetting_summary(history, PHASES)
            for r in rows[-len(PHASES):]:
                r["mean_forgetting_final"] = fs["mean_forgetting"]
                r["backward_transfer_final"] = fs["backward_transfer"]
    tbl = aggregate_seeds(rows, ["strategy", "phase"])
    write_results(Path(__file__).parent, "continual_learning", tbl, rows,
                  "recall = supervised head's class recall on the held-out test split (closed-set label correct).\n"
                  "mean_forgetting_final: mean over old classes of (best earlier recall - final recall); >0 means forgetting.\n"
                  "static = never updated; naive = new stream only; replay = new + class-balanced buffer; full = oracle.\n")
    write_results(Path(__file__).parent, "drift_detection", aggregate_seeds(drift_rows, ["new_class"]), drift_rows,
                  "PSI/KS drift of a stream containing the new class vs the benign reference.\n")


if __name__ == "__main__":
    main()
