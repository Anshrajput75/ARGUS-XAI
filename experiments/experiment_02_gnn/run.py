"""Experiment 02 - Does network topology help? Tabular models vs graph model vs hybrids.

RQ2: Does graph context (neighbourhood behaviour of the hosts involved) improve anomaly
detection over per-flow tabular models trained on identical flow features?
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from argus.config import load_config
from argus.evaluation import aggregate_seeds, write_results

from argus.evaluation.metrics import auroc_vs_benign, threshold_at_fpr
from argus.models.hybrid import fuse
from argus.pipeline import make_autoencoder, make_context, make_graph_model, make_supervised, train_detector


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
        known, unknown = cfg["data"]["known_attacks"], cfg["data"]["unknown_attacks"]
        te, va = ctx.idx["test"], ctx.idx["val"]
        ben_va = va[ctx.y[va] == "BENIGN"]
        yte = ctx.y[te]
        det = train_detector(ctx)
        sc_te = det.component_scores(ctx.X, ctx.graph, te)
        sc_va = det.component_scores(ctx.X, ctx.graph, ben_va)

        def record(name, w, fusion="max"):
            s_va = fuse({k: sc_va[k] for k in det.COMPONENTS}, w, fusion)
            s_te = fuse({k: sc_te[k] for k in det.COMPONENTS}, w, fusion)
            fl = s_te >= threshold_at_fpr(s_va, 0.05)
            r = {"seed": seed, "system": name, "auroc_known": auroc_vs_benign(s_te, yte, known),
                 "auroc_unseen": auroc_vs_benign(s_te, yte, unknown),
                 "recall_unseen@5%FPR": fl[np.isin(yte, unknown)].mean(), "fpr": fl[yte == "BENIGN"].mean()}
            for u in unknown:
                r[f"recall_{u}"] = fl[yte == u].mean()
            rows.append(r)

        record("autoencoder (tabular)", {"autoencoder": 1.0})
        record(f"graph model ({det.graph_model.backend_used_})", {"graph": 1.0})
        record("supervised only", {"supervised": 1.0})
        record("hybrid: sup + ae (no topology)", {"supervised": 1.0, "autoencoder": 1.0})
        record("hybrid: sup + ae + graph", {"supervised": 1.0, "autoencoder": 1.0, "graph": 1.0})
    tbl = aggregate_seeds(rows, ["system"])
    write_results(Path(__file__).parent, "tabular_vs_graph", tbl, rows,
                  "Same features, same split, same 5% benign-FPR operating point for every row.\n"
                  "CAVEAT: graph is built transductively over the evaluation timeline (labels are never used).\n"
                  "With torch absent the 'graph model' is a neighbourhood-context autoencoder, not message passing.\n")


if __name__ == "__main__":
    main()
