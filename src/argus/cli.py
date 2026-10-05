"""argus generate-data | train | evaluate | serve | demo"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from argus.config import load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="argus")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate-data"); g.add_argument("--n-flows", type=int, default=40000)
    g.add_argument("--out", default="data/raw/synthetic.csv")
    t = sub.add_parser("train"); t.add_argument("--n-flows", type=int); t.add_argument("--out")
    t.add_argument("--seed", type=int)
    e = sub.add_parser("evaluate"); e.add_argument("--model", default=None)
    s = sub.add_parser("serve"); s.add_argument("--host", default="0.0.0.0"); s.add_argument("--port", type=int, default=8000)
    d = sub.add_parser("demo"); d.add_argument("--n-flows", type=int, default=15000)
    a = ap.parse_args(argv)

    if a.cmd == "generate-data":
        from argus.data import generate_flows
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        generate_flows(a.n_flows).to_csv(a.out, index=False)
        print(f"wrote {a.out}")
    elif a.cmd in ("train", "demo"):
        from argus.inference import ArgusEngine
        from argus.tracking import track
        ov = {"data": {"n_flows": a.n_flows}} if getattr(a, "n_flows", None) else None
        cfg = load_config(a.config, ov)
        out = getattr(a, "out", None) or cfg["artifacts_dir"]
        with track("train", cfg) as run:
            eng, ctx, metrics = ArgusEngine.train(cfg, getattr(a, "seed", None))
            eng.save(out)
            run.log_metrics(metrics); run.log_artifacts(out)
        print(json.dumps({k: v for k, v in metrics.items() if k != "flag_rate_by_class"}, indent=2, default=float))
        print(f"saved model to {out}")
    elif a.cmd == "evaluate":
        from argus.inference import ArgusEngine
        cfg = load_config(a.config)
        eng = ArgusEngine.load(a.model or cfg["artifacts_dir"])
        print(json.dumps(eng.meta["metrics"], indent=2, default=float))
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("api.main:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
