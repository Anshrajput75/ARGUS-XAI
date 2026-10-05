"""ARGUS-XAI inference API.  Run: uvicorn api.main:app --port 8000"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from argus.config import load_config
from argus.inference import ArgusEngine
from argus.storage import Cache, Store

STATE: dict = {}
LOCK = threading.Lock()


class FlowIn(BaseModel):
    flow_id: int
    timestamp: float
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str = "TCP"
    duration: float
    fwd_packets: int
    bwd_packets: int
    fwd_bytes: float
    bwd_bytes: float
    syn_flag_count: int = 0


class AnalyzeRequest(BaseModel):
    flows: list[FlowIn] = Field(min_length=1)
    store_alerts: bool = True


class ExplainRequest(BaseModel):
    flows: list[FlowIn] = Field(min_length=1, description="batch providing network context")
    flow_id: int
    method: str = "auto"


class FeedbackIn(BaseModel):
    flow_id: int
    label: str = Field(description="analyst label, e.g. WebAttack, Botnet, or BENIGN")
    analyst: str = "analyst"


class UpdateRequest(BaseModel):
    strategy: str = "replay"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    path = os.environ.get("ARGUS_MODEL_DIR", cfg["artifacts_dir"])
    if os.path.exists(os.path.join(path, "engine.joblib")):
        STATE["engine"] = ArgusEngine.load(path)
    elif os.environ.get("ARGUS_AUTO_TRAIN", "0") == "1":
        eng, _, _ = ArgusEngine.train(load_config(overrides={"data": {"n_flows": 15000}}))
        eng.save(path)
        STATE["engine"] = eng
    STATE["store"], STATE["cache"], STATE["last"] = Store(), Cache(), {}
    yield


app = FastAPI(title="ARGUS-XAI", version="0.1.0", lifespan=lifespan)


def _engine() -> ArgusEngine:
    if "engine" not in STATE:
        raise HTTPException(503, "No trained model. Run `argus train` or set ARGUS_AUTO_TRAIN=1.")
    return STATE["engine"]


def _frame(flows: list[FlowIn]) -> pd.DataFrame:
    return pd.DataFrame([f.model_dump() for f in flows])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "engine" in STATE}


@app.get("/model/info")
def model_info():
    m = _engine().meta
    return {k: m[k] for k in ("version", "classes", "known_attacks", "threshold", "updates", "n_train")}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    eng = _engine()
    with LOCK:
        an = eng.analyze(_frame(req.flows))
    n_stored = STATE["store"].add_alerts(an.flows, an.X) if req.store_alerts else 0
    STATE["cache"].clear("hosts")
    STATE["last"] = {"flows": an.flows, "graph": an}
    f = an.flows
    cols = ["flow_id", "src_ip", "dst_ip", "risk", "risk_level", "verdict", "pred_class", "class_confidence"]
    return {"n_flows": len(f), "n_anomalies": int(f["is_anomaly"].sum()), "stored_alerts": n_stored,
            "threshold": eng.detector.threshold,
            "top_alerts": f[f["is_anomaly"]].nlargest(50, "risk")[cols].to_dict("records"),
            "host_risk": an.host_risk.head(20).to_dict("records")}


@app.post("/explain")
def explain(req: ExplainRequest):
    eng = _engine()
    with LOCK:
        an = eng.analyze(_frame(req.flows))
    hit = an.flows.index[an.flows["flow_id"] == req.flow_id]
    if len(hit) == 0:
        raise HTTPException(404, "flow_id not in batch")
    return eng.explain(an, int(hit[0]), req.method)


@app.get("/alerts")
def alerts(min_risk: float = 0.0, limit: int = 100):
    return STATE["store"].alerts(min_risk, limit).to_dict("records")


@app.get("/hosts/risk")
def hosts_risk():
    cached = STATE["cache"].get("hosts")
    if cached is not None:
        return cached
    a = STATE["store"].alerts(0.0, 100000)
    if a.empty:
        return []
    g = a.groupby("src_ip")
    out = pd.DataFrame({"alerts": g.size(), "max_risk": g["risk"].max(), "mean_risk": g["risk"].mean()})
    res = out.sort_values("max_risk", ascending=False).reset_index().to_dict("records")
    STATE["cache"].set("hosts", res)
    return res


@app.get("/graph")
def graph(min_risk: float = 0.0):
    a = STATE["store"].alerts(min_risk, 100000)
    if a.empty:
        return {"nodes": [], "edges": []}
    nodes = {}
    for r in a.itertuples():
        for ip in (r.src_ip, r.dst_ip):
            nodes[ip] = max(nodes.get(ip, 0.0), r.risk)
    e = a.groupby(["src_ip", "dst_ip"]).agg(flows=("risk", "size"), max_risk=("risk", "max")).reset_index()
    return {"nodes": [{"id": k, "risk": v} for k, v in nodes.items()], "edges": e.to_dict("records")}


@app.post("/feedback")
def feedback(fb: FeedbackIn):
    STATE["store"].add_feedback(fb.flow_id, fb.label, fb.analyst)
    return {"status": "recorded"}


@app.post("/update")
def update(req: UpdateRequest):
    eng = _engine()
    X, y, ids = STATE["store"].pending_feedback()
    if len(y) == 0:
        raise HTTPException(409, "no pending feedback that matches stored alerts")
    with LOCK:
        res = eng.update_with_feedback(X, y, req.strategy)
        eng.save(os.environ.get("ARGUS_MODEL_DIR", eng.cfg["artifacts_dir"]))
    STATE["store"].mark_used(ids)
    return {**res, "feedback_used": len(ids)}
