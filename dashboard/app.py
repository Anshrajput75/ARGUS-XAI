"""ARGUS-XAI Security Center (Streamlit). Run: streamlit run dashboard/app.py"""
from __future__ import annotations

import os

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from argus.config import load_config
from argus.data import generate_flows
from argus.inference import ArgusEngine

st.set_page_config(page_title="ARGUS-XAI", layout="wide")
MODEL_DIR = os.environ.get("ARGUS_MODEL_DIR", load_config()["artifacts_dir"])
COLORS = {"HIGH": "#d62728", "MEDIUM": "#ff9f1c", "LOW": "#2ca02c"}


@st.cache_resource(show_spinner="Loading model (training a demo model on first run)...")
def get_engine() -> ArgusEngine:
    if os.path.exists(os.path.join(MODEL_DIR, "engine.joblib")):
        return ArgusEngine.load(MODEL_DIR)
    eng, _, _ = ArgusEngine.train(load_config(overrides={"data": {"n_flows": 15000}}))
    eng.save(MODEL_DIR)
    return eng


def investigation_graph(flows: pd.DataFrame, min_risk: float) -> go.Figure:
    a = flows[flows["risk"] >= min_risk]
    G = nx.DiGraph()
    risk = {}
    for r in a.itertuples():
        G.add_edge(r.src_ip, r.dst_ip)
        for ip in (r.src_ip, r.dst_ip):
            risk[ip] = max(risk.get(ip, 0.0), r.risk)
    if G.number_of_nodes() == 0:
        return go.Figure()
    pos = nx.spring_layout(G, seed=3, k=0.9)
    ex, ey = [], []
    for u, v in G.edges():
        ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.5, color="#999"), hoverinfo="none"))
    nodes = list(G.nodes())
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", text=nodes,
        marker=dict(size=[10 + 24 * risk[n] for n in nodes], color=[risk[n] for n in nodes],
                    colorscale="RdYlGn_r", cmin=0, cmax=1, showscale=True, colorbar=dict(title="risk")),
        hovertemplate="%{text}<br>risk %{marker.color:.2f}<extra></extra>"))
    fig.update_layout(showlegend=False, height=520, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


eng = get_engine()
st.title("ARGUS-XAI Security Center")

with st.sidebar:
    st.header("Traffic source")
    seed = st.number_input("Simulation seed", 0, 9999, 7)
    n = st.slider("Flows", 2000, 20000, 6000, 1000)
    if st.button("Generate new traffic window") or "an" not in st.session_state:
        raw = generate_flows(n, seed=int(seed))
        st.session_state["truth"] = raw.set_index("flow_id")["label"]
        st.session_state["an"] = eng.analyze(raw.drop(columns=["label"]))
    min_risk = st.slider("Graph: minimum risk", 0.0, 1.0, float(eng.detector.threshold), 0.05)
    st.caption("Risk is a calibrated-to-benign anomaly score, not a probability of attack.")

an = st.session_state["an"]
f = an.flows
c1, c2, c3, c4 = st.columns(4)
c1.metric("Flows analysed", f"{len(f):,}")
c2.metric("Anomalous flows", int(f["is_anomaly"].sum()))
c3.metric("Unseen-behaviour alerts", int(f["verdict"].str.startswith("Anomalous").sum()))
hi = eng.detector.threshold + 0.5 * (1 - eng.detector.threshold)
c4.metric("HIGH-risk hosts", int((an.host_risk["max_risk"] >= hi).sum()))

tab1, tab2, tab3 = st.tabs(["Investigation graph", "Alerts & explanations", "Feedback / learning"])

with tab1:
    st.plotly_chart(investigation_graph(f, min_risk), use_container_width=True)
    host = st.selectbox("Inspect host", an.host_risk["src_ip"].tolist())
    h = f[f["src_ip"] == host]
    a, b = st.columns(2)
    a.write(an.host_risk[an.host_risk["src_ip"] == host].T)
    tl = go.Figure(go.Scatter(x=h["timestamp"], y=h["risk"], mode="markers",
                              marker=dict(color=h["risk"], colorscale="RdYlGn_r", cmin=0, cmax=1)))
    tl.update_layout(height=260, title="Risk timeline", margin=dict(l=0, r=0, t=30, b=0))
    b.plotly_chart(tl, use_container_width=True)

with tab2:
    alerts = f[f["is_anomaly"]].sort_values("risk", ascending=False)
    st.dataframe(alerts[["flow_id", "src_ip", "dst_ip", "dst_port", "risk", "risk_level", "verdict",
                         "pred_class", "trigger"]].head(200), use_container_width=True, height=300)
    if len(alerts):
        fid = st.selectbox("Explain flow", alerts["flow_id"].head(200).tolist())
        method = st.radio("Attribution", ["auto", "perturbation", "lime"], horizontal=True)
        pos = int(f.index[f["flow_id"] == fid][0])
        ex = eng.explain(an, pos, method)
        st.code(ex["narrative"])
        st.caption(f"Attribution method: {ex['method']}  |  components: "
                   + ", ".join(f"{k}={v:.2f}" for k, v in ex["components"].items()))
        if ex["top_features"]:
            tf = pd.DataFrame(ex["top_features"])
            st.plotly_chart(go.Figure(go.Bar(x=tf["attribution"], y=tf["description"], orientation="h"))
                            .update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0)), use_container_width=True)
        if "truth" in st.session_state:
            st.caption(f"(simulation ground truth for this flow: {st.session_state['truth'].get(fid)})")

with tab3:
    st.write("Label flagged flows as an analyst, then update the supervised head with replay.")
    unk = f[f["verdict"].str.startswith("Anomalous")]
    st.write(f"{len(unk)} unseen-behaviour alerts available.")
    label = st.text_input("Label for ALL unseen-behaviour alerts shown (e.g. WebAttack)", "")
    if st.button("Apply feedback and update model") and label and len(unk):
        res = eng.update_with_feedback(an.X[unk.index.to_numpy()], np.array([label] * len(unk)), "replay")
        st.success(f"Classes: {res['classes_before']} -> {res['classes_after']}; "
                   f"new threshold {res['threshold']:.3f}")
        st.session_state.pop("an")
        st.rerun()
    st.caption("Bulk labelling is a demo shortcut. In the API, feedback is per flow via /feedback.")
