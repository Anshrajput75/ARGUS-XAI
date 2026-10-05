"""Flow-level and host-aggregate feature engineering (label-free except baseline fitting)."""
from __future__ import annotations

import numpy as np
import pandas as pd

SERVICE_PORTS = {80, 443, 22, 53, 3306, 8080, 25, 445}

FEATURE_DESCRIPTIONS = {
    "log_duration": "flow duration",
    "log_fwd_packets": "forward packet count",
    "log_bwd_packets": "backward packet count",
    "log_fwd_bytes": "forward bytes",
    "log_bwd_bytes": "backward bytes",
    "syn_flag_count": "SYN flag count",
    "bytes_per_packet": "average packet size",
    "log_pkt_ratio": "forward/backward packet ratio",
    "log_byte_ratio": "forward/backward byte ratio",
    "log_packets_per_sec": "packet rate",
    "log_bytes_per_sec": "byte rate",
    "log_iat_mean_ms": "mean inter-packet gap",
    "is_wellknown_port": "well-known destination port",
    "is_service_port": "destination port is a known service",
    "log_dst_port": "destination port number",
    "proto_tcp": "TCP protocol", "proto_udp": "UDP protocol", "proto_icmp": "ICMP protocol",
    "log_src_conn_count": "connection frequency of the source host",
    "log_src_unique_dsts": "number of distinct destinations contacted by the source",
    "log_src_unique_ports": "number of distinct destination ports probed by the source",
    "log_dst_conn_count": "connection frequency toward the destination",
    "log_dst_unique_srcs": "number of distinct sources contacting the destination",
    "dst_unseen": "destination host never seen in the baseline period",
}


class FeatureBuilder:
    def __init__(self, window: float = 60.0, baseline_frac: float = 0.25, use_novelty: bool = True):
        self.window = window
        self.baseline_frac = baseline_frac
        self.use_novelty = use_novelty
        self.known_dsts_: set[str] = set()
        self.feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        mask = df["timestamp"] <= df["timestamp"].quantile(self.baseline_frac)
        if "label" in df:
            mask &= df["label"] == "BENIGN"
        self.known_dsts_ = set(df.loc[mask, "dst_ip"].unique())
        self.feature_names_ = list(self.transform(df.head(50)).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df
        dur = d["duration"].clip(lower=1e-6).to_numpy(float)
        fp = d["fwd_packets"].to_numpy(float); bp = d["bwd_packets"].to_numpy(float)
        fb = d["fwd_bytes"].to_numpy(float); bb = d["bwd_bytes"].to_numpy(float)
        tp, tb = fp + bp, fb + bb
        port = d["dst_port"].to_numpy(float)
        X = pd.DataFrame(index=d.index)
        X["log_duration"] = np.log1p(dur * 1000)
        X["log_fwd_packets"] = np.log1p(fp)
        X["log_bwd_packets"] = np.log1p(bp)
        X["log_fwd_bytes"] = np.log1p(fb)
        X["log_bwd_bytes"] = np.log1p(bb)
        X["syn_flag_count"] = d["syn_flag_count"].to_numpy(float)
        X["bytes_per_packet"] = np.log1p(tb / np.maximum(tp, 1))
        X["log_pkt_ratio"] = np.log((fp + 1) / (bp + 1))
        X["log_byte_ratio"] = np.log((fb + 1) / (bb + 1))
        X["log_packets_per_sec"] = np.log1p(tp / dur)
        X["log_bytes_per_sec"] = np.log1p(tb / dur)
        X["log_iat_mean_ms"] = np.log1p(dur / np.maximum(tp, 1) * 1000)
        X["is_wellknown_port"] = ((port > 0) & (port < 1024)).astype(float)
        X["is_service_port"] = np.isin(port, list(SERVICE_PORTS)).astype(float)
        X["log_dst_port"] = np.log1p(port)
        proto = d["protocol"].to_numpy()
        X["proto_tcp"] = (proto == "TCP").astype(float)
        X["proto_udp"] = (proto == "UDP").astype(float)
        X["proto_icmp"] = (proto == "ICMP").astype(float)
        g = pd.DataFrame({"s": d["src_ip"].to_numpy(), "d": d["dst_ip"].to_numpy(),
                          "p": d["dst_port"].to_numpy(),
                          "w": np.floor(d["timestamp"].to_numpy() / self.window)}, index=d.index)
        gs = g.groupby(["s", "w"])
        gd = g.groupby(["d", "w"])
        X["log_src_conn_count"] = np.log1p(gs["d"].transform("size"))
        X["log_src_unique_dsts"] = np.log1p(gs["d"].transform("nunique"))
        X["log_src_unique_ports"] = np.log1p(gs["p"].transform("nunique"))
        X["log_dst_conn_count"] = np.log1p(gd["s"].transform("size"))
        X["log_dst_unique_srcs"] = np.log1p(gd["s"].transform("nunique"))
        if self.use_novelty:
            X["dst_unseen"] = (~d["dst_ip"].isin(self.known_dsts_)).astype(float)
        return X.astype(float)
