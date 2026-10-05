"""Dataset loading (synthetic / CICIDS-style / UNSW-NB15-style) and open-set splitting."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from argus.data.synthetic import FLOW_COLUMNS, generate_flows

CICIDS_MAP = {
    "Destination Port": "dst_port", "Flow Duration": "duration", "Total Fwd Packets": "fwd_packets",
    "Total Backward Packets": "bwd_packets", "Total Length of Fwd Packets": "fwd_bytes",
    "Total Length of Bwd Packets": "bwd_bytes", "SYN Flag Count": "syn_flag_count",
    "Source IP": "src_ip", "Destination IP": "dst_ip", "Protocol": "protocol",
    "Timestamp": "timestamp", "Label": "label",
}
UNSW_MAP = {"srcip": "src_ip", "dstip": "dst_ip", "dsport": "dst_port", "proto": "protocol",
            "dur": "duration", "spkts": "fwd_packets", "dpkts": "bwd_packets",
            "sbytes": "fwd_bytes", "dbytes": "bwd_bytes", "stime": "timestamp",
            "attack_cat": "label"}
PROTO_NUM = {6: "TCP", 17: "UDP", 1: "ICMP"}


def normalize_labels(labels: pd.Series) -> pd.Series:
    """Collapse dataset-specific labels into the ARGUS taxonomy.

    NOTE: all DoS variants are merged into 'DDoS'; state this in any paper.
    """
    def f(x: str) -> str:
        low = str(x).strip().lower()
        if low in ("benign", "normal", "", "nan", "none"):
            return "BENIGN"
        if low.startswith("ddos") or low.startswith("dos") or low == "generic":
            return "DDoS"
        if "patator" in low or "brute" in low:
            return "BruteForce"
        if "portscan" in low or low in ("reconnaissance", "scan"):
            return "PortScan"
        if low.startswith("web attack") or low in ("exploits", "fuzzers", "analysis", "webattack"):
            return "WebAttack"
        if low in ("bot", "botnet", "backdoor", "backdoors"):
            return "Botnet"
        return "Other"
    return labels.map(f)


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["protocol"] = df["protocol"].map(
        lambda p: p.upper() if isinstance(p, str) else PROTO_NUM.get(int(p) if p == p else -1, "TCP"))
    df["protocol"] = df["protocol"].where(df["protocol"].isin(["TCP", "UDP", "ICMP"]), "TCP")
    if "syn_flag_count" not in df:
        df["syn_flag_count"] = (df["protocol"] == "TCP").astype(int)
    if not pd.api.types.is_numeric_dtype(df["timestamp"]):
        ts = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
        df["timestamp"] = (ts - ts.min()).dt.total_seconds()
    for c in ["timestamp", "duration", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
              "dst_port", "syn_flag_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", "duration", "dst_port"])
    df = df.fillna(0)
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df["label"] = normalize_labels(df["label"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "flow_id", np.arange(len(df)))
    return df[FLOW_COLUMNS]


def _load_csvs(path: str | Path, mapping: dict, duration_scale: float) -> pd.DataFrame:
    files = sorted(Path(path).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {path}. See data/README.md for download links.")
    frames = []
    for f in files:
        d = pd.read_csv(f, skipinitialspace=True, low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d.rename(columns=mapping))
    df = pd.concat(frames, ignore_index=True)
    if "src_ip" not in df or "dst_ip" not in df:
        warnings.warn("CSV has no IP columns (MachineLearningCVE-style): pseudo-hosts are "
                      "synthesized from ports, so graph results are NOT meaningful. Use the "
                      "GeneratedLabelledFlows variant.")
        port = pd.to_numeric(df["dst_port"], errors="coerce").fillna(0).astype(int)
        df["src_ip"] = "pseudo-src-" + (port % 50).astype(str)
        df["dst_ip"] = "pseudo-dst-" + (port % 8).astype(str)
    if "timestamp" not in df:
        df["timestamp"] = np.arange(len(df), dtype=float)
    if "protocol" not in df:
        df["protocol"] = "TCP"
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce") * duration_scale
    return _finish(df)


def load_dataset(cfg: dict, seed: int | None = None) -> pd.DataFrame:
    d = cfg["data"]
    seed = cfg["seed"] if seed is None else seed
    if d["source"] == "synthetic":
        return generate_flows(d["n_flows"], seed=seed)
    if d["source"] == "cicids":
        df = _load_csvs(d["path"], CICIDS_MAP, 1e-6)  # CICFlowMeter duration is in microseconds
    elif d["source"] == "unsw":
        df = _load_csvs(d["path"], UNSW_MAP, 1.0)
    else:
        raise ValueError(f"unknown data.source {d['source']}")
    n = d.get("n_flows")
    if n and len(df) > n:
        df = df.sample(n, random_state=seed).sort_values("timestamp").reset_index(drop=True)
        df["flow_id"] = np.arange(len(df))
    return df


def split_indices(y, known: list[str], unknown: list[str], seed: int,
                  test_size: float = 0.3, val_size: float = 0.15) -> dict[str, np.ndarray]:
    """Open-set split. Unknown attack classes appear ONLY in the test set."""
    y = np.asarray(y)
    known_idx = np.where(np.isin(y, ["BENIGN", *known]))[0]
    unknown_idx = np.where(np.isin(y, unknown))[0]
    trv, te = train_test_split(known_idx, test_size=test_size, stratify=y[known_idx],
                               random_state=seed)
    tr, va = train_test_split(trv, test_size=val_size / (1 - test_size), stratify=y[trv],
                              random_state=seed)
    return {"train": np.sort(tr), "val": np.sort(va),
            "test": np.sort(np.concatenate([te, unknown_idx]))}
