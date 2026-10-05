"""Synthetic enterprise-network flow generator.

Purpose: let anyone run the *whole* pipeline offline (CI, Docker, demos) with realistic
structure: hosts, servers, bursty attacks, a periodic botnet beacon to a never-seen external host.
It is NOT a substitute for CICIDS2017 / CSE-CIC-IDS2018 / UNSW-NB15 for the reported research
numbers; it is a controlled testbed and a smoke-test fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ATTACK_CLASSES = ["DDoS", "BruteForce", "PortScan", "WebAttack", "Botnet"]
DEFAULT_MIX = {"BENIGN": 0.70, "DDoS": 0.08, "BruteForce": 0.06, "PortScan": 0.06,
               "WebAttack": 0.05, "Botnet": 0.05}
SERVICE_PORTS = [80, 443, 22, 53, 3306, 8080, 25, 445]
FLOW_COLUMNS = ["flow_id", "timestamp", "src_ip", "dst_ip", "dst_port", "protocol", "duration",
                "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes", "syn_flag_count", "label"]


def generate_flows(n_flows: int = 40000, seed: int = 42, duration_s: float = 3600.0,
                   mix: dict[str, float] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mix = mix or DEFAULT_MIX
    names = list(mix)
    p = np.array([mix[k] for k in names], dtype=float)
    counts = dict(zip(names, rng.multinomial(n_flows, p / p.sum())))

    clients = np.array([f"10.0.1.{i}" for i in range(10, 70)])
    servers = np.array([f"10.0.2.{i}" for i in range(1, 9)])
    srv_port = dict(zip(servers, SERVICE_PORTS))
    ddos_ips = np.array([f"203.0.113.{i}" for i in range(10, 26)])
    brute_ips = np.array(["198.51.100.20", "198.51.100.21"])
    scan_ips = np.array(["192.0.2.50"])
    web_ips = np.array(["198.51.100.77", "198.51.100.78"])
    bots = np.array(["10.0.1.15", "10.0.1.33", "10.0.1.48"])
    c2 = "192.0.2.99"

    def ln(mu, sd, n):
        return rng.lognormal(mu, sd, n)

    def frame(label, ts, src, dst, port, proto, dur, fp, bp, fb, bb, syn):
        return pd.DataFrame({
            "timestamp": np.clip(ts, 0, duration_s), "src_ip": src, "dst_ip": dst,
            "dst_port": np.asarray(port).astype(int), "protocol": proto, "duration": dur,
            "fwd_packets": np.asarray(fp).astype(int), "bwd_packets": np.asarray(bp).astype(int),
            "fwd_bytes": fb, "bwd_bytes": bb, "syn_flag_count": np.asarray(syn).astype(int),
            "label": label})

    parts = []
    n = counts.get("BENIGN", 0)
    if n:
        act = rng.dirichlet(np.ones(len(clients)) * 2)
        src = rng.choice(clients, n, p=act)
        dst = rng.choice(servers, n, p=[.30, .35, .05, .12, .06, .04, .04, .04])
        port = np.array([srv_port[d] for d in dst])
        icmp = rng.random(n) < 0.02
        udp = (port == 53) & ~icmp
        proto = np.where(icmp, "ICMP", np.where(udp, "UDP", "TCP"))
        port = np.where(icmp, 0, port)
        fp = 1 + rng.poisson(7, n); bp = 1 + rng.poisson(9, n)
        fp = np.where(udp | icmp, 1 + rng.poisson(.5, n), fp)
        bp = np.where(udp | icmp, 1 + rng.poisson(.5, n), bp)
        parts.append(frame("BENIGN", rng.uniform(0, duration_s, n), src, dst, port, proto,
                           ln(-1.0, 1.1, n), fp, bp, fp * ln(4.8, .5, n), bp * ln(6.3, .7, n),
                           (proto == "TCP").astype(int)))
    n = counts.get("DDoS", 0)
    if n:
        centers = rng.uniform(300, duration_s - 300, 4)
        ts = rng.choice(centers, n) + rng.normal(0, 15, n)
        fp = 1 + rng.poisson(1.5, n); bp = rng.poisson(.4, n)
        parts.append(frame("DDoS", ts, rng.choice(ddos_ips, n), np.full(n, servers[0]),
                           np.full(n, 80), np.full(n, "TCP"), ln(-5, .7, n), fp, bp,
                           fp * ln(4.2, .3, n), bp * ln(4.5, .3, n), np.ones(n)))
    n = counts.get("BruteForce", 0)
    if n:
        centers = rng.uniform(300, duration_s - 300, 3)
        ts = rng.choice(centers, n) + rng.uniform(-150, 150, n)
        fp = 9 + rng.poisson(3, n); bp = 9 + rng.poisson(3, n)
        parts.append(frame("BruteForce", ts, rng.choice(brute_ips, n), np.full(n, servers[2]),
                           np.full(n, 22), np.full(n, "TCP"), ln(-1.2, .25, n), fp, bp,
                           fp * ln(4.6, .2, n), bp * ln(4.9, .2, n), np.ones(n)))
    n = counts.get("PortScan", 0)
    if n:
        centers = rng.uniform(300, duration_s - 300, 2)
        ts = rng.choice(centers, n) + rng.uniform(-120, 120, n)
        dport = np.where(rng.random(n) < .1, rng.choice(SERVICE_PORTS, n), rng.integers(1, 65536, n))
        fp = 1 + rng.poisson(.3, n); bp = rng.poisson(.3, n)
        parts.append(frame("PortScan", ts, rng.choice(scan_ips, n), rng.choice(servers, n), dport,
                           np.full(n, "TCP"), ln(-6, .5, n), fp, bp, fp * ln(3.9, .2, n),
                           bp * 40.0, np.ones(n)))
    n = counts.get("WebAttack", 0)
    if n:
        centers = rng.uniform(400, duration_s - 400, 2)
        ts = rng.choice(centers, n) + rng.uniform(-250, 250, n)
        tgt = rng.choice([0, 1, 5], n)
        fp = 12 + rng.poisson(8, n); bp = 6 + rng.poisson(5, n)
        parts.append(frame("WebAttack", ts, rng.choice(web_ips, n), servers[tgt],
                           np.array(SERVICE_PORTS)[tgt], np.full(n, "TCP"), ln(0.3, .5, n), fp, bp,
                           fp * ln(6.0, .4, n), bp * ln(6.5, .6, n), np.ones(n)))
    n = counts.get("Botnet", 0)
    if n:
        ts = rng.integers(0, int(duration_s // 60), n) * 60 + rng.normal(0, 3, n)
        fp = 3 + rng.poisson(1, n); bp = 3 + rng.poisson(1, n)
        parts.append(frame("Botnet", ts, rng.choice(bots, n), np.full(n, c2),
                           rng.choice([443, 8080, 4444], n, p=[.5, .3, .2]), np.full(n, "TCP"),
                           ln(-1.5, .2, n), fp, bp, fp * ln(4.8, .15, n), bp * ln(5.0, .15, n),
                           np.ones(n)))

    df = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "flow_id", np.arange(len(df)))
    return df[FLOW_COLUMNS]
