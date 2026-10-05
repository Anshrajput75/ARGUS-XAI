"""Configuration: built-in defaults merged with an optional YAML file and overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data": {
        "source": "synthetic", "path": "data/raw", "n_flows": 40000,
        "known_attacks": ["DDoS", "BruteForce", "PortScan"],
        "unknown_attacks": ["WebAttack", "Botnet"],
        "test_size": 0.30, "val_size": 0.15,
    },
    "features": {"window": 60.0, "baseline_frac": 0.25, "use_novelty": True},
    "models": {
        "supervised": {"backend": "auto", "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        "autoencoder": {"hidden": [32, 16, 8], "epochs": 30, "lr": 1e-3, "batch_size": 256,
                        "variational": False, "backend": "auto"},
        "graph": {"hidden": 32, "epochs": 80, "lr": 5e-3, "backend": "auto"},
    },
    "hybrid": {"fusion": "max", "weights": {"supervised": 1.0, "autoencoder": 1.0, "graph": 1.0},
               "target_fpr": 0.05, "known_class_confidence": 0.6, "z_offset": 2.0},
    "continual": {"replay_capacity": 4000, "feedback_per_class": 400, "drift_psi_threshold": 0.2},
    "artifacts_dir": "artifacts/model",
}


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    path = path or os.environ.get("ARGUS_CONFIG")
    if path is None and Path("configs/default.yaml").exists():
        path = "configs/default.yaml"
    if path and Path(path).exists():
        with open(path) as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        _merge(cfg, copy.deepcopy(overrides))
    return cfg
