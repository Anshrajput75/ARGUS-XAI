"""Helpers shared by experiment scripts: seed aggregation and result writing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_seeds(rows: list[dict], group_cols: list[str]) -> pd.DataFrame:
    """rows: one dict per (seed, condition). Returns mean ± std per condition."""
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in group_cols + ["seed"] and np.issubdtype(df[c].dtype, np.number)]
    g = df.groupby(group_cols, sort=False)[num]
    mean, std = g.mean(), g.std(ddof=0)
    out = mean.copy()
    for c in num:
        out[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c])]
    return out.reset_index()


def write_results(out_dir: str | Path, name: str, table: pd.DataFrame, raw: list[dict],
                  notes: str = "") -> None:
    out = Path(out_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}_raw.json").write_text(json.dumps(raw, indent=2, default=float))
    md = [f"# {name}\n", notes, "", table.to_markdown(index=False) if _has_tabulate() else table.to_string(index=False)]
    (out / f"{name}.md").write_text("\n".join(md))
    print(table.to_string(index=False))


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:
        return False
