import pandas as pd

from argus.config import load_config
from argus.preprocessing import load_dataset


def test_cicids_style_csv(tmp_path):
    rows = []
    for i in range(200):
        rows.append({" Source IP": f"10.0.0.{i % 7}", " Destination IP": "10.0.1.1", " Destination Port": 80 + i % 3,
                     " Protocol": 6, " Timestamp": f"7/7/2017 3:{i % 60:02d}:00", " Flow Duration": 1_000_000 + i,
                     " Total Fwd Packets": 3, " Total Backward Packets": 2, " Total Length of Fwd Packets": 300,
                     " Total Length of Bwd Packets": 900, " SYN Flag Count": 1,
                     " Label": "BENIGN" if i % 5 else "DoS Hulk"})
    pd.DataFrame(rows).to_csv(tmp_path / "x.csv", index=False)
    cfg = load_config(overrides={"data": {"source": "cicids", "path": str(tmp_path), "n_flows": None}})
    df = load_dataset(cfg)
    assert set(df.label) == {"BENIGN", "DDoS"} and df.duration.max() < 2 and len(df) == 200
