import numpy as np
import pandas as pd

from argus.data import generate_flows
from argus.features import FeatureBuilder
from argus.graph import build_flow_graph, edge_context_features
from argus.preprocessing import normalize_labels, split_indices


def test_generator_schema_and_determinism():
    a, b = generate_flows(2000, seed=1), generate_flows(2000, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.label) == {"BENIGN", "DDoS", "BruteForce", "PortScan", "WebAttack", "Botnet"}
    assert a["timestamp"].is_monotonic_increasing


def test_features_finite_and_aligned():
    df = generate_flows(2000)
    X = FeatureBuilder().fit(df).transform(df)
    assert len(X) == len(df) and np.isfinite(X.to_numpy()).all()


def test_graph_edges_match_rows():
    df = generate_flows(1500)
    X = FeatureBuilder().fit(df).transform(df)
    g = build_flow_graph(df, X)
    assert g.n_edges == len(df)
    assert edge_context_features(g).shape[0] == len(df)


def test_open_set_split_hides_unknown_from_train():
    df = generate_flows(4000)
    idx = split_indices(df.label.values, ["DDoS", "BruteForce", "PortScan"], ["WebAttack", "Botnet"], 0)
    y = df.label.values
    assert not np.isin(y[idx["train"]], ["WebAttack", "Botnet"]).any()
    assert not np.isin(y[idx["val"]], ["WebAttack", "Botnet"]).any()
    assert np.isin(y[idx["test"]], ["WebAttack", "Botnet"]).any()
    assert len(set(idx["train"]) & set(idx["test"])) == 0


def test_label_normalisation():
    s = normalize_labels(pd.Series(["BENIGN", "DoS Hulk", "FTP-Patator", "Web Attack \x96 XSS", "Bot", "Heartbleed"]))
    assert s.tolist() == ["BENIGN", "DDoS", "BruteForce", "WebAttack", "Botnet", "Other"]
