import numpy as np

from argus.data import generate_flows
from argus.models.hybrid import UNKNOWN
from tests.helpers import small_engine


def test_detects_and_labels_open_set():
    eng, ctx, m = small_engine()
    assert m["all_attacks"]["recall"] > 0.8
    assert m["benign_fpr"] < 0.12                      # near the 5% target
    assert m["unseen_flagged"] > 0.7
    assert m["unseen_labelled_unknown"] > 0.5          # not forced into a known class
    assert m["known_correct_class"] > 0.9


def test_analyze_outputs_and_host_ranking():
    eng, _, _ = small_engine()
    an = eng.analyze(generate_flows(2500, seed=11).drop(columns=["label"]))
    f = an.flows
    assert {"risk", "verdict", "pred_class", "risk_level", "trigger"} <= set(f.columns)
    assert set(f.verdict) <= {"Normal", "Known attack", UNKNOWN}
    assert an.host_risk["max_risk"].is_monotonic_decreasing


def test_explanation_is_structured():
    eng, _, _ = small_engine()
    an = eng.analyze(generate_flows(2500, seed=12).drop(columns=["label"]))
    pos = int(np.flatnonzero(an.flows.is_anomaly.to_numpy())[0])
    ex = eng.explain(an, pos)
    assert ex["narrative"] and "components" in ex and "top_features" in ex
    assert ex["counterfactual"] is None or {"valid", "changes"} <= set(ex["counterfactual"])


def test_save_load_roundtrip(tmp_path):
    from argus.inference import ArgusEngine
    eng, _, _ = small_engine()
    eng.save(tmp_path)
    e2 = ArgusEngine.load(tmp_path)
    df = generate_flows(1500, seed=5).drop(columns=["label"])
    np.testing.assert_allclose(eng.analyze(df).flows["risk"], e2.analyze(df).flows["risk"])
