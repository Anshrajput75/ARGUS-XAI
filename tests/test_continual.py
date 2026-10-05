import numpy as np

from argus.continual_learning import ContinualLearner, DriftDetector, ReplayBuffer, forgetting_summary, psi


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(2, 1, 5000)) > 0.25


def test_drift_detector_flags_shifted_stream():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(2000, 4)); new = rng.normal(loc=3, size=(500, 4))
    dd = DriftDetector().fit(ref, list("abcd"))
    assert dd.check(new).flagged and not dd.check(rng.normal(size=(500, 4))).flagged


def test_replay_buffer_is_class_balanced():
    b = ReplayBuffer(capacity=100)
    b.add(np.zeros((500, 2)), np.array(["a"] * 500)); b.add(np.ones((50, 2)), np.array(["b"] * 50))
    _, y = b.arrays()
    assert (y == "a").sum() <= 50 and (y == "b").sum() == 50


def test_replay_prevents_forgetting_naive_does_not():
    from argus.models import SupervisedDetector
    rng = np.random.default_rng(0)
    def blob(c, n=300): return rng.normal(c, 0.5, size=(n, 3))
    Xb, Xa, Xn = blob(0), blob(4), blob(-4)
    mk = lambda: SupervisedDetector(backend="random_forest", n_estimators=50)
    res = {}
    for strat in ("naive", "replay"):
        L = ContinualLearner(strat, 600)
        L.observe(np.vstack([Xb, Xa]), np.array(["BENIGN"] * 300 + ["A"] * 300))
        det = L.update(mk, np.vstack([blob(0, 100), Xn]), np.array(["BENIGN"] * 100 + ["N"] * 300))
        res[strat] = (det.predict(Xa) == "A").mean()
    assert res["replay"] > 0.9 and res["naive"] < 0.2


def test_forgetting_summary():
    hist = [{"A": 1.0, "B": 0.0}, {"A": 0.4, "B": 1.0}]
    s = forgetting_summary(hist, [["A"], ["B"]])
    assert abs(s["forgetting"]["A"] - 0.6) < 1e-9
