import pytest

pytest.importorskip("fastapi"); pytest.importorskip("sqlalchemy"); pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from argus.data import generate_flows  # noqa: E402


def test_api_flow(tmp_path, monkeypatch):
    from tests.helpers import small_engine
    eng, _, _ = small_engine()
    eng.save(tmp_path / "m")
    monkeypatch.setenv("ARGUS_MODEL_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    from api.main import app
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"]
        flows = generate_flows(1500, seed=3).drop(columns=["label"]).to_dict("records")
        r = c.post("/analyze", json={"flows": flows}).json()
        assert r["n_anomalies"] > 0 and r["top_alerts"]
        fid = r["top_alerts"][0]["flow_id"]
        assert "narrative" in c.post("/explain", json={"flows": flows, "flow_id": fid}).json()
        assert c.get("/hosts/risk").json()
        assert c.post("/feedback", json={"flow_id": fid, "label": "WebAttack"}).status_code == 200
        assert c.post("/update", json={"strategy": "replay"}).status_code == 200
