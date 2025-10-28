
from fastapi.testclient import TestClient
from vct.api.app import app

def test_act_endpoint_ok():
    c = TestClient(app)
    r = c.post("/robot/act", json={"text":"сидіти","confidence":0.9,"reward_bias":0.5,"mood":0.0})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "result" in data
