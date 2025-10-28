from __future__ import annotations

from fastapi.testclient import TestClient

from vct.api.app import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_act_endpoint() -> None:
    client = TestClient(app)
    response = client.post(
        "/robot/act",
        json={"text": "сидіти", "confidence": 0.9},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["result"]["action"] in {"SIT", "NONE"}
