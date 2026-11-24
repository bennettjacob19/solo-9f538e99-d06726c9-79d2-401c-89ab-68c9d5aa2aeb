"""FastAPI 端到端测试。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app, SESSIONS

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def test_health_and_docs():
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_invalid_inputs_rejected():
    # 超过 24 批。
    r = client.post("/plans", json={
        "durations": [1] * 25,
        "evaluations": [],
        "max_checkpoints": 0,
        "max_rerun_batches": 0,
    })
    assert r.status_code == 422

    # 非正整数耗时。
    r = client.post("/plans", json={
        "durations": [3, 0],
        "max_checkpoints": 0,
        "max_rerun_batches": 0,
    })
    assert r.status_code == 422

    # 超过 4 场评测。
    r = client.post("/plans", json={
        "durations": [2] * 10,
        "evaluations": [
            {"eval_id": i, "start": 10 * i, "end": 10 * i + 2,
             "min_checkpoint_batches": 1}
            for i in range(5)
        ],
        "max_checkpoints": 5,
        "max_rerun_batches": 50,
    })
    assert r.status_code == 422

    # 重叠评测。
    r = client.post("/plans", json={
        "durations": [2] * 10,
        "evaluations": [
            {"eval_id": 1, "start": 4, "end": 8,
             "min_checkpoint_batches": 1},
            {"eval_id": 2, "start": 6, "end": 9,
             "min_checkpoint_batches": 1},
        ],
        "max_checkpoints": 5,
        "max_rerun_batches": 50,
    })
    assert r.status_code == 422


def test_infeasible_plan_is_not_started():
    r = client.post("/plans", json={
        "durations": [2] * 5,
        "evaluations": [
            {"eval_id": 1, "start": 10, "end": 12,
             "min_checkpoint_batches": 1}
        ],
        "max_checkpoints": 0,
        "max_rerun_batches": 10,
    })
    assert r.status_code == 422
    body = r.json()["detail"]
    assert body["accepted"] is False
    assert "未启动" in body["message"]
    assert SESSIONS == {}


def test_feasible_plan_full_lifecycle():
    r = client.post("/plans", json={
        "durations": [3] * 8,
        "evaluations": [
            {"eval_id": 1, "start": 12, "end": 14,
             "min_checkpoint_batches": 2},
            {"eval_id": 2, "start": 30, "end": 32,
             "min_checkpoint_batches": 3},
        ],
        "max_checkpoints": 1,
        "max_rerun_batches": 20,
    })
    assert r.status_code == 201
    body = r.json()
    assert body["accepted"] is True
    sid = body["session_id"]
    assert sid in SESSIONS

    # 分多次推进直到完成。
    finished = False
    for _ in range(200):
        out = client.post(
            f"/sessions/{sid}/advance?seconds=1"
        ).json()
        finished = out["finished"]
        if finished:
            break
    assert finished

    status = client.get(f"/sessions/{sid}").json()
    kinds = [e["kind"] for e in status["executed_segments"]]
    assert "evaluation" in kinds and "checkpoint" in kinds
    assert status["reruns_executed"] == body["reruns_total"]

    results = client.get(f"/sessions/{sid}/results").json()
    assert results["consistent_with_uninterrupted"] is True
    assert results["max_abs_diff"] == 0.0
    assert len(results["final_weights"]) == 5

    listed = client.get("/sessions").json()["sessions"]
    assert any(s["session_id"] == sid for s in listed)


def test_advance_requires_positive_seconds():
    r = client.post("/plans", json={
        "durations": [3, 3],
        "max_checkpoints": 0,
        "max_rerun_batches": 0,
    })
    sid = r.json()["session_id"]
    assert client.post(f"/sessions/{sid}/advance?seconds=0").status_code == 400
    assert client.post("/sessions/nope/advance?seconds=1").status_code == 404
