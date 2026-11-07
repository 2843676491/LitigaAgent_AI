"""接口层测试：用 TestClient 覆盖鉴权/ABAC、SSE 事件、HITL 幂等与驳回校验。"""

import json

import pytest
from fastapi.testclient import TestClient

from legalcourt.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _case(client, title="接口测试案件", facts="卖方逾期交付，我方主张违约金 30 万元；卖方称违约金过高请求调减。",
          user="alice", role="lawyer", dept="法务一部"):
    r = client.post("/api/v1/cases", json={"title": title, "facts": facts},
                    headers={"X-User": user, "X-Role": role, "X-Dept": "dept_a" if dept == "法务一部" else "dept_b"})
    assert r.status_code == 200, r.text
    return r.json()


def test_healthz_and_config(client):
    h = client.get("/healthz").json()
    assert h["status"] == "ok"
    assert h["index_size"] > 0
    assert h["hitl_threshold"] > 0
    cfg = client.get("/api/v1/config").json()
    assert "lawyer" in cfg["roles"]


def test_samples_available(client):
    samples = client.get("/api/v1/samples", headers={"X-Role": "lawyer"}).json()
    assert samples
    assert all("facts" in s for s in samples)


def test_cross_dept_access_denied(client):
    case = _case(client, user="alice", dept="法务一部")
    r = client.get(f"/api/v1/cases/{case['case_id']}",
                   headers={"X-User": "bob", "X-Role": "lawyer", "X-Dept": "dept_b"})
    assert r.status_code == 403


def test_admin_can_read_cross_dept(client):
    case = _case(client, user="alice", dept="法务一部")
    r = client.get(f"/api/v1/cases/{case['case_id']}",
                   headers={"X-User": "root", "X-Role": "admin", "X-Dept": "dept_b"})
    assert r.status_code == 200


def test_sse_stream_emits_timeline_and_done(client):
    case = _case(client)
    events = []
    with client.stream("POST", "/api/v1/hearings/stream",
                       json={"case_id": case["case_id"], "max_rounds": 2, "auto_review": True},
                       headers={"X-User": "alice", "X-Role": "lawyer"}) as r:
        assert r.status_code == 200
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                events.append((event, json.loads(line[5:].strip())))

    names = [e for e, _ in events]
    assert "init" in names
    assert "timeline" in names
    assert "done" in names
    assert any(e == "timeline" and d.get("node") == "plaintiff" for e, d in events)


def test_hitl_rejects_without_comment(client):
    case = _case(client)
    with client.stream("POST", "/api/v1/hearings/stream",
                       json={"case_id": case["case_id"], "max_rounds": 2, "auto_review": False},
                       headers={"X-User": "alice", "X-Role": "lawyer"}) as r:
        list(r.iter_lines())
    hearings = client.get(f"/api/v1/cases/{case['case_id']}/hearings",
                          headers={"X-Role": "lawyer"}).json()
    hearing_id = hearings[0]["id"]

    r = client.post(f"/api/v1/hearings/{hearing_id}/hitl",
                    json={"decision": "reject", "reviewer": "张复核", "comment": ""},
                    headers={"X-User": "reviewer", "X-Role": "reviewer"})
    assert r.status_code == 422


def test_hitl_conflict_when_not_awaiting(client):
    case = _case(client, facts="我方已交付全部货物，对方拖欠尾款 12 万元，经两次催告仍未支付。")
    with client.stream("POST", "/api/v1/hearings/stream",
                       json={"case_id": case["case_id"], "max_rounds": 2, "auto_review": True},
                       headers={"X-User": "alice", "X-Role": "lawyer"}) as r:
        list(r.iter_lines())
    hearings = client.get(f"/api/v1/cases/{case['case_id']}/hearings", headers={"X-Role": "lawyer"}).json()
    hearing_id = hearings[0]["id"]

    r = client.post(f"/api/v1/hearings/{hearing_id}/hitl",
                    json={"decision": "approve", "reviewer": "张复核", "comment": "ok"},
                    headers={"X-User": "reviewer", "X-Role": "reviewer"})
    assert r.status_code == 409


def test_audit_verify_endpoint(client):
    r = client.get("/api/v1/audit/verify", headers={"X-Role": "auditor"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_lawyer_cannot_decide_hitl(client):
    """RBAC：律师没有 hitl:decide 权限。"""
    case = _case(client)
    with client.stream("POST", "/api/v1/hearings/stream",
                       json={"case_id": case["case_id"], "max_rounds": 2, "auto_review": False},
                       headers={"X-User": "alice", "X-Role": "lawyer"}) as r:
        list(r.iter_lines())
    hearings = client.get(f"/api/v1/cases/{case['case_id']}/hearings", headers={"X-Role": "lawyer"}).json()
    hearing_id = hearings[0]["id"]
    r = client.post(f"/api/v1/hearings/{hearing_id}/hitl",
                    json={"decision": "approve", "reviewer": "alice", "comment": "ok"},
                    headers={"X-User": "alice", "X-Role": "lawyer"})
    assert r.status_code == 403
