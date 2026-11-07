import pytest

from legalcourt import audit


def test_append_and_verify_chain():
    audit.append("case_t1", "test.action", "pytest", {"n": 1})
    audit.append("case_t1", "test.action", "pytest", {"n": 2})
    result = audit.verify_chain()
    assert result["ok"] is True
    assert result["checked"] > 0


def test_tampered_payload_breaks_chain(monkeypatch):
    rows = audit.db.list_audit(limit=200)
    assert rows

    tampered = [dict(r) for r in rows]
    # 篡改最近一条的 payload，但不改 hash —— 校验必须能发现
    tampered[-1]["payload"] = {"n": 999}

    monkeypatch.setattr(audit.db, "list_audit", lambda **kw: tampered)
    result = audit.verify_chain()
    assert result["ok"] is False
    assert result["broken_at"] == tampered[-1]["seq"]


def test_genesis_hash_is_zeros():
    assert audit.GENESIS == "0" * 64
