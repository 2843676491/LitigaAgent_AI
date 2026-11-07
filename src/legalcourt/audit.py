"""审计模块：哈希链式留痕。

每条记录 = SHA256(上一条哈希 | 业务字段 | 服务端密钥)。
单机部署也能证明"日志未被篡改/未被插入"，无需外部不可变存储。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from . import db
from .config import settings

GENESIS = "0" * 64


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _canonical(
    prev_hash: str,
    case_id: str,
    hearing_id: str | None,
    actor: str,
    action: str,
    payload: dict[str, Any],
    ts: str,
) -> str:
    body = "|".join(
        [
            prev_hash,
            case_id,
            hearing_id or "-",
            actor,
            action,
            json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ts,
            settings.audit_secret,
        ]
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def append(
    case_id: str,
    action: str,
    actor: str,
    payload: dict[str, Any] | None = None,
    hearing_id: str | None = None,
) -> dict[str, Any]:
    """追加一条审计记录，返回记录本身（含 hash）。"""
    prev = db.last_audit()          # 全局链：跨案件共用一条链，避免"删整个案件"绕过
    prev_hash = prev["hash"] if prev else GENESIS
    ts = now_iso()
    payload = payload or {}
    h = _canonical(prev_hash, case_id, hearing_id, actor, action, payload, ts)
    seq = db.insert_audit(case_id, hearing_id, actor, action, payload, prev_hash, h, ts)
    return {
        "seq": seq, "case_id": case_id, "hearing_id": hearing_id, "actor": actor,
        "action": action, "payload": payload, "prev_hash": prev_hash, "hash": h, "ts": ts,
    }


def verify_chain(case_id: str | None = None, limit: int = 5000) -> dict[str, Any]:
    """重算全链。返回 ok / checked / broken_at。"""
    records = db.list_audit(case_id=case_id, limit=limit)
    prev_hash = GENESIS
    for rec in records:
        expect_prev = rec.get("prev_hash", "")
        # 注意：按 case_id 过滤时链路是"全链的子集"，prev 可能指向被过滤掉的记录，
        # 因此只在全链校验时严格比对 prev_hash 连接关系。
        if case_id is None and expect_prev != prev_hash:
            return {
                "ok": False, "checked": rec["seq"], "broken_at": rec["seq"],
                "reason": "prev_hash 断裂，疑似删除或重排审计记录",
            }
        recalc = _canonical(
            expect_prev, rec["case_id"], rec.get("hearing_id"), rec["actor"],
            rec["action"], rec.get("payload", {}), rec["ts"],
        )
        if recalc != rec["hash"]:
            return {
                "ok": False, "checked": rec["seq"], "broken_at": rec["seq"],
                "reason": "hash 不匹配，记录内容被修改",
            }
        prev_hash = rec["hash"]
    return {"ok": True, "checked": len(records), "broken_at": None, "reason": None}
