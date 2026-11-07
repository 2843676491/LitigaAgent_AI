"""SQLite 持久化层：案件 / 文书 / 条款切片 / 庭审 / 意见书 / 复核 / 审计。

生产可切换 MySQL/PostgreSQL，接口保持不变（Repository 风格）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from .config import settings

_LOCK = threading.RLock()
_CONN: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    facts TEXT NOT NULL,
    contract_text TEXT,
    contract_filename TEXT,
    clause_count INTEGER DEFAULT 0,
    owner TEXT,
    dept TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    filename TEXT,
    md5 TEXT,
    size INTEGER,
    path TEXT,
    chunk_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    clause_no TEXT,
    text TEXT NOT NULL,
    locator TEXT,
    tags TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hearings (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    status TEXT NOT NULL,
    route TEXT,
    round_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0,
    opinion_json TEXT,
    timeline_json TEXT,
    state_json TEXT,
    hitl_json TEXT,
    llm_calls INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hitl (
    id TEXT PRIMARY KEY,
    hearing_id TEXT NOT NULL,
    decision TEXT,
    reviewer TEXT,
    comment TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    hearing_id TEXT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_case ON chunks(case_id);
CREATE INDEX IF NOT EXISTS idx_audit_case ON audit(case_id, seq);
CREATE INDEX IF NOT EXISTS idx_hearings_case ON hearings(case_id);
"""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def get_conn() -> sqlite3.Connection:
    global _CONN
    with _LOCK:
        if _CONN is None:
            conn = sqlite3.connect(str(settings.db_path), check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(SCHEMA)
            conn.commit()
            _CONN = conn
        return _CONN


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    conn = get_conn()
    with _LOCK:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


# ---------------- 案件 ----------------

def create_case(
    title: str,
    facts: str,
    contract_text: str | None,
    contract_filename: str | None,
    owner: str,
    dept: str,
    clause_count: int = 0,
) -> dict[str, Any]:
    case_id = new_id("case")
    with transaction() as conn:
        conn.execute(
            "INSERT INTO cases(id,title,facts,contract_text,contract_filename,clause_count,owner,dept,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (case_id, title, facts, contract_text, contract_filename, clause_count, owner, dept, now_iso()),
        )
    return get_case(case_id)  # type: ignore[return-value]


def get_case(case_id: str) -> dict[str, Any] | None:
    conn = get_conn()
    with _LOCK:
        row = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    return _row_to_dict(row)


def list_cases(limit: int = 50, owner: str | None = None, dept: str | None = None) -> list[dict[str, Any]]:
    conn = get_conn()
    sql = "SELECT id,title,facts,contract_filename,clause_count,owner,dept,created_at FROM cases"
    params: list[Any] = []
    conds = []
    if owner:
        conds.append("owner=?")
        params.append(owner)
    if dept:
        conds.append("dept=?")
        params.append(dept)
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with _LOCK:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def save_document(case_id: str, filename: str, content: bytes, storage_path: str) -> dict[str, Any]:
    import hashlib

    md5 = hashlib.md5(content).hexdigest()
    doc_id = new_id("doc")
    with transaction() as conn:
        conn.execute(
            "INSERT INTO documents(id,case_id,filename,md5,size,path,created_at) VALUES(?,?,?,?,?,?,?)",
            (doc_id, case_id, filename, md5, len(content), storage_path, now_iso()),
        )
    return {"doc_id": doc_id, "md5": md5}


def save_chunks(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return
    with transaction() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO chunks(chunk_id,case_id,clause_no,text,locator,tags,created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            [
                (
                    c["chunk_id"],
                    c["case_id"],
                    c.get("clause_no"),
                    c["text"],
                    c.get("locator"),
                    json.dumps(c.get("tags", []), ensure_ascii=False),
                    now_iso(),
                )
                for c in chunks
            ],
        )
        conn.execute(
            "UPDATE cases SET clause_count=? WHERE id=?",
            (len(chunks), chunks[0]["case_id"]),
        )


def load_chunks(case_id: str | None = None) -> list[dict[str, Any]]:
    conn = get_conn()
    sql = "SELECT * FROM chunks"
    params: tuple = ()
    if case_id:
        sql += " WHERE case_id=?"
        params = (case_id,)
    with _LOCK:
        rows = conn.execute(sql, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["tags"] = json.loads(d.get("tags") or "[]")
        except json.JSONDecodeError:
            d["tags"] = []
        out.append(d)
    return out


# ---------------- 庭审 ----------------

def create_hearing(case_id: str, route: str = "dispute_simulation") -> dict[str, Any]:
    hearing_id = new_id("hear")
    with transaction() as conn:
        conn.execute(
            "INSERT INTO hearings(id,case_id,status,route,created_at,updated_at) VALUES(?,?,?,?,?,?)",
            (hearing_id, case_id, "created", route, now_iso(), now_iso()),
        )
    return get_hearing(hearing_id)  # type: ignore[return-value]


def update_hearing(hearing_id: str, **fields: Any) -> None:
    if not fields:
        return
    allowed = {
        "status", "route", "round_count", "confidence", "opinion_json",
        "timeline_json", "state_json", "hitl_json", "llm_calls",
    }
    payload = {k: v for k, v in fields.items() if k in allowed}
    payload["updated_at"] = now_iso()
    cols = ",".join(f"{k}=?" for k in payload)
    with transaction() as conn:
        conn.execute(f"UPDATE hearings SET {cols} WHERE id=?", (*payload.values(), hearing_id))


def get_hearing(hearing_id: str) -> dict[str, Any] | None:
    conn = get_conn()
    with _LOCK:
        row = conn.execute("SELECT * FROM hearings WHERE id=?", (hearing_id,)).fetchone()
    return _row_to_dict(row)


def list_hearings(case_id: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
    conn = get_conn()
    sql = "SELECT id,case_id,status,route,round_count,confidence,llm_calls,created_at FROM hearings"
    params: list[Any] = []
    if case_id:
        sql += " WHERE case_id=?"
        params.append(case_id)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with _LOCK:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# ---------------- 人工复核 ----------------

def add_hitl(hearing_id: str, decision: str, reviewer: str, comment: str) -> dict[str, Any]:
    hitl_id = new_id("hitl")
    with transaction() as conn:
        conn.execute(
            "INSERT INTO hitl(id,hearing_id,decision,reviewer,comment,created_at) VALUES(?,?,?,?,?,?)",
            (hitl_id, hearing_id, decision, reviewer, comment, now_iso()),
        )
    return {"hitl_id": hitl_id, "decision": decision, "reviewer": reviewer, "comment": comment, "created_at": now_iso()}


def get_hitl(hearing_id: str) -> dict[str, Any] | None:
    conn = get_conn()
    with _LOCK:
        row = conn.execute(
            "SELECT * FROM hitl WHERE hearing_id=? ORDER BY created_at DESC LIMIT 1", (hearing_id,)
        ).fetchone()
    return _row_to_dict(row)


# ---------------- 审计 ----------------

def insert_audit(
    case_id: str,
    hearing_id: str | None,
    actor: str,
    action: str,
    payload: dict[str, Any],
    prev_hash: str,
    hash_value: str,
    ts: str,
) -> int:
    with transaction() as conn:
        cur = conn.execute(
            "INSERT INTO audit(case_id,hearing_id,actor,action,payload_json,prev_hash,hash,ts)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (case_id, hearing_id, actor, action, json.dumps(payload, ensure_ascii=False, sort_keys=True),
             prev_hash, hash_value, ts),
        )
        return int(cur.lastrowid)


def last_audit(case_id: str | None = None) -> dict[str, Any] | None:
    conn = get_conn()
    sql = "SELECT * FROM audit"
    params: tuple = ()
    if case_id:
        sql += " WHERE case_id=?"
        params = (case_id,)
    sql += " ORDER BY seq DESC LIMIT 1"
    with _LOCK:
        row = conn.execute(sql, params).fetchone()
    return _row_to_dict(row)


def list_audit(case_id: str | None = None, hearing_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    conn = get_conn()
    sql = "SELECT * FROM audit"
    conds, params = [], []
    if case_id:
        conds.append("case_id=?")
        params.append(case_id)
    if hearing_id:
        conds.append("hearing_id=?")
        params.append(hearing_id)
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY seq ASC LIMIT ?"
    params.append(limit)
    with _LOCK:
        rows = conn.execute(sql, tuple(params)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["payload"] = json.loads(d.pop("payload_json"))
        except json.JSONDecodeError:
            d["payload"] = {}
        out.append(d)
    return out


def reset_for_test(db_file: str) -> None:
    """测试用：切换到独立的库文件。"""
    global _CONN
    with _LOCK:
        if _CONN is not None:
            _CONN.close()
        _CONN = None
    settings.db_path = __import__("pathlib").Path(db_file)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
