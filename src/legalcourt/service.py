"""庭审运行服务：建案 → 起跑 → SSE 事件流 → 暂停/恢复 → 详情。

API 层只做协议转换，所有编排/持久化逻辑集中在这里，便于测试直接调用。
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from langgraph.types import Command

from . import audit, db
from .config import settings
from .graph.build import get_compiled_graph
from .schemas import HearingDetail, LegalOpinion, TimelineEvent


class HearingStateError(Exception):
    """状态不满足操作前提（如重复复核）。"""


def _cfg(hearing_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": hearing_id}, "recursion_limit": 40}


def _initial_state(case: dict[str, Any], hearing_id: str, max_rounds: int,
                   auto_review: bool, intent_hint: str | None) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "hearing_id": hearing_id,
        "title": case["title"],
        "facts": case["facts"],
        "contract_text": case.get("contract_text") or "",
        "contract_filename": case.get("contract_filename"),
        "intent_hint": intent_hint,
        "auto_review": auto_review,
        "max_rounds": max_rounds,
        "route": "dispute_simulation",
        "round": 0,
        "revision_count": 0,
        "prev_gap": 0.0,
        "llm_calls": 0,
        "used_evidence": [],
        "review_notes": [],
        "timeline": [],
        "conflicts": [],
        "focuses": [],
        "plaintiff_arguments": [],
        "defendant_arguments": [],
        "citations": [],
        "status": "running",
    }


def _persist(values: dict[str, Any], hearing_id: str, status: str | None = None) -> None:
    fields: dict[str, Any] = {
        "route": values.get("route"),
        "round_count": int(values.get("round", 0)),
        "confidence": float(values.get("confidence", 0) if values.get("confidence") is not None else 0.0),
        "llm_calls": int(values.get("llm_calls", 0)),
        "timeline_json": json.dumps(values.get("timeline") or [], ensure_ascii=False),
        "state_json": json.dumps({k: v for k, v in values.items() if k != "timeline"}, ensure_ascii=False, default=str),
    }
    if status:
        fields["status"] = status
    db.update_hearing(hearing_id, **fields)


def create_hearing(case_id: str) -> str:
    case = db.get_case(case_id)
    if not case:
        raise ValueError(f"案件不存在：{case_id}")
    hearing = db.create_hearing(case_id, route="dispute_simulation")
    audit.append(case_id, "hearing.created", "system", {"hearing_id": hearing["id"]}, hearing_id=hearing["id"])
    return hearing["id"]


def run_streaming(
    case_id: str,
    max_rounds: int | None = None,
    auto_review: bool = False,
    actor: str = "system",
    intent_hint: str | None = None,
) -> Iterator[dict[str, Any]]:
    """以事件流方式执行一次推演。事件：init / timeline / interrupt / awaiting / done / error。"""
    case = db.get_case(case_id)
    if not case:
        yield {"event": "error", "data": {"message": f"案件不存在：{case_id}"}}
        return

    max_rounds = int(max_rounds or settings.max_rounds)
    hearing_id = create_hearing(case_id)
    db.update_hearing(hearing_id, status="running")
    graph = get_compiled_graph()
    state = _initial_state(case, hearing_id, max_rounds, auto_review, intent_hint)

    yield {"event": "init", "data": {"hearing_id": hearing_id, "case_id": case_id,
                                    "max_rounds": max_rounds, "auto_review": auto_review,
                                    "llm_provider": settings.llm_provider}}

    try:
        for chunk in graph.stream(state, _cfg(hearing_id), stream_mode="updates"):
            for node_name, update in chunk.items():
                if node_name == "__interrupt__":
                    items = list(update) if update else []
                    payload = items[0].value if items else {}
                    db.update_hearing(
                        hearing_id, status="awaiting_review",
                        hitl_json=json.dumps({"status": "pending", **payload}, ensure_ascii=False, default=str),
                    )
                    audit.append(case_id, "hitl.pending", actor, {"triggers": payload.get("triggers")},
                                 hearing_id=hearing_id)
                    yield {"event": "interrupt", "data": payload}
                    continue
                for ev in (update or {}).get("timeline") or []:
                    yield {"event": "timeline", "data": ev}

        snap = graph.get_state(_cfg(hearing_id))
        values = dict(snap.values or {})
        if snap.next:  # 仍停在 interrupt 处
            _persist(values, hearing_id, status="awaiting_review")
            yield {"event": "awaiting", "data": {"hearing_id": hearing_id,
                                                 "triggers": (values.get("hitl") or {}).get("triggers"),
                                                 "confidence": values.get("confidence")}}
        else:
            _persist(values, hearing_id, status="completed")
            audit.append(case_id, "hearing.completed", actor,
                         {"hearing_id": hearing_id, "confidence": values.get("confidence")},
                         hearing_id=hearing_id)
            yield {"event": "done", "data": {"hearing_id": hearing_id,
                                            "status": values.get("status"),
                                            "confidence": values.get("confidence"),
                                            "risk_level": values.get("risk_level"),
                                            "round": values.get("round")}}
    except Exception as exc:  # noqa: BLE001 - 失败要落库、留痕，再抛给调用方
        db.update_hearing(hearing_id, status="failed")
        audit.append(case_id, "hearing.failed", actor, {"error": str(exc)}, hearing_id=hearing_id)
        yield {"event": "error", "data": {"message": str(exc), "hearing_id": hearing_id}}


def run_sync(case_id: str, max_rounds: int | None = None, auto_review: bool = True,
             intent_hint: str | None = None) -> dict[str, Any]:
    """评测/测试用：跑完整个图（遇到 interrupt 自动批准），返回最终 state。"""
    case = db.get_case(case_id)
    if not case:
        raise ValueError(f"案件不存在：{case_id}")
    hearing_id = create_hearing(case_id)
    graph = get_compiled_graph()
    state = _initial_state(case, hearing_id, int(max_rounds or settings.max_rounds), auto_review, intent_hint)
    result = graph.invoke(state, _cfg(hearing_id))
    for _ in range(settings.max_revisions + 1):
        if "__interrupt__" not in result:
            break
        result = graph.invoke(Command(resume={"decision": "approve", "reviewer": "auto-eval",
                                              "comment": "评测自动通过"}), _cfg(hearing_id))
    values = {k: v for k, v in result.items() if k != "__interrupt__"}
    _persist(values, hearing_id)
    return {"values": values, "hearing_id": hearing_id}


def resume(hearing_id: str, decision: dict[str, Any], actor: str = "reviewer") -> dict[str, Any]:
    row = db.get_hearing(hearing_id)
    if not row:
        raise ValueError(f"庭审不存在：{hearing_id}")
    if row["status"] != "awaiting_review":
        raise HearingStateError(f"当前状态为 {row['status']}，不接受复核")

    graph = get_compiled_graph()
    db.add_hitl(hearing_id, decision["decision"], decision["reviewer"], decision.get("comment", ""))
    result = graph.invoke(Command(resume=decision), _cfg(hearing_id))
    values = {k: v for k, v in result.items() if k != "__interrupt__"}
    pending = "__interrupt__" in result

    if pending:
        items = list(result["__interrupt__"])
        payload = items[0].value if items else {}
        db.update_hearing(hearing_id, status="awaiting_review",
                          hitl_json=json.dumps({"status": "pending", **payload}, ensure_ascii=False, default=str))
        _persist(values, hearing_id, status="awaiting_review")
    else:
        _persist(values, hearing_id, status="completed")
        audit.append(row["case_id"], "hearing.completed", actor, {"hearing_id": hearing_id}, hearing_id=hearing_id)
    return {"pending": pending, "values": values}


def get_detail(hearing_id: str) -> HearingDetail:
    row = db.get_hearing(hearing_id)
    if not row:
        raise ValueError(f"庭审不存在：{hearing_id}")

    def _load(raw: str | None, default):
        if not raw:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    opinion_raw = _load(row.get("opinion_json"), None)
    opinion = LegalOpinion(**opinion_raw) if opinion_raw else None
    timeline = [TimelineEvent(**e) for e in (_load(row.get("timeline_json"), []) or [])]
    hitl = _load(row.get("hitl_json"), None)

    return HearingDetail(
        hearing_id=row["id"],
        case_id=row["case_id"],
        status=row["status"],
        route=row.get("route") or "dispute_simulation",
        round_count=int(row.get("round_count") or 0),
        confidence=float(row.get("confidence") or 0),
        opinion=opinion,
        timeline=timeline,
        hitl=hitl,
        llm_calls=int(row.get("llm_calls") or 0),
        created_at=row.get("created_at", ""),
        updated_at=row.get("updated_at", ""),
    )
