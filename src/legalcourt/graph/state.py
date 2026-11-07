"""LangGraph 状态定义。

约束：state 必须可 JSON 序列化（Checkpointer 要落 SQLite），
因此不放 client / connection / 模型对象，只放纯数据。
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class HearingState(TypedDict, total=False):
    # 输入
    case_id: str
    hearing_id: str
    title: str
    facts: str
    contract_text: str
    contract_filename: str
    intent_hint: str | None
    auto_review: bool
    max_rounds: int

    # 路由
    route: str
    route_confidence: float
    route_reason: str

    # 证据与论点
    chunks: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    focus_candidates: list[dict[str, Any]]
    used_evidence: list[str]
    plaintiff_arguments: list[dict[str, Any]]
    defendant_arguments: list[dict[str, Any]]

    # 裁决
    focuses: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    missing_clauses: list[str]
    clause_risks: list[dict[str, Any]]
    summary: str
    inclination: str
    inclination_party: str
    remediation: list[str]
    risk_score: float
    risk_level: str
    confidence: float
    citations: list[str]

    # 收敛控制
    round: int
    new_evidence_count: int
    prev_gap: float
    converged: bool
    convergence_reason: str
    revision_count: int
    review_notes: list[str]

    # 运行元信息
    llm_calls: int
    hitl: dict[str, Any]
    status: str
    opinion: dict[str, Any]

    # 追加式
    timeline: Annotated[list[dict[str, Any]], add]
    errors: Annotated[list[str], add]
