"""Pydantic 数据契约：证据、论点、争议焦点、冲突、法律意见书、接口出入参。

设计约束：
1. 意见书必须能序列化成 JSON 并过 Schema（前端、入库、审计共用同一份契约）。
2. 任何"结论"都必须能追到 evidence_id；没有证据时只能是 undetermined。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class SourceType(str, Enum):
    STATUTE = "statute"
    JUDICIAL_INTERPRETATION = "judicial_interpretation"
    CASE = "case"
    CONTRACT = "contract"
    FACTS = "facts"


# 证据效力层级：数值越小越优先（法条 > 司法解释 > 类案 > 事实陈述）
AUTHORITY_TIER: dict[SourceType, int] = {
    SourceType.STATUTE: 1,
    SourceType.JUDICIAL_INTERPRETATION: 2,
    SourceType.CASE: 3,
    SourceType.CONTRACT: 4,
    SourceType.FACTS: 5,
}

SOURCE_LABEL: dict[SourceType, str] = {
    SourceType.STATUTE: "法律条文",
    SourceType.JUDICIAL_INTERPRETATION: "司法解释",
    SourceType.CASE: "类案裁判",
    SourceType.CONTRACT: "合同条款",
    SourceType.FACTS: "事实陈述",
}


class RiskLevel(str, Enum):
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"


class Conclusion(str, Enum):
    SUPPORT = "support"
    REBUT = "rebut"
    UNDETERMINED = "undetermined"


class FocusStatus(str, Enum):
    RESOLVED = "resolved"          # 双方均有证据，强度可比，可给倾向
    CONFLICT = "conflict"          # 同层级证据结论相反，无法直接取舍
    UNDETERMINED = "undetermined"  # 至少一方缺证据


class Evidence(BaseModel):
    """检索命中的一条证据。locator 用于精确回溯到"哪一条"。"""

    evidence_id: str
    source_type: SourceType
    title: str
    locator: str                      # 例：民法典#第585条 / 合同#第7条 /（2021）京03民终1234号
    snippet: str
    score: float = 0.0
    tier: int = 5
    tags: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def tier_label(self) -> str:
        return SOURCE_LABEL.get(self.source_type, "其他")


class Argument(BaseModel):
    argument_id: str
    side: Literal["plaintiff", "defendant"]
    claim: str
    basis: list[str] = Field(default_factory=list)    # evidence_id 列表
    conclusion: Conclusion = Conclusion.SUPPORT
    strength: float = 0.0
    targets: list[str] = Field(default_factory=list)   # 抗辩所针对的对方 argument_id
    focus_key: str = ""
    round: int = 1
    has_evidence: bool = True


class ConflictRecord(BaseModel):
    focus_key: str
    evidence_a: str
    evidence_b: str
    reason: str
    tier_a: int
    tier_b: int
    resolution: Literal["higher_tier", "relevance", "unresolved"] = "unresolved"
    resolved_by: str | None = None


class DisputeFocus(BaseModel):
    focus_id: str
    key: str
    title: str
    plaintiff_view: str = ""
    defendant_view: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    plaintiff_strength: float = 0.0
    defendant_strength: float = 0.0
    status: FocusStatus = FocusStatus.UNDETERMINED
    note: str = ""


class LegalOpinion(BaseModel):
    """最终产物：结构化法律意见书（可入库、可复核、可审计）。"""

    opinion_id: str
    case_id: str
    hearing_id: str
    intent: str = "dispute_simulation"
    schema_version: str = SCHEMA_VERSION
    generated_at: str = Field(default_factory=utc_now_iso)

    summary: str = ""
    focuses: list[DisputeFocus] = Field(default_factory=list)
    plaintiff_arguments: list[Argument] = Field(default_factory=list)
    defendant_arguments: list[Argument] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    missing_clauses: list[str] = Field(default_factory=list)
    clause_risks: list[dict[str, Any]] = Field(default_factory=list)   # 合同审查分支的条款风险清单

    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_score: float = 0.0
    inclination: str = ""
    inclination_party: Literal["plaintiff", "defendant", "undetermined"] = "undetermined"
    remediation: list[str] = Field(default_factory=list)

    citations: list[str] = Field(default_factory=list)      # evidence_id 汇总
    evidence_snapshot: list[Evidence] = Field(default_factory=list)
    confidence: float = 0.0
    undetermined: bool = True

    round_count: int = 0
    llm_calls: int = 0
    model: str = "mock"

    human_review: dict[str, Any] | None = None
    audit_tail: str | None = None       # 落库时写入的审计哈希，便于反查

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("risk_score")
    @classmethod
    def _clamp_risk(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


# ---------------- 接口出入参 ----------------


class CaseCreate(BaseModel):
    title: str = Field(min_length=2, max_length=80)
    facts: str = Field(min_length=10, description="争议事实陈述")
    contract_text: str | None = Field(default=None, description="合同全文；也可走文件上传")
    contract_filename: str | None = None
    intent_hint: str | None = None    # 用户可指定，留空由 Orchestrator 路由
    dept: str = "法务一部"


class CaseView(BaseModel):
    case_id: str
    title: str
    facts: str
    contract_filename: str | None = None
    clause_count: int = 0
    created_at: str
    owner: str
    dept: str


class HearingRequest(BaseModel):
    case_id: str
    max_rounds: int | None = None
    auto_review: bool = False    # True 时跳过 HITL（仅评测/测试用，生产应保持 False）


class HitlDecision(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=2)
    comment: str = ""

    @field_validator("comment")
    @classmethod
    def _reject_needs_reason(cls, v: str, info) -> str:
        decision = (info.data or {}).get("decision")
        if decision == "reject" and not v.strip():
            raise ValueError("驳回必须填写复核理由（comment）")
        return v


class TimelineEvent(BaseModel):
    seq: int
    node: str
    actor: str
    title: str
    detail: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: str = Field(default_factory=utc_now_iso)


class HearingDetail(BaseModel):
    hearing_id: str
    case_id: str
    status: Literal["created", "running", "awaiting_review", "completed", "failed"]
    route: str = "dispute_simulation"
    round_count: int = 0
    confidence: float = 0.0
    opinion: LegalOpinion | None = None
    timeline: list[TimelineEvent] = Field(default_factory=list)
    hitl: dict[str, Any] | None = None
    llm_calls: int = 0
    created_at: str = ""
    updated_at: str = ""


class SearchHit(BaseModel):
    evidence: Evidence


class AuditRecord(BaseModel):
    seq: int
    case_id: str
    hearing_id: str | None
    actor: str
    action: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str
    ts: str


class ApiError(BaseModel):
    error: str
    message: str
    detail: dict[str, Any] | None = None
