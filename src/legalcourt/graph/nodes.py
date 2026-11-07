"""图节点实现：P1~P8 全流程。

设计原则：
1. 语义文本交给 LLM（主张/抗辩/焦点措辞/补强建议）；
2. 一切"可计算、可校验"的判断（强度、风险、置信、收敛、引用有效性）由代码完成，
   避免模型自由发挥污染评测口径；
3. 每个节点都写审计，且节点内 interrupt 之前不产生副作用（LangGraph 恢复时会重放）。
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langgraph.types import interrupt

from .. import audit, db
from ..config import settings
from ..llm.base import get_llm
from ..llm.mock import VALID_INTENTS
from ..llm.topics import detect_focus_candidates, missing_clauses, scan_clause_risks, topic_title
from ..rag.hybrid import get_retriever
from ..rag.ingest import split_contract
from ..schemas import (
    SCHEMA_VERSION,
    Argument,
    Conclusion,
    ConflictRecord,
    DisputeFocus,
    Evidence,
    FocusStatus,
    LegalOpinion,
    RiskLevel,
    utc_now_iso,
)
from .state import HearingState

MAX_EVIDENCE = 14


# ---------------- 工具函数 -----------------------------

def _event(state: HearingState, node: str, actor: str, title: str,
           detail: str = "", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "seq": len(state.get("timeline") or []) + 1,
        "node": node,
        "actor": actor,
        "title": title,
        "detail": detail,
        "payload": payload or {},
        "ts": utc_now_iso(),
    }


def _ev_dict(ev: Evidence) -> dict[str, Any]:
    d = ev.model_dump(mode="json")
    d["favor"] = (ev.meta or {}).get("favor")
    d["text"] = ev.snippet
    return d


def _ev_map(state: HearingState) -> dict[str, dict[str, Any]]:
    return {e["evidence_id"]: e for e in (state.get("evidence") or [])}


def _strength(basis: list[str], evmap: dict[str, dict[str, Any]]) -> float:
    """证据强度：法条(1)→1.0，司法解释(2)→0.75，类案(3)→0.5，合同(4)→0.25，事实(5)→0。"""
    if not basis:
        return 0.0
    vals = [(5 - int(evmap[b].get("tier", 5))) / 4.0 for b in basis if b in evmap]
    if not vals:
        return 0.0
    return round(min(1.0, sum(vals) / 1.8), 3)


def _args_of(args: list[dict[str, Any]], key: str, with_evidence: bool = True) -> list[dict[str, Any]]:
    out = [a for a in args if a.get("focus_key") == key]
    if with_evidence:
        out = [a for a in out if a.get("basis")]
    return out


def _mean_strength(args: list[dict[str, Any]]) -> float:
    return round(sum(float(a.get("strength", 0)) for a in args) / len(args), 3) if args else 0.0


# ---------------- P1/P2 文档治理 ----------------

def ingest(state: HearingState) -> dict[str, Any]:
    case_id = state["case_id"]
    facts = state.get("facts", "")
    contract = state.get("contract_text") or ""

    chunks = split_contract(contract, case_id, state.get("contract_filename")) if contract.strip() else []
    if chunks:
        db.save_chunks([c.to_dict() for c in chunks])
        index = get_retriever().index
        index.add_chunks([c.to_dict() for c in chunks])
        get_retriever().clear_cache()

    missing = missing_clauses(contract) if contract.strip() else ["书面合同文本"]
    candidates = detect_focus_candidates(f"{facts}\n{contract[:1500]}")

    audit.append(case_id, "ingest.done", "system:ingest",
                 {"clauses": len(chunks), "missing_clauses": missing, "focus_candidates": [c["key"] for c in candidates]},
                 hearing_id=state.get("hearing_id"))
    ev = _event(state, "ingest", "system", "文档治理与条款切片",
                f"识别 {len(chunks)} 个条款切片，{len(missing)} 项必备条款缺失",
                {"clauses": len(chunks), "missing": missing, "locators": [c.locator for c in chunks][:6]})
    return {
        "chunks": [c.to_dict() for c in chunks],
        "missing_clauses": missing,
        "focus_candidates": candidates,
        "timeline": [ev],
    }


# ---------------- P3 意图路由 ----------------

def orchestrate(state: HearingState) -> dict[str, Any]:
    llm = get_llm()
    res = llm.generate("route", {
        "facts": state.get("facts", "")[:1200],
        "contract": (state.get("contract_text") or "")[:1500],
        "intent_hint": state.get("intent_hint"),
    })
    data = res.data or {}
    route = data.get("intent")
    if route not in VALID_INTENTS:
        route = "contract_review" if state.get("contract_text") else "statute_consult"
    conf = float(data.get("confidence") or 0.6)
    reason = str(data.get("reason") or "")

    audit.append(state["case_id"], "orchestrate.route", "agent:orchestrator",
                 {"route": route, "confidence": conf, "reason": reason}, hearing_id=state.get("hearing_id"))
    ev = _event(state, "orchestrate", "Orchestrator", f"路由：{route}",
                f"置信度 {conf:.2f}；{reason}", {"route": route, "confidence": conf})
    return {
        "route": route, "route_confidence": conf, "route_reason": reason,
        "llm_calls": int(state.get("llm_calls", 0)) + res.calls,
        "timeline": [ev],
    }


# ---------------- P4 混合检索 ----------------

def retrieve(state: HearingState) -> dict[str, Any]:
    case_id = state["case_id"]
    retriever = get_retriever()
    focuses = state.get("focus_candidates") or []
    queries = [f"{f['title']} {f['key']}" for f in focuses]
    queries.append((state.get("facts", "")[:220] or (state.get("contract_text") or "")[:220]))

    evidence = retriever.search_multi(queries, top_k=settings.top_k_final, case_id=case_id, max_total=MAX_EVIDENCE)
    evs = [_ev_dict(e) for e in evidence]
    tiers = {t: sum(1 for e in evs if e["tier"] == t) for t in (1, 2, 3, 4, 5)}

    audit.append(case_id, "retrieve.search", "agent:retriever",
                 {"queries": len(queries), "returned": len(evs), "tier_dist": tiers,
                  "evidence_ids": [e["evidence_id"] for e in evs]}, hearing_id=state.get("hearing_id"))
    ev = _event(state, "retrieve", "检索 Agent", f"混合检索召回 {len(evs)} 条证据",
                f"法条 {tiers[1]} / 司法解释 {tiers[2]} / 类案 {tiers[3]} / 合同条款 {tiers[4]}",
                {"tier_dist": tiers, "returned": len(evs), "evidence": evs[:6]})
    return {"evidence": evs, "timeline": [ev]}


# ---------------- P5 主张 ----------------

def plaintiff(state: HearingState) -> dict[str, Any]:
    rnd = int(state.get("round", 0)) + 1
    llm = get_llm()
    evmap = _ev_map(state)
    payload = {
        "facts": state.get("facts", "")[:1200],
        "focuses": state.get("focus_candidates") or [],
        "evidence": [{k: e.get(k) for k in ("evidence_id", "locator", "title", "tags", "tier", "favor", "text")}
                     for e in (state.get("evidence") or [])],
        "round": rnd,
        "used_evidence": state.get("used_evidence", []),
        "prior_rebuttals": [a.get("claim") for a in (state.get("defendant_arguments") or [])][:6],
        "review_notes": state.get("review_notes", []),
    }
    res = llm.generate("plaintiff", payload)
    claims = (res.data or {}).get("arguments") or []

    args: list[dict[str, Any]] = []
    for i, c in enumerate(claims, start=1):
        basis = [b for b in (c.get("basis") or []) if b in evmap][:3]
        try:
            concl = Conclusion(c.get("conclusion") or ("support" if basis else "undetermined"))
        except ValueError:
            concl = Conclusion.UNDETERMINED
        if not basis:
            concl = Conclusion.UNDETERMINED
        args.append(
            Argument(
                argument_id=f"PA-R{rnd}-{i:02d}",
                side="plaintiff",
                claim=str(c.get("claim") or "")[:600],
                basis=basis,
                conclusion=concl,
                strength=_strength(basis, evmap),
                focus_key=str(c.get("focus_key") or ""),
                round=rnd,
                has_evidence=bool(basis),
            ).model_dump(mode="json")
        )

    used = list(state.get("used_evidence", []))
    new_ids = [b for b in {b for a in args for b in a["basis"]} if b not in used]
    used.extend(new_ids)

    audit.append(state["case_id"], "plaintiff.arguments", "agent:plaintiff",
                 {"round": rnd, "count": len(args), "new_evidence": len(new_ids)}, hearing_id=state.get("hearing_id"))
    ev = _event(state, "plaintiff", "原告/主张 Agent",
                f"第 {rnd} 轮主张：{len(args)} 条，新增证据 {len(new_ids)} 条",
                "；".join(a["claim"][:46] for a in args[:3]),
                {"round": rnd, "count": len(args), "arguments": args})
    return {
        "plaintiff_arguments": args,
        "round": rnd,
        "used_evidence": used,
        "new_evidence_count": len(new_ids),
        "llm_calls": int(state.get("llm_calls", 0)) + res.calls,
        "timeline": [ev],
    }


# ---------------- P6 抗辩 ----------------

def defendant(state: HearingState) -> dict[str, Any]:
    rnd = int(state.get("round", 1))
    llm = get_llm()
    evmap = _ev_map(state)
    payload = {
        "facts": state.get("facts", "")[:1200],
        "focuses": state.get("focus_candidates") or [],
        "plaintiff_arguments": state.get("plaintiff_arguments") or [],
        "evidence": [{k: e.get(k) for k in ("evidence_id", "locator", "title", "tags", "tier", "favor", "text")}
                     for e in (state.get("evidence") or [])],
        "round": rnd,
        "review_notes": state.get("review_notes", []),
    }
    res = llm.generate("defendant", payload)
    claims = (res.data or {}).get("arguments") or []
    valid_targets = {a["argument_id"] for a in (state.get("plaintiff_arguments") or [])}

    args: list[dict[str, Any]] = []
    for i, c in enumerate(claims, start=1):
        basis = [b for b in (c.get("basis") or []) if b in evmap][:3]
        try:
            concl = Conclusion(c.get("conclusion") or ("rebut" if basis else "undetermined"))
        except ValueError:
            concl = Conclusion.UNDETERMINED
        if not basis:
            concl = Conclusion.UNDETERMINED
        targets = [t for t in (c.get("targets") or []) if t in valid_targets]
        args.append(
            Argument(
                argument_id=f"DA-R{rnd}-{i:02d}",
                side="defendant",
                claim=str(c.get("claim") or "")[:600],
                basis=basis,
                conclusion=concl,
                strength=_strength(basis, evmap),
                targets=targets,
                focus_key=str(c.get("focus_key") or ""),
                round=rnd,
                has_evidence=bool(basis),
            ).model_dump(mode="json")
        )

    audit.append(state["case_id"], "defendant.arguments", "agent:defendant",
                 {"round": rnd, "count": len(args)}, hearing_id=state.get("hearing_id"))
    ev = _event(state, "defendant", "被告/抗辩 Agent",
                f"第 {rnd} 轮抗辩：{len(args)} 条（逐条反驳）",
                "；".join(a["claim"][:46] for a in args[:3]),
                {"round": rnd, "count": len(args), "arguments": args})
    return {
        "defendant_arguments": args,
        "llm_calls": int(state.get("llm_calls", 0)) + res.calls,
        "timeline": [ev],
    }


# ---------------- P7 裁决 + 收敛 ----------------

def judge(state: HearingState) -> dict[str, Any]:
    llm = get_llm()
    evmap = _ev_map(state)
    p_args = state.get("plaintiff_arguments") or []
    d_args = state.get("defendant_arguments") or []
    rnd = int(state.get("round", 1))
    max_rounds = int(state.get("max_rounds", settings.max_rounds))

    focuses_in: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for idx, fc in enumerate(state.get("focus_candidates") or [], start=1):
        key = fc["key"]
        p_f = _args_of(p_args, key)
        d_f = _args_of(d_args, key)
        p_strength = _mean_strength(p_f)
        d_strength = _mean_strength(d_f)
        p_eids = [b for a in p_f for b in a["basis"]]
        d_eids = [b for a in d_f for b in a["basis"]]

        # 同层级冲突：双方各自援引了"结论相反"的类案
        p_case_win = [e for e in p_eids if (evmap.get(e, {}).get("favor") == "plaintiff")]
        d_case_win = [e for e in d_eids if (evmap.get(e, {}).get("favor") == "defendant")]
        conflict: dict[str, Any] | None = None
        if p_case_win and d_case_win:
            has_high = any(int(evmap.get(e, {}).get("tier", 5)) <= 2 for e in (p_eids + d_eids))
            conflict = ConflictRecord(
                focus_key=key,
                evidence_a=p_case_win[0],
                evidence_b=d_case_win[0],
                reason="双方分别援引结论相反的类案，且同属类案层级（tier=3）",
                tier_a=int(evmap.get(p_case_win[0], {}).get("tier", 3)),
                tier_b=int(evmap.get(d_case_win[0], {}).get("tier", 3)),
                resolution="higher_tier" if has_high else "unresolved",
                resolved_by=("存在更高层级依据（法条/司法解释）" if has_high else None),
            ).model_dump(mode="json")

        if "书面合同文本" in (state.get("missing_clauses") or []):
            # 无书面合同：条款依据缺失，仅凭证言/聊天记录不足以对争点定性
            status = FocusStatus.UNDETERMINED
        elif p_strength <= 0 or d_strength <= 0:
            status = FocusStatus.UNDETERMINED
        elif conflict and conflict["resolution"] == "unresolved":
            status = FocusStatus.CONFLICT
        else:
            status = FocusStatus.RESOLVED

        if conflict:
            conflicts.append(conflict)

        focuses_in.append({
            "focus_id": f"F{idx}",
            "key": key,
            "title": fc.get("title") or topic_title(key),
            "plaintiff_strength": p_strength,
            "defendant_strength": d_strength,
            "status": status.value,
            "plaintiff_claim": (p_f[0]["claim"] if p_f else ""),
            "defendant_claim": (d_f[0]["claim"] if d_f else ""),
            "evidence_ids": sorted(set(p_eids + d_eids)),
        })

    res = llm.generate("judge", {
        "facts": state.get("facts", "")[:1200],
        "focuses": focuses_in,
        "missing_clauses": state.get("missing_clauses", []),
        "round": rnd,
        "max_rounds": max_rounds,
    })
    data = res.data or {}
    llm_focus = {f.get("key"): f for f in (data.get("focuses") or [])}

    for f in focuses_in:
        lf = llm_focus.get(f["key"]) or {}
        f["title"] = lf.get("title") or f["title"]
        f["plaintiff_view"] = str(lf.get("plaintiff_view") or f["plaintiff_claim"])[:300]
        f["defendant_view"] = str(lf.get("defendant_view") or f["defendant_claim"])[:300]
        f["note"] = str(lf.get("note") or "")

    n = len(focuses_in) or 1
    und_ratio = sum(1 for f in focuses_in if f["status"] == FocusStatus.UNDETERMINED.value) / n
    unres = [c for c in conflicts if c["resolution"] == "unresolved"]
    unresolved_ratio = len(unres) / n
    missing_ratio = min(1.0, len(state.get("missing_clauses", [])) / 6.0)

    p_total = sum(f["plaintiff_strength"] for f in focuses_in)
    d_total = sum(f["defendant_strength"] for f in focuses_in)
    total = p_total + d_total
    d_ratio = (d_total / total) if total > 0 else 0.5
    avg_gap = round((p_total - d_total) / n, 3)          # 平均每焦点强度差，取值 -1~1
    p_mean = round(p_total / n, 3)

    # 风险 = 我方举证不足 + 焦点无法定性 + 冲突未决 + 条款缺失 + 对方对抗压力
    risk_score = round(
        min(1.0,
            0.10
            + 0.30 * (1 - p_mean)
            + 0.35 * und_ratio
            + 0.15 * unresolved_ratio
            + 0.10 * missing_ratio
            + 0.15 * d_ratio
            + (0.15 if "书面合同文本" in (state.get("missing_clauses") or []) else 0.0)), 3
    )
    if risk_score >= 0.55:
        risk_level = RiskLevel.HIGH.value
    elif risk_score >= 0.35:
        risk_level = RiskLevel.MEDIUM.value
    else:
        risk_level = RiskLevel.LOW.value

    coverage = sum(1 for f in focuses_in if f["plaintiff_strength"] > 0 and f["defendant_strength"] > 0) / n
    avg_strength = sum(max(f["plaintiff_strength"], f["defendant_strength"]) for f in focuses_in) / n
    # 置信度 = 可判定性（强度差）+ 交锋覆盖 + 证据强度 - 冲突/举证不足惩罚
    confidence = round(
        max(0.20, min(0.95,
            0.55 + 0.20 * abs(avg_gap) + 0.15 * coverage + 0.10 * avg_strength
            - 0.22 * unresolved_ratio - 0.15 * und_ratio)), 3
    )

    if und_ratio >= 0.5 or unres:
        party = "undetermined"
    elif avg_gap > 0.12:
        party = "plaintiff"
    elif avg_gap < -0.12:
        party = "defendant"
    else:
        party = "undetermined"

    reason = str(data.get("inclination_reason") or "")
    party_label = {
        "plaintiff": "有利于原告/我方主张",
        "defendant": "有利于被告/对方抗辩",
        "undetermined": "证据不足或双方势均力敌，暂不作倾向判断",
    }[party]
    inclination = f"综合双方举证：{reason}；据此，本系统倾向性意见为「{party_label}」。"

    # ---- 收敛判据（三重）----
    converged = False
    conv_reason = ""
    new_count = int(state.get("new_evidence_count", 0))
    if rnd >= max_rounds:
        converged, conv_reason = True, f"达到轮次上限（{max_rounds} 轮）"
    elif rnd > 1 and new_count == 0 and und_ratio == 0:
        converged, conv_reason = True, "本轮无新增证据且各焦点双方均已交锋，继续辩论不产生新信息"
    elif rnd > 1 and new_count == 0:
        converged, conv_reason = True, "本轮无新增证据；剩余焦点属举证不足，继续辩论无法弥补（应转人工补证）"
    elif rnd > 1 and abs(avg_gap - float(state.get("prev_gap", 0.0))) < 0.05 and und_ratio == 0:
        converged, conv_reason = True, "双方证据强度差已收敛（变化 < 0.05）"
    else:
        conv_reason = f"第 {rnd} 轮仍有新增信息或焦点未充分交锋"

    audit.append(state["case_id"], "judge.verdict", "agent:judge",
                 {"round": rnd, "focuses": len(focuses_in), "conflicts": len(conflicts),
                  "risk": risk_level, "confidence": confidence, "converged": converged,
                  "inclination": party}, hearing_id=state.get("hearing_id"))
    ev = _event(state, "judge", "法官/仲裁 Agent",
                f"裁决：风险 {risk_level}（{risk_score:.2f}），置信 {confidence:.2f}",
                f"焦点 {len(focuses_in)} 项，冲突 {len(conflicts)} 项（未解决 {len(unres)}）；{conv_reason}",
                {"risk": risk_level, "risk_score": risk_score, "confidence": confidence,
                 "focuses": focuses_in, "conflicts": conflicts, "converged": converged,
                 "convergence_reason": conv_reason, "inclination_party": party})

    return {
        "focuses": focuses_in,
        "conflicts": conflicts,
        "summary": str(data.get("summary") or "")[:1200],
        "inclination": inclination,
        "inclination_party": party,
        "remediation": [str(x)[:200] for x in (data.get("remediation") or [])][:6],
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence": confidence,
        "converged": converged,
        "convergence_reason": conv_reason,
        "prev_gap": avg_gap,
        "llm_calls": int(state.get("llm_calls", 0)) + res.calls,
        "timeline": [ev],
    }


# ---------------- P7.5 引用校验 ----------------

def cite_guard(state: HearingState) -> dict[str, Any]:
    index = get_retriever().index
    p_args = state.get("plaintiff_arguments") or []
    d_args = state.get("defendant_arguments") or []
    dropped: list[dict[str, str]] = []
    citations: set[str] = set()

    def clean(arg: dict[str, Any]) -> dict[str, Any]:
        kept = [b for b in arg.get("basis", []) if index.exists(b)]
        lost = [b for b in arg.get("basis", []) if b not in kept]
        for b in lost:
            dropped.append({"argument_id": arg["argument_id"], "evidence_id": b})
        out = dict(arg)
        out["basis"] = kept
        out["has_evidence"] = bool(kept)
        if not kept:
            out["conclusion"] = Conclusion.UNDETERMINED.value
        citations.update(kept)
        return out

    p_clean = [clean(a) for a in p_args]
    d_clean = [clean(a) for a in d_args]

    penalty = min(0.24, 0.08 * len(dropped))
    confidence = round(max(0.2, float(state.get("confidence", 0.5)) - penalty), 3)

    audit.append(state["case_id"], "cite_guard.verify", "agent:cite_guard",
                 {"checked": sum(len(a.get("basis", [])) for a in p_args + d_args),
                  "dropped": len(dropped), "citations": sorted(citations)}, hearing_id=state.get("hearing_id"))
    ev = _event(state, "cite_guard", "引用校验 Agent",
                f"校验完成：剔除无效引用 {len(dropped)} 条" if dropped else "校验完成：全部引用可定位",
                f"有效引用 {len(citations)} 条；置信度调整至 {confidence:.2f}",
                {"dropped": dropped, "citations": sorted(citations)})

    return {
        "plaintiff_arguments": p_clean,
        "defendant_arguments": d_clean,
        "citations": sorted(citations),
        "confidence": confidence,
        "timeline": [ev],
    }


# ---------------- P8 人工复核闸门 ----------------

def hitl_gate(state: HearingState) -> dict[str, Any]:
    triggers: list[str] = []
    if float(state.get("confidence", 0)) < settings.hitl_confidence_threshold:
        triggers.append(f"置信度低于阈值（{state.get('confidence')} < {settings.hitl_confidence_threshold}）")
    if any(c.get("resolution") == "unresolved" for c in state.get("conflicts", [])):
        triggers.append("存在未解决的同层级证据冲突")
    if state.get("risk_level") == RiskLevel.HIGH.value:
        triggers.append("风险等级为高")
    if state.get("conflicts") and state.get("risk_level") in (RiskLevel.MEDIUM.value, RiskLevel.HIGH.value):
        # 即便按效力层级取舍过，取舍是否合理仍要有人看
        triggers.append(f"存在 {len(state['conflicts'])} 项证据冲突且风险为{state.get('risk_level')}，需复核取舍")

    if not triggers:
        return {"hitl": {"status": "skipped", "triggers": []}, "timeline": [
            _event(state, "hitl_gate", "HITL 闸门", "未命中复核条件，直接签署", "")
        ]}

    if bool(state.get("auto_review")):
        audit.append(state["case_id"], "hitl.auto_approved", "system:auto",
                     {"triggers": triggers}, hearing_id=state.get("hearing_id"))
        return {
            "hitl": {"status": "approved", "auto": True, "triggers": triggers,
                     "reviewer": "auto-review", "comment": "评测/测试模式自动通过"},
            "timeline": [_event(state, "hitl_gate", "HITL 闸门", "自动复核通过（评测模式）",
                                "；".join(triggers))],
        }

    # interrupt 之前只用纯计算，不写库（恢复时本节点会被重放）
    decision = interrupt({
        "hearing_id": state.get("hearing_id"),
        "case_id": state["case_id"],
        "risk_level": state.get("risk_level"),
        "confidence": state.get("confidence"),
        "triggers": triggers,
        "focuses": [
            {"title": f.get("title"), "status": f.get("status"),
             "plaintiff_strength": f.get("plaintiff_strength"), "defendant_strength": f.get("defendant_strength"),
             "note": f.get("note")}
            for f in (state.get("focuses") or [])
        ],
        "conflicts": state.get("conflicts", []),
        "missing_clauses": state.get("missing_clauses", []),
        "instruction": "请核验上述关键判断后签署（approve）或驳回（reject，必须填写理由）",
    })

    if not isinstance(decision, dict):
        decision = {"decision": "approve", "reviewer": "unknown", "comment": ""}
    act = str(decision.get("decision") or "approve").lower()
    reviewer = str(decision.get("reviewer") or "unknown")
    comment = str(decision.get("comment") or "")

    audit.append(state["case_id"], f"hitl.{act}", f"human:{reviewer}",
                 {"triggers": triggers, "comment": comment}, hearing_id=state.get("hearing_id"))

    if act == "reject":
        notes = list(state.get("review_notes", [])) + [f"[复核驳回/{reviewer}] {comment}"]
        return {
            "hitl": {"status": "rejected", "decision": "reject", "reviewer": reviewer,
                     "comment": comment, "triggers": triggers},
            "review_notes": notes,
            "revision_count": int(state.get("revision_count", 0)) + 1,
            "timeline": [_event(state, "hitl_gate", "HITL 闸门",
                                f"人工驳回（{reviewer}）", comment)],
        }

    return {
        "hitl": {"status": "approved", "decision": "approve", "reviewer": reviewer,
                 "comment": comment, "triggers": triggers},
        "timeline": [_event(state, "hitl_gate", "HITL 闸门", f"人工签署通过（{reviewer}）", comment)],
    }


# ---------------- 合同审查分支 ----------------

def clause_risk(state: HearingState) -> dict[str, Any]:
    llm = get_llm()
    chunks = state.get("chunks") or []
    rule_risks = scan_clause_risks(chunks)
    missing = state.get("missing_clauses", [])

    # 规则依据（法条/司法解释）直接从索引取正文，保证可溯源
    citations = []
    for rid in rule_risks:
        for b in rid.get("basis", []):
            if get_retriever().index.exists(b) and b not in citations:
                citations.append(b)

    res = llm.generate("clause_risk", {
        "rule_risks": rule_risks,
        "clauses": [{"clause_no": c.get("clause_no"), "text": c.get("text", "")[:200]} for c in chunks][:10],
        "missing_clauses": missing,
    })
    risks = (res.data or {}).get("risks") or []
    for r, src in zip(risks, rule_risks):
        r.setdefault("locator", src.get("locator"))

    level_rank = {"高": 3, "中": 2, "低": 1}
    top = max([level_rank.get(str(r.get("level")), 1) for r in risks] or [1])
    risk_level = {3: RiskLevel.HIGH.value, 2: RiskLevel.MEDIUM.value}.get(top, RiskLevel.LOW.value)
    risk_score = round(min(1.0, 0.12 * len(risks) + 0.05 * len(missing)), 3)
    confidence = 0.82 if risks or missing else 0.6

    remediation = [f"第{r.get('clause_no')}条：{r.get('suggestion')}" for r in risks][:6]
    if missing:
        remediation.append("补充缺失条款：" + "、".join(missing[:5]))

    audit.append(state["case_id"], "review.clause_risk", "agent:clause_reviewer",
                 {"risks": len(risks), "level": risk_level}, hearing_id=state.get("hearing_id"))
    ev = _event(state, "clause_risk", "合同审查 Agent",
                f"条款风险 {len(risks)} 处，风险等级 {risk_level}",
                "、".join(r.get("name", "") for r in risks[:4]),
                {"risks": risks, "missing": missing})

    return {
        "clause_risks": risks,
        "citations": citations,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "confidence": confidence,
        "summary": f"合同审查完成：命中 {len(risks)} 处条款风险，{len(missing)} 项必备条款缺失。",
        "remediation": remediation,
        "focuses": [],
        "llm_calls": int(state.get("llm_calls", 0)) + res.calls,
        "timeline": [ev],
    }


# ---------------- 条文咨询分支 ----------------

def consult_answer(state: HearingState) -> dict[str, Any]:
    llm = get_llm()
    answer = (llm.generate("consult", {"question": state.get("facts", "")[:800],
                                       "evidence": state.get("evidence") or []}).data or {}).get("answer", "")
    audit.append(state["case_id"], "consult.answer", "agent:consultant",
                 {"citations": len(state.get("evidence") or [])}, hearing_id=state.get("hearing_id"))
    ev = _event(state, "consult_answer", "条文咨询 Agent", "生成条文咨询答复", str(answer)[:120])
    return {
        "summary": str(answer)[:1200],
        "citations": [e["evidence_id"] for e in (state.get("evidence") or [])][:6],
        "confidence": 0.7,
        "risk_level": RiskLevel.LOW.value,
        "remediation": [],
        "focuses": [],
        "llm_calls": int(state.get("llm_calls", 0)) + 1,
        "timeline": [ev],
    }


# ---------------- 归档 ----------------

def finalize(state: HearingState) -> dict[str, Any]:
    case_id = state["case_id"]
    hearing_id = state.get("hearing_id", "")
    evmap = _ev_map(state)

    focuses = []
    for f in state.get("focuses") or []:
        focuses.append(DisputeFocus(
            focus_id=f.get("focus_id", ""),
            key=f.get("key", ""),
            title=f.get("title", ""),
            plaintiff_view=f.get("plaintiff_view", ""),
            defendant_view=f.get("defendant_view", ""),
            evidence_ids=f.get("evidence_ids", []),
            plaintiff_strength=float(f.get("plaintiff_strength", 0)),
            defendant_strength=float(f.get("defendant_strength", 0)),
            status=FocusStatus(f.get("status", "undetermined")),
            note=f.get("note", ""),
        ))

    p_args = [Argument(**a) for a in (state.get("plaintiff_arguments") or [])]
    d_args = [Argument(**a) for a in (state.get("defendant_arguments") or [])]
    conflicts = [ConflictRecord(**c) for c in (state.get("conflicts") or [])]
    snapshot = [Evidence(**{k: v for k, v in e.items() if k in Evidence.model_fields})
                for e in (state.get("evidence") or [])][:12]

    opinion = LegalOpinion(
        opinion_id=f"OP-{uuid.uuid4().hex[:10]}",
        case_id=case_id,
        hearing_id=hearing_id,
        intent=state.get("route", "dispute_simulation"),
        schema_version=SCHEMA_VERSION,
        summary=state.get("summary", ""),
        focuses=focuses,
        plaintiff_arguments=p_args,
        defendant_arguments=d_args,
        conflicts=conflicts,
        missing_clauses=state.get("missing_clauses", []),
        clause_risks=state.get("clause_risks", []),
        risk_level=RiskLevel(state.get("risk_level", RiskLevel.MEDIUM.value)),
        risk_score=float(state.get("risk_score", 0)),
        inclination=state.get("inclination", ""),
        inclination_party=state.get("inclination_party", "undetermined"),
        remediation=state.get("remediation", []),
        citations=state.get("citations", []),
        evidence_snapshot=snapshot,
        confidence=float(state.get("confidence", 0)),
        undetermined=(state.get("inclination_party", "undetermined") == "undetermined"),
        round_count=int(state.get("round", 0)),
        llm_calls=int(state.get("llm_calls", 0)),
        model=get_llm().model,
        human_review=state.get("hitl"),
    )

    payload = json.loads(opinion.model_dump_json())
    rec = audit.append(case_id, "opinion.finalized", "system:finalize",
                       {"opinion_id": opinion.opinion_id, "risk": opinion.risk_level.value,
                        "confidence": opinion.confidence, "focuses": len(focuses),
                        "citations": len(opinion.citations)}, hearing_id=hearing_id)
    opinion.audit_tail = rec["hash"]
    payload["audit_tail"] = rec["hash"]

    db.update_hearing(
        hearing_id,
        status="completed",
        confidence=opinion.confidence,
        round_count=opinion.round_count,
        llm_calls=opinion.llm_calls,
        opinion_json=json.dumps(payload, ensure_ascii=False),
        hitl_json=json.dumps(state.get("hitl") or {}, ensure_ascii=False),
        state_json=json.dumps(_jsonable(state), ensure_ascii=False),
        timeline_json=json.dumps(state.get("timeline") or [], ensure_ascii=False),
    )

    ev = _event(state, "finalize", "系统", "意见书签署归档",
                f"opinion_id={opinion.opinion_id}，审计哈希 {rec['hash'][:12]}…",
                {"opinion_id": opinion.opinion_id, "audit_hash": rec["hash"],
                 "risk": opinion.risk_level.value, "confidence": opinion.confidence})
    return {"opinion": payload, "status": "completed", "timeline": [ev]}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k not in {"timeline"}}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj
