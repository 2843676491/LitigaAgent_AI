"""Mock LLM：把"证据 → 论点 → 裁决"的推理规则显式化。

它不是随机假数据：同一输入必然得到同一输出，因此可以离线跑通全链路、
做回归测试与评测基线。规则生成的所有论点都必须挂真实 evidence_id，
无证据时一律输出 undetermined——这与生产 prompt 的约束完全一致。
"""

from __future__ import annotations

from typing import Any

from .base import LLMResult
from .topics import evidence_matches_topic, topic_title

VALID_INTENTS = ("dispute_simulation", "contract_review", "statute_consult")

PLAINTIFF_TEXTS: dict[str, str] = {
    "违约金": "我方主张按合同约定标准计付违约金；该标准系双方协商确定，对方主张过分高于损失的，应由其就损失数额承担举证责任。",
    "逾期交付": "我方主张对方构成迟延履行主要债务，经书面催告后仍未履行，应承担继续履行或赔偿损失等违约责任。",
    "质量瑕疵": "我方主张交付标的物不符合约定质量标准并已影响合同目的实现，有权主张修复、减价或解除合同。",
    "合同解除": "我方主张解除条件已成就，解除通知已送达，对方应承担返还已付款项并赔偿损失的责任。",
    "付款义务": "我方主张对方未按约支付价款，应支付到期款项并承担逾期付款的违约责任。",
    "格式条款": "我方主张涉案免责条款未尽提示说明义务，不应成为合同内容；且存在不合理免除提供方责任的情形，应认定无效。",
    "损失赔偿": "我方主张赔偿范围应包含已实际支出的损失与可得利益损失，且未超出对方订立合同时的可预见范围。",
    "减损义务": "我方主张已采取合理措施防止损失扩大，扩大部分不应由我方承担。",
    "不可抗力": "我方主张对方以不可抗力免责不能成立：其未及时通知、未提供有效证明，亦未采取替代履行方案。",
    "情势变更": "我方主张价格波动属正常商业风险，不构成情势变更，应按原约定继续履行。",
    "分期付款": "我方主张欠款比例已达到法定标准，经催告仍不支付，解除条件已成就。",
    "验收": "我方主张对方未按约定组织验收即投入使用，应视为验收合格，其质量异议不能成立。",
    "建设工程": "我方主张工程款已到期且工程质量合格，承包人依法享有建设工程价款优先受偿权。",
    "举证责任": "我方主张对方对其抗辩所依据的事实负举证责任，举证不能应承担不利后果。",
}

DEFENDANT_TEXTS: dict[str, str] = {
    "违约金": "约定违约金已明显超出对方可举证的实际损失，依民法典第五百八十五条第二款及相关司法解释应予以适当调减。",
    "逾期交付": "逾期系对方未按期提供安装场地、未及时确认图纸所致，且对方自身存在逾期付款，双方均有过错，应相应减免我方责任。",
    "质量瑕疵": "涉案瑕疵属可修复缺陷，未致合同目的不能实现，对方主张解除缺乏依据，仅应支持合理修复费用及停机损失。",
    "合同解除": "解除条件并未成就：欠款比例未达法定标准且未经催告，对方无权单方解除合同。",
    "付款义务": "付款义务与对方先履行义务相互关联，我方依法享有先履行抗辩权，不构成违约。",
    "格式条款": "涉案条款系双方协商确定而非格式条款，且已作合理提示，应认定有效。",
    "损失赔偿": "对方主张的停产损失与违约行为无直接因果关系，且其未履行减损义务，扩大部分不应支持。",
    "减损义务": "对方未及时采购替代货物或安排替代产线，扩大损失应由其自行承担。",
    "不可抗力": "该事件客观上导致履行不能，应依不可抗力规则部分或全部免除我方责任。",
    "情势变更": "合同基础条件发生当事人无法预见的重大变化，继续履行明显不公平，应变更价款。",
    "分期付款": "欠款未达全部价款五分之一，不符合法定解除条件，仅应支付到期欠款。",
    "验收": "对方未在约定异议期内提出书面异议，依约应视为验收合格。",
    "建设工程": "工程款结算尚未完成且质量存在争议，优先受偿范围不应包含违约金与损害赔偿。",
    "举证责任": "对方对其主张的损失数额未提供有效凭证，应承担举证不能的后果。",
}


def _clip(text: str, n: int = 90) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _sid(ev: dict[str, Any]) -> str:
    return str(ev.get("evidence_id") or ev.get("id"))


def _pick(
    evidence: list[dict[str, Any]],
    key: str,
    exclude: set[str] | None = None,
    prefer: str = "plaintiff",
    limit: int = 2,
) -> list[dict[str, Any]]:
    """按焦点+立场挑选证据，并刻意为双方各保留一条"同层级类案"。

    若只挑法条，冲突检测永远不会触发——而真实争议里最常见的正是
    "同一争点、两份判决结论相反"。因此这里做**结构化配比**：
    最高层级依据 + 己方立场类案 + 相关合同条款。
    """
    exclude = exclude or set()
    cand = [
        e for e in evidence
        if evidence_matches_topic(e.get("tags", []), e.get("title", ""), key)
        and _sid(e) not in exclude
    ]
    if not cand:
        return []

    chosen: list[dict[str, Any]] = []

    def _tier(e: dict[str, Any]) -> int:
        return int(e.get("tier", 5))

    high = sorted([e for e in cand if _tier(e) <= 2], key=_tier)
    if high:
        chosen.append(high[0])

    same_side_cases = [e for e in cand if _tier(e) == 3 and (e.get("favor") or "neutral") == prefer]
    if same_side_cases:
        chosen.append(same_side_cases[0])

    clauses = [e for e in cand if _tier(e) == 4]
    if len(chosen) < limit and clauses:
        chosen.append(clauses[0])

    for e in sorted(cand, key=_tier):
        if len(chosen) >= limit:
            break
        if all(e is not c for c in chosen):
            chosen.append(e)
    return chosen[:limit]


class MockLLM:
    name = "mock"
    model = "mock-rule-v1"

    def generate(self, task: str, payload: dict[str, Any]) -> LLMResult:
        handler = getattr(self, f"_task_{task}", None)
        if handler is None:
            return LLMResult(data={}, model=self.model)
        return LLMResult(data=handler(payload), model=self.model)

    # ---------------- 路由 -------------------
    def _task_route(self, payload: dict[str, Any]) -> dict[str, Any]:
        hint = (payload.get("intent_hint") or "").strip()
        if hint in VALID_INTENTS:
            return {"intent": hint, "confidence": 1.0, "reason": "用户指定任务类型"}
        facts = payload.get("facts") or ""
        contract = payload.get("contract") or ""
        dispute_words = ["违约", "逾期", "索赔", "争议", "损失", "解除通知", "催告", "抗辩", "起诉", "仲裁",
                         "要求返还", "拒付", "拖欠", "未支付", "不支付", "欠款", "口头约定", "主张", "协商未果"]
        consult_words = ["请问", "如何理解", "是否合法", "咨询", "依据哪一条"]
        hits = sum(1 for w in dispute_words if w in facts)
        if hits >= 2:
            return {"intent": "dispute_simulation", "confidence": min(0.95, 0.6 + hits * 0.08),
                    "reason": f"事实陈述命中 {hits} 个争议要素"}
        if any(w in facts for w in consult_words) and not contract:
            return {"intent": "statute_consult", "confidence": 0.8, "reason": "咨询型表述且无合同文本"}
        if contract:
            return {"intent": "contract_review", "confidence": 0.75, "reason": "提供合同文本且争议要素不足"}
        return {"intent": "statute_consult", "confidence": 0.6, "reason": "信息不足，降级为条文咨询"}

    # ---------------- 主张 -------------------
    def _task_plaintiff(self, payload: dict[str, Any]) -> dict[str, Any]:
        evidence = payload.get("evidence", [])
        focuses = payload.get("focuses", [])
        rnd = int(payload.get("round", 1))
        used = set(payload.get("used_evidence", []))
        args: list[dict[str, Any]] = []
        for f in focuses:
            key = f["key"]
            picked = _pick(evidence, key, exclude=used if rnd == 1 else None, prefer="plaintiff")
            new_ids = [i for i in (_sid(e) for e in picked) if i not in used]
            basis = [_sid(e) for e in picked]
            body = PLAINTIFF_TEXTS.get(key, "我方主张依约履行，对方应承担相应的违约责任。")
            if picked:
                lead = f"依{picked[0].get('locator')}（{picked[0].get('title')}），"
                prefix = "" if rnd == 1 else ("【第%d轮·新证据】" % rnd if new_ids else "【第%d轮·无新增证据】" % rnd)
                claim = f"{prefix}{lead}{body}"
                args.append({"focus_key": key, "claim": claim, "basis": basis, "conclusion": "support"})
            else:
                args.append({
                    "focus_key": key,
                    "claim": f"就「{topic_title(key)}」，我方暂未检索到有效依据，仅陈述事实，不作定性结论。",
                    "basis": [], "conclusion": "undetermined",
                })
        return {"arguments": args}

    # ---------------- 抗辩 -------------------
    def _task_defendant(self, payload: dict[str, Any]) -> dict[str, Any]:
        evidence = payload.get("evidence", [])
        p_args = payload.get("plaintiff_arguments", [])
        rnd = int(payload.get("round", 1))
        args: list[dict[str, Any]] = []
        for pa in p_args:
            key = pa.get("focus_key") or "违约责任"
            picked = _pick(evidence, key, exclude=set(pa.get("basis", [])), prefer="defendant")
            body = DEFENDANT_TEXTS.get(key, "对方的该项主张缺乏证据支撑，我方不予认可。")
            prefix = "" if rnd == 1 else f"【第{rnd}轮】"
            if picked:
                lead = f"依{picked[0].get('locator')}（{picked[0].get('title')}），"
                args.append({
                    "focus_key": key,
                    "claim": f"{prefix}针对对方主张「{_clip(pa.get('claim'), 40)}」，{lead}{body}",
                    "basis": [_sid(e) for e in picked],
                    "targets": [pa.get("argument_id", "")],
                    "conclusion": "rebut",
                })
            else:
                args.append({
                    "focus_key": key,
                    "claim": f"{prefix}就「{topic_title(key)}」，我方暂无有效反证，仅在举证责任层面提出抗辩：对方应就其主张的损失与因果关系提供证据。",
                    "basis": [],
                    "targets": [pa.get("argument_id", "")],
                    "conclusion": "undetermined",
                })
        return {"arguments": args}

    # ---------------- 裁决 -------------------
    def _task_judge(self, payload: dict[str, Any]) -> dict[str, Any]:
        focuses = payload.get("focuses", [])
        missing = payload.get("missing_clauses", [])
        out_focuses = []
        for f in focuses:
            pv = _clip(f.get("plaintiff_claim") or "（原告未就该焦点提出主张）", 80)
            dv = _clip(f.get("defendant_claim") or "（被告未就该焦点提出抗辩）", 80)
            if f.get("status") == "conflict":
                note = "双方就同一焦点提交了结论相反的同层级证据，需按效力层级与贴合度取舍"
            elif f.get("status") == "undetermined":
                note = "至少一方缺乏有效证据，本焦点不宜作定性判断"
            else:
                gap = abs(float(f.get("plaintiff_strength", 0)) - float(f.get("defendant_strength", 0)))
                note = f"双方证据层级可比，强度差 {gap:.2f}"
            out_focuses.append({
                "key": f["key"],
                "title": f.get("title") or topic_title(f["key"]),
                "plaintiff_view": pv,
                "defendant_view": dv,
                "note": note,
            })

        n_conflict = sum(1 for f in focuses if f.get("status") == "conflict")
        n_und = sum(1 for f in focuses if f.get("status") == "undetermined")
        n_res = len(focuses) - n_conflict - n_und

        summary = (
            f"本次推演共归纳 {len(focuses)} 项争议焦点：其中 {n_res} 项双方证据充分可比，"
            f"{n_conflict} 项存在同层级证据冲突，{n_und} 项至少一方证据不足。"
            f"合同文本另有 {len(missing)} 项必备条款缺失或未明确约定。"
        )
        remediation: list[str] = []
        if n_conflict:
            remediation.append("就存在冲突的焦点补充更高层级依据（法律条文/司法解释），或补充能直接证明损失金额的财务凭证。")
        if n_und:
            remediation.append("对证据不足的焦点补充合同原件、往来函件、验收记录等书证，避免仅以陈述主张。")
        if missing:
            remediation.append("补签补充协议，明确以下缺失条款：" + "、".join(missing[:4]) + "。")
        if not remediation:
            remediation.append("现有证据链基本完整，建议按意见书倾向准备谈判方案，并保留诉讼方案备选。")

        p_total = sum(float(f.get("plaintiff_strength", 0)) for f in focuses) or 0.0
        d_total = sum(float(f.get("defendant_strength", 0)) for f in focuses) or 0.0
        if n_und and n_und >= len(focuses) / 2:
            party = "undetermined"
            reason = "多数焦点证据不足，无法形成可靠倾向"
        elif n_conflict:
            party = "undetermined"
            reason = "存在同层级证据冲突且未被更高层级依据解决"
        elif p_total - d_total > 0.15:
            party = "plaintiff"
            reason = f"原告方证据强度合计 {p_total:.2f}，高于被告方 {d_total:.2f}"
        elif d_total - p_total > 0.15:
            party = "defendant"
            reason = f"被告方证据强度合计 {d_total:.2f}，高于原告方 {p_total:.2f}"
        else:
            party = "undetermined"
            reason = f"双方证据强度接近（{p_total:.2f} vs {d_total:.2f}），不足以形成倾向"

        return {
            "focuses": out_focuses,
            "inclination_reason": reason,
            "summary": summary,
            "remediation": remediation,
            "party_suggestion": party,
        }

    # ---------------- 条款风险 -------------------
    def _task_clause_risk(self, payload: dict[str, Any]) -> dict[str, Any]:
        risks = payload.get("rule_risks", [])
        out = []
        for r in risks:
            out.append({
                "clause_no": r.get("clause_no", ""),
                "risk": f"{r.get('name')}：{_clip(r.get('snippet'), 60)}",
                "level": r.get("level", "中"),
                "basis": r.get("basis", []),
                "suggestion": r.get("advice", ""),
            })
        return {"risks": out}

    # ---------------- 条文咨询 -------------------
    def _task_consult(self, payload: dict[str, Any]) -> dict[str, Any]:
        evidence = payload.get("evidence", [])
        if not evidence:
            return {"answer": "现有语料不足以支撑结论，建议补充合同文本或具体争议事实。"}
        parts = []
        for e in evidence[:4]:
            parts.append(f"{e.get('locator')}：{_clip(e.get('text') or e.get('title'), 100)}")
        return {"answer": "依据检索到的以下规定：" + "；".join(parts) + "。具体适用仍需结合合同条款与事实证据判断。"}
