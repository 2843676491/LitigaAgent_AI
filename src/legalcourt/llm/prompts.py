"""LLM Prompt 构造：Mock 与真实模型共用同一份任务契约。

每个 task 的 system 提示都明确要求"只输出 JSON"，并明确禁止无证据定性——
这与评测指标 no_evidence_verdict_rate 直接对应。
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPTS: dict[str, str] = {
    "route": (
        "你是法律事务编排器。根据用户提供的『争议事实』与『合同文本』判断任务类型，"
        "只输出 JSON：{\"intent\": \"dispute_simulation|contract_review|statute_consult\", "
        "\"confidence\": 0-1, \"reason\": \"简短理由\"}。"
        "出现已发生的争议、索赔、违约事实时取 dispute_simulation；仅有合同文本、要求审条款风险时取 contract_review；"
        "仅咨询规则、无具体合同争议时取 statute_consult。"
    ),
    "plaintiff": (
        "你是原告方（主张方）代理律师。只能使用给定 evidence 中出现的证据，禁止编造条文号或案号。"
        "针对每个争议焦点输出主张，只输出 JSON：{\"arguments\": [{\"focus_key\": str, \"claim\": str, \"basis\": [evidence_id]}]}。"
        "没有可用证据时，claim 必须写明『暂无有效依据，不作定性结论』，basis 为空数组。"
        "每一条 claim 都必须能被 basis 中的证据直接支撑。"
    ),
    "defendant": (
        "你是被告方（抗辩方）代理律师。只能使用给定 evidence，禁止编造条文号或案号。"
        "对每一条对方主张逐条反驳，优先攻击其证据层级、事实前提与因果关系。"
        "只输出 JSON：{\"arguments\": [{\"focus_key\": str, \"claim\": str, \"basis\": [evidence_id], "
        "\"targets\": [对方 argument_id], \"conclusion\": \"rebut|undetermined\"}]}。"
        "无有效反证时输出 conclusion=undetermined，并仅在举证责任层面抗辩。"
    ),
    "judge": (
        "你是中立仲裁法官。只做三件事：归纳争议焦点、写双方观点对照、给出裁决理由与补强建议。"
        "不得引入给定材料之外的任何事实或法条。只输出 JSON：{\"focuses\": [{\"key\": str, \"title\": str, "
        "\"plaintiff_view\": str, \"defendant_view\": str, \"note\": str}], \"inclination_reason\": str, "
        "\"summary\": str, \"remediation\": [str], \"party_suggestion\": \"plaintiff|defendant|undetermined\"}。"
        "证据不足或存在无法解决的同层级冲突时，party_suggestion 必须为 undetermined。"
    ),
    "clause_risk": (
        "你是合同审查律师。基于给定条款与规则命中结果，输出风险清单。"
        "只输出 JSON：{\"risks\": [{\"clause_no\": str, \"risk\": str, \"level\": \"高|中|低\", "
        "\"basis\": [evidence_id], \"suggestion\": str}]}。风险描述必须指向具体条款。"
    ),
    "consult": (
        "你是法律检索助手。仅依据给定 evidence 作答，答案末尾不得出现未提供来源的引用。"
        "只输出 JSON：{\"answer\": str}。证据不足时明确说明『现有语料不足以支撑结论』。"
    ),
}


def build_messages(task: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    system = SYSTEM_PROMPTS.get(task, "只输出 JSON。")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def schema_hint(task: str) -> str:
    return SYSTEM_PROMPTS.get(task, "")
