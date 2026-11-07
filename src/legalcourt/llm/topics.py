"""领域词表：争议焦点主题、必备条款清单、条款风险规则。

这些是"法律领域知识"，与 LLM 无关——无论 Mock 还是真实模型，都用同一份词表，
保证可评测、可回归。
"""

from __future__ import annotations

import re
from typing import Any

# key -> {title, keywords, tags}
TOPICS: dict[str, dict[str, Any]] = {
    "违约金": {
        "title": "违约金标准是否过分高于实际损失",
        "keywords": ["违约金", "过分高于", "调减", "酌减", "30%", "百分之三十"],
        "tags": ["违约金", "违约金调整", "违约金过高", "违约责任"],
    },
    "逾期交付": {
        "title": "逾期交付的责任归属与免责事由",
        "keywords": ["逾期", "迟延", "延期", "逾期交付", "逾期交货", "催告"],
        "tags": ["违约", "逾期", "交付", "催告", "迟延履行"],
    },
    "质量瑕疵": {
        "title": "标的物质量是否符合约定及能否补救",
        "keywords": ["质量", "瑕疵", "缺陷", "不合格", "良品率", "质量不符", "阻断性"],
        "tags": ["质量瑕疵", "质量标准", "质量", "验收", "解除"],
    },
    "合同解除": {
        "title": "是否满足法定或约定的解除条件",
        "keywords": ["解除", "终止合同", "退还", "返还", "撤销合同"],
        "tags": ["合同解除", "法定解除", "约定解除", "催告"],
    },
    "付款义务": {
        "title": "价款支付义务与逾期付款责任",
        "keywords": ["付款", "价款", "到货款", "尾款", "预付款", "欠款", "拒付", "未付款", "欠付", "服务费"],
        "tags": ["付款义务", "价款", "逾期付款", "分期付款"],
    },
    "格式条款": {
        "title": "格式条款是否订入合同或是否无效",
        "keywords": ["格式条款", "免责条款", "未提示", "未说明", "加粗", "单独提示"],
        "tags": ["格式条款", "提示说明义务", "免责条款", "条款无效"],
    },
    "损失赔偿": {
        "title": "损失范围、可预见性与因果关系",
        "keywords": ["损失", "赔偿", "停产", "停摆", "差价", "可得利益"],
        "tags": ["损失赔偿", "可得利益", "可预见规则"],
    },
    "减损义务": {
        "title": "守约方是否履行减损义务及过错分担",
        "keywords": ["减损", "扩大损失", "未采取", "与有过失", "双方违约"],
        "tags": ["减损义务", "与有过失", "双方违约", "损失赔偿"],
    },
    "不可抗力": {
        "title": "不可抗力能否免责及通知证明义务",
        "keywords": ["不可抗力", "疫情", "管控", "自然灾害", "封控"],
        "tags": ["不可抗力", "免责", "通知义务"],
    },
    "情势变更": {
        "title": "是否构成情势变更而非商业风险",
        "keywords": ["情势变更", "涨价", "成本上涨", "价格波动", "市场变化"],
        "tags": ["情势变更", "商业风险", "合同变更"],
    },
    "分期付款": {
        "title": "分期付款解除条件是否成就",
        "keywords": ["分期", "分期付款", "五分之一"],
        "tags": ["分期付款", "合同解除", "付款义务"],
    },
    "验收": {
        "title": "验收程序与视为验收的法律后果",
        "keywords": ["验收", "签收", "视为合格", "书面异议"],
        "tags": ["验收", "签收", "质量瑕疵"],
    },
    "建设工程": {
        "title": "工程款支付与优先受偿权",
        "keywords": ["工程款", "建设工程", "承包", "发包", "优先受偿", "竣工"],
        "tags": ["建设工程", "优先受偿", "工程款"],
    },
    "举证责任": {
        "title": "举证责任分配与证据充分性",
        "keywords": ["举证", "证据不足", "凭证", "无法证明", "未签订书面", "口头约定", "仅有微信", "未提供书面"],
        "tags": ["举证责任", "证据"],
    },
}

MAX_FOCUS = 5


def detect_focus_candidates(text: str, limit: int = MAX_FOCUS) -> list[dict[str, Any]]:
    """从事实陈述中识别争议焦点候选，按命中强度排序。"""
    scored: list[tuple[float, str]] = []
    for key, meta in TOPICS.items():
        hits = sum(text.count(kw) for kw in meta["keywords"])
        if hits:
            scored.append((hits, key))
    scored.sort(key=lambda x: (-x[0], x[1]))
    keys = [k for _, k in scored][:limit]
    if not keys:
        keys = ["违约责任" if "违约责任" in TOPICS else "违约金"]
    return [{"key": k, "title": TOPICS[k]["title"], "hits": dict(scored).get(k, 0)} for k in keys]


def topic_title(key: str) -> str:
    return TOPICS.get(key, {}).get("title", key)


def topic_tags(key: str) -> list[str]:
    return list(TOPICS.get(key, {}).get("tags", []))


def evidence_matches_topic(tags: list[str], title: str, key: str) -> bool:
    """证据是否属于某焦点：标签交集优先，其次标题/条文关键词命中。"""
    ttags = topic_tags(key)
    if set(tags) & set(ttags):
        return True
    for kw in TOPICS.get(key, {}).get("keywords", []):
        if kw and kw in (title or ""):
            return True
    return False


# ---------------- 必备条款清单（用于 missing_clauses） ----------------

REQUIRED_CLAUSES: dict[str, list[str]] = {
    "当事人信息": ["甲方", "乙方", "买方", "卖方", "委托方", "开发方"],
    "标的与数量": ["标的", "数量", "型号", "清单"],
    "质量标准": ["质量标准", "技术协议", "良品率", "国家标准"],
    "价款与支付": ["价款", "支付", "付款", "总金额"],
    "履行期限": ["期限", "交付", "交货", "完成", "上线"],
    "验收": ["验收", "签收", "试运行"],
    "违约责任": ["违约", "违约金", "赔偿"],
    "合同解除": ["解除", "终止"],
    "不可抗力": ["不可抗力"],
    "保密": ["保密", "商业秘密"],
    "争议解决": ["争议解决", "仲裁", "诉讼", "管辖"],
    "知识产权": ["知识产权", "著作权", "专利"],
}


def missing_clauses(contract_text: str) -> list[str]:
    text = contract_text or ""
    return [name for name, kws in REQUIRED_CLAUSES.items() if not any(k in text for k in kws)]


# ---------------- 条款风险规则（contract_review 路由） ----------------

RISK_RULES: list[dict[str, Any]] = [
    {
        "id": "R1",
        "name": "违约金比例偏高",
        "patterns": [r"30\s*%", r"百分之三十", r"每日\s*1\s*%", r"日千分之一", r"日万分之五"],
        "level": "中",
        "statutes": ["S-028", "J-001", "J-003"],
        "advice": "将违约金比例调整至可举证的实际损失区间（通常不超过损失的 130%），并约定损失计算方式。",
    },
    {
        "id": "R2",
        "name": "单方解除权/单方变更权",
        "patterns": [r"有权单方(解除|变更)", r"无需(通知|征得)对方同意"],
        "level": "高",
        "statutes": ["S-022", "S-023"],
        "advice": "解除权应对等设置，明确解除事由、通知期与善后义务。",
    },
    {
        "id": "R3",
        "name": "笼统免责条款",
        "patterns": [r"不承担(任何)?(赔偿)?责任", r"自行承担(全部)?(风险|损失)"],
        "level": "高",
        "statutes": ["S-009", "S-010", "J-002"],
        "advice": "删除概括免责表述，改为列明具体免责情形，并对重大利害条款加粗提示、单独确认。",
    },
    {
        "id": "R4",
        "name": "争议解决由对方所在地专属管辖",
        "patterns": [r"(乙方|卖方|开发方)所在地(人民法院|仲裁)", r"仲裁委员会仲裁"],
        "level": "中",
        "statutes": ["S-007"],
        "advice": "争取约定我方所在地法院管辖，或改为中立地仲裁机构。",
    },
    {
        "id": "R5",
        "name": "缺少验收/异议期条款",
        "patterns": [r"验收"],
        "level": "中",
        "statutes": ["S-007", "S-034"],
        "advice": "补充验收标准、异议期与逾期未验收的推定规则。",
    },
    {
        "id": "R6",
        "name": "保密义务无期限或范围过宽",
        "patterns": [r"永久保密", r"无限期保密"],
        "level": "低",
        "statutes": ["S-012"],
        "advice": "明确保密信息范围、期限（一般 2-5 年）与例外情形。",
    },
]


def scan_clause_risks(clauses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """规则扫描 + 保留 clause_no，供前端定位原文。"""
    risks: list[dict[str, Any]] = []
    for rule in RISK_RULES:
        for c in clauses:
            text = c.get("text", "")
            if any(re.search(p, text) for p in rule["patterns"]):
                # R5 是"缺失型"规则，由 missing_clauses 单独处理，避免误报
                if rule["id"] == "R5":
                    continue
                risks.append(
                    {
                        "rule_id": rule["id"],
                        "name": rule["name"],
                        "clause_no": c.get("clause_no", ""),
                        "locator": c.get("locator", ""),
                        "snippet": text[:120],
                        "level": rule["level"],
                        "basis": rule["statutes"],
                        "advice": rule["advice"],
                    }
                )
                break  # 每个规则只报一次，避免同类刷屏
    return risks
