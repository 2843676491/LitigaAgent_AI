#!/usr/bin/env python3
"""生成架构图 SVG（脚本化布局，保证文字精确、可复现）。

用法：python scripts/make_architecture_svg.py
产物：docs/assets/architecture.svg
"""
from __future__ import annotations

import html
from pathlib import Path

W, H = 1260, 1240
BG = "#0f1720"
BAND_LABEL = "#8fa3b8"
TITLE_C = "#f2f5f8"

# 颜色：墨蓝底 + 铜金/青灰强调，避免大面积紫蓝渐变
LAYERS = [
    {
        "name": "接入层",
        "en": "ACCESS",
        "color": "#3f5a72",
        "items": [
            ("Web 前端", ["推演台 / 庭审时间线", "意见书 / 人工复核 / 审计链"]),
            ("REST + SSE API", ["FastAPI 0.162", "流式推送各节点进度"]),
            ("RBAC / ABAC", ["角色→权限矩阵", "案件归属部门校验"]),
        ],
    },
    {
        "name": "编排层",
        "en": "ORCHESTRATION",
        "color": "#7a6134",
        "items": [
            ("Orchestrator", ["意图路由：推演/审查/咨询", "轮次与调用预算控制"]),
            ("LangGraph 1.2", ["StateGraph 有环状态图", "条件边 + interrupt 暂停"]),
            ("Checkpointer", ["SQLite 状态持久化", "thread_id ↔ hearing_id"]),
            ("收敛控制器", ["轮次上限 / 新增信息", "法官 converged 判据"]),
        ],
    },
    {
        "name": "Agent 层",
        "en": "AGENTS",
        "color": "#2f6157",
        "items": [
            ("检索 Agent", ["共享证据池", "只写 evidence_id"]),
            ("主张 Agent", ["原告：论点 + 法条", "对称 Prompt"]),
            ("抗辩 Agent", ["被告：逐条反驳", "弱点攻击 + 反证"]),
            ("法官 Agent", ["焦点归纳 + 强度比对", "风险分级 + 倾向意见"]),
            ("引用校验", ["回查索引定位", "编造引用即剔除"]),
            ("HITL 闸门", ["低置信/冲突/高风险", "必须人工签署"]),
        ],
    },
    {
        "name": "能力层",
        "en": "CAPABILITY",
        "color": "#4a4f6b",
        "items": [
            ("混合检索器", ["Milvus 向量 + BM25", "RRF(k=60) + 规则重排"]),
            ("LLM 适配层", ["Mock（离线可复现）", "OpenAI 兼容 / 方舟"]),
            ("Embedder", ["哈希 384 维（默认）", "BGE-M3（可选）"]),
            ("Schema 校验", ["Pydantic 2.14", "不合法→重试1次→降级"]),
        ],
    },
    {
        "name": "存储层",
        "en": "STORAGE",
        "color": "#5a4b3c",
        "items": [
            ("SQLite", ["案件/庭审/意见书", "审计哈希链"]),
            ("Milvus Lite", ["本地向量索引", "可切远程 Milvus"]),
            ("对象目录", ["原件 + MD5 去重", "可切 MinIO"]),
            ("语料库", ["法条 / 类案 / 评测集", "JSONL 版本化"]),
        ],
    },
]

TOP = 150
BAND_H = 152
GAP = 26
LEFT_LABEL_W = 158
PAD = 26


def esc(s: str) -> str:
    return html.escape(s)


def build() -> str:
    out: list[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Noto Sans SC, PingFang SC, Microsoft YaHei, sans-serif">'
    )
    out.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    # 网格底纹
    out.append('<defs><pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">'
               '<path d="M32 0H0V32" fill="none" stroke="#1b2836" stroke-width="1"/></pattern></defs>')
    out.append(f'<rect width="{W}" height="{H}" fill="url(#grid)" opacity="0.9"/>')

    # 标题
    out.append(f'<text x="{PAD}" y="62" fill="{TITLE_C}" font-size="30" font-weight="700">'
               'LegalCourt AI · 合同争议模拟庭审多 Agent 推演系统</text>')
    out.append(f'<text x="{PAD}" y="96" fill="{BAND_LABEL}" font-size="15">'
               '分层架构 ｜ 有环状态图 + 证据溯源 + 人工复核闸门 + 全链路哈希审计链</text>')
    out.append(f'<text x="{W - PAD}" y="62" fill="#8c7a52" font-size="14" text-anchor="end">'
               'v1.0 · 2026-11-07</text>')

    legend_y = TOP + len(LAYERS) * (BAND_H + GAP) + 6

    for li, layer in enumerate(LAYERS):
        y = TOP + li * (BAND_H + GAP)
        # 左侧层标签
        out.append(f'<rect x="{PAD}" y="{y}" width="{LEFT_LABEL_W - 16}" height="{BAND_H}" rx="10" '
                   f'fill="#16222f" stroke="{layer["color"]}" stroke-width="1.2"/>')
        out.append(f'<text x="{PAD + (LEFT_LABEL_W - 16) / 2}" y="{y + BAND_H / 2 - 6}" fill="#e8eef4" '
                   f'font-size="18" font-weight="600" text-anchor="middle">{layer["name"]}</text>')
        out.append(f'<text x="{PAD + (LEFT_LABEL_W - 16) / 2}" y="{y + BAND_H / 2 + 18}" fill="{BAND_LABEL}" '
                   f'font-size="10" letter-spacing="1.5" text-anchor="middle">{layer["en"]}</text>')

        items = layer["items"]
        n = len(items)
        avail = W - PAD * 2 - LEFT_LABEL_W
        bw = (avail - (n - 1) * 14) / n
        for i, (title, subs) in enumerate(items):
            x = PAD + LEFT_LABEL_W + i * (bw + 14)
            out.append(f'<rect x="{x:.1f}" y="{y + 10}" width="{bw:.1f}" height="{BAND_H - 20}" rx="9" '
                       f'fill="#1a2735" stroke="{layer["color"]}" stroke-width="1.4"/>')
            out.append(f'<rect x="{x:.1f}" y="{y + 10}" width="4" height="{BAND_H - 20}" rx="2" fill="{layer["color"]}"/>')
            out.append(f'<text x="{x + bw / 2:.1f}" y="{y + 46}" fill="{TITLE_C}" font-size="14.5" '
                       f'font-weight="600" text-anchor="middle">{esc(title)}</text>')
            for k, s in enumerate(subs):
                out.append(f'<text x="{x + bw / 2:.1f}" y="{y + 72 + k * 20}" fill="{BAND_LABEL}" '
                           f'font-size="11.5" text-anchor="middle">{esc(s)}</text>')

        # 层间下行箭头（含数据流编号）
        if li < len(LAYERS) - 1:
            ay = y + BAND_H
            cx = PAD + LEFT_LABEL_W + avail / 2
            out.append(f'<path d="M{cx:.1f} {ay}V{ay + GAP - 4}" stroke="#5d7a92" stroke-width="2" '
                       f'marker-end="url(#arrow)"/>')
            out.append(f'<circle cx="{cx - 18:.1f}" cy="{ay + GAP / 2}" r="9" fill="#22303f" '
                       f'stroke="#5d7a92" stroke-width="1"/>')
            out.append(f'<text x="{cx - 18:.1f}" y="{ay + GAP / 2 + 4}" fill="#b8c9d8" font-size="10.5" '
                       f'text-anchor="middle">{li + 1}</text>')

    # 箭头定义
    out.append('<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
               '<path d="M0 0L8 4L0 8Z" fill="#5d7a92"/></marker></defs>')

    # 图例：主链路
    out.append(f'<rect x="{PAD}" y="{legend_y}" width="{W - PAD * 2}" height="132" rx="10" '
               f'fill="#131e29" stroke="#2b3b4b"/>')
    out.append(f'<text x="{PAD + 20}" y="{legend_y + 30}" fill="{TITLE_C}" font-size="14" font-weight="600">'
               '主链路（一次推演的闭环）</text>')
    flow = [
        ("①", "合同/事实上传", "MD5 去重 + 原件留存"),
        ("②", "条款级切片", "保留 clause_no / offset"),
        ("③", "意图路由", "推演 / 审查 / 咨询"),
        ("④", "混合检索 + RRF", "证据分级 + 出处定位"),
        ("⑤", "主张 ↔ 抗辩", "有环辩论，轮次受控"),
        ("⑥", "法官裁决", "焦点 + 风险 + 倾向"),
        ("⑦", "引用校验", "编造即剔除、降置信"),
        ("⑧", "人工复核", "interrupt 真暂停"),
        ("⑨", "归档 + 审计", "哈希链 + 评测回流"),
    ]
    col_w = (W - PAD * 2 - 40) / 5
    for i, (num, name, desc) in enumerate(flow):
        row, col = divmod(i, 5)
        x = PAD + 20 + col * col_w
        y = legend_y + 62 + row * 52
        out.append(f'<circle cx="{x + 11}" cy="{y - 4}" r="11" fill="#2b3b4b"/>')
        out.append(f'<text x="{x + 11}" y="{y}" fill="#d8b77a" font-size="11" text-anchor="middle">{num}</text>')
        out.append(f'<text x="{x + 30}" y="{y - 6}" fill="#e8eef4" font-size="12" font-weight="600">{esc(name)}</text>')
        out.append(f'<text x="{x + 30}" y="{y + 12}" fill="{BAND_LABEL}" font-size="10.5">{esc(desc)}</text>')

    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    target = Path(__file__).resolve().parent.parent / "docs" / "assets" / "architecture.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build(), encoding="utf-8")
    print(f"written: {target}")


if __name__ == "__main__":
    main()
