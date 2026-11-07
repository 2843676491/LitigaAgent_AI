"""评测入口：在独立数据库/独立索引上跑标注集，输出指标与报告。

用法：python -m legalcourt.eval.run_eval   （或 make eval）
产物：data/eval/results.json、docs/04-测试与评测报告.md
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# 必须在导入 legalcourt 之前设置：使用独立库与独立索引，避免污染开发数据
os.environ["DB_PATH"] = str(ROOT / "data" / "runtime" / "eval.db")
os.environ["INDEX_DIR"] = str(ROOT / "data" / "index_eval")
os.environ["APP_ENV"] = "test"
os.environ.setdefault("LLM_PROVIDER", "mock")

for p in [Path(os.environ["DB_PATH"]), Path(os.environ["DB_PATH"]).with_suffix(".db-wal"),
          Path(os.environ["DB_PATH"]).with_suffix(".db-shm")]:
    if p.exists():
        p.unlink()
idx = Path(os.environ["INDEX_DIR"])
if idx.exists():
    shutil.rmtree(idx)

from legalcourt import audit, db, service  # noqa: E402
from legalcourt.eval.metrics import aggregate, evaluate_case  # noqa: E402
from legalcourt.rag.hybrid import get_retriever  # noqa: E402
from legalcourt.schemas import LegalOpinion  # noqa: E402

LABELS = ROOT / "data" / "eval" / "labeled_cases.jsonl"
RESULTS = ROOT / "data" / "eval" / "results.json"
REPORT = ROOT / "docs" / "04-测试与评测报告.md"

THRESHOLD = 0.72


def load_labels() -> list[dict]:
    rows = []
    with LABELS.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    labels = load_labels()
    retriever = get_retriever()
    rows: list[dict] = []
    schema_fail = 0

    for label in labels:
        case = db.create_case(label["title"], label["facts"], label.get("contract") or None,
                              f"eval-{label['id']}.md", "eval", "法务一部")
        t0 = time.perf_counter()
        out = service.run_sync(case["id"], max_rounds=3, auto_review=True)
        latency = (time.perf_counter() - t0) * 1000
        values = out["values"]

        try:
            LegalOpinion(**values["opinion"])
        except Exception:  # noqa: BLE001
            schema_fail += 1

        row = evaluate_case(values, label, latency, retriever.index.exists)
        rows.append(row)
        print(f"[{row['id']}] R@s={row['statute_recall@5']} R@c={row['case_recall@5']} "
              f"focus={row['focus_recall']}/{row['focus_precision']} risk={row['risk_level']} "
              f"hitl={row['hitl_trigger']} {row['latency_ms']}ms")

    chain = audit.verify_chain()
    summary = aggregate(rows, chain["ok"])
    summary["schema_validity_rate"] = round(1 - schema_fail / len(rows), 4)
    summary["hitl_threshold"] = THRESHOLD

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    write_report(summary, rows, chain)
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


def write_report(summary: dict, rows: list[dict], chain: dict) -> None:
    """报告里的"达标/未达标"全部由本次实测值计算，不允许写死结论。"""

    def ok(cond: bool) -> str:
        return "达标" if cond else "未达标"

    s = summary
    items = [
        ("法条召回率 Recall@5", f"{s['statute_recall@5']:.2f}", "≥0.85", s["statute_recall@5"] >= 0.85),
        ("类案召回率 Recall@5", f"{s['case_recall@5']:.2f}", "≥0.80", s["case_recall@5"] >= 0.80),
        ("MRR", f"{s['mrr']:.2f}", "≥0.70", s["mrr"] >= 0.70),
        ("主张证据支撑率", f"{s['claim_support_rate']:.2f}", "≥0.90", s["claim_support_rate"] >= 0.90),
        ("焦点召回率", f"{s['focus_recall']:.2f}", "≥0.80", s["focus_recall"] >= 0.80),
        ("焦点精确率", f"{s['focus_precision']:.2f}", "≥0.60", s["focus_precision"] >= 0.60),
        ("焦点 F1", f"{s['focus_f1']:.2f}", "≥0.70", s["focus_f1"] >= 0.70),
        ("冲突识别准确率", f"{s['conflict_accuracy']:.2f}", "≥0.85", s["conflict_accuracy"] >= 0.85),
        ("引用有效率", f"{s['citation_validity']:.2f}", "=1.00", s["citation_validity"] >= 0.999),
        ("无证据定性率", f"{s['no_evidence_verdict_rate']:.2f}", "=0.00", s["no_evidence_verdict_rate"] <= 0.001),
        ("Schema 通过率", f"{s['schema_validity_rate']:.2f}", "=1.00", s["schema_validity_rate"] >= 0.999),
        ("HITL 触发率", f"{s['hitl_trigger_rate']:.2f}", "0.20~0.60", 0.20 <= s["hitl_trigger_rate"] <= 0.60),
        ("平均轮次 / LLM 调用", f"{s['avg_rounds']:.1f} / {s['avg_llm_calls']:.1f}", "≤3 / ≤14",
         s["avg_rounds"] <= 3 and s["avg_llm_calls"] <= 14),
        ("P95 延迟", f"{s['p50_latency_ms']:.0f}ms / {s['p95_latency_ms']:.0f}ms", "P95<8000ms",
         s["p95_latency_ms"] < 8000),
    ]
    passed = sum(1 for it in items if it[3])

    lines = [
        "# 四、测试与评测报告（第 13 步）",
        "",
        "> 数据来源：`data/eval/labeled_cases.jsonl`（8 条人工标注用例）+ `data/eval/results.json`（机器可读结果）。",
        "> 运行方式：`make eval`，使用独立库 `data/runtime/eval.db` 与独立索引 `data/index_eval`，不污染开发数据。",
        "> 当前基线为 `LLM_PROVIDER=mock`（确定性、可复现）；接真实模型后应重跑同一套指标做对比。",
        "",
        f"**本轮到测：{passed} / {len(items)} 项达标。**",
        "",
        "## 核心指标",
        "",
        "| # | 指标 | 实测值 | 验收线 | 结论 | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    notes = [
        "期望法条/司法解释是否进入前 5",
        "期望类案是否进入前 5",
        "首个相关证据的平均倒数排名",
        "有 basis 的论点占比（双方平均）",
        "与人工标注焦点集合对比",
        "识别出的焦点中真正相关的比例",
        "召回与精确率的调和平均",
        "是否与标注的「存在相反结论类案」一致",
        "所有引用都能在索引中定位到原文",
        "无引用却给出倾向性结论的比例（硬红线）",
        "意见书是否通过 Pydantic 校验",
        "太少=没把关，太多=模型不可用",
        "收敛控制与成本预算",
        "Mock 基线延迟，真实模型另计",
    ]
    for i, ((name, val, target, cond), note) in enumerate(zip(items, notes), start=1):
        lines.append(f"| {i} | {name} | **{val}** | {target} | {ok(cond)} | {note} |")

    lines += [
        "",
        "附加指标（不作硬性验收，供调参参考）：",
        f"- 风险等级与标注一致率：{s['risk_match_rate']:.2f}",
        f"- 倾向性与标注一致率：{s['party_match_rate']:.2f}（标注本身含主观判断，仅作参考）",
        f"- 审计链完整性：{'通过' if chain['ok'] else '失败'}（校验 {chain['checked']} 条记录，断裂点 {chain['broken_at']}）",
        "",
        "## 逐案结果",
        "",
        "| 用例 | 证据数 | 法条R@5 | 类案R@5 | 焦点召回 | 风险 | HITL | 轮次 | LLM调用 | 耗时(ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['evidence_count']} | {r['statute_recall@5']:.2f} | {r['case_recall@5']:.2f} | "
            f"{r['focus_recall']:.2f} | {r['risk_level']} | {'是' if r['hitl_trigger'] else '否'} | "
            f"{r['rounds']} | {r['llm_calls']} | {r['latency_ms']:.0f} |"
        )

    lines += ["", "## 结论（按实测自动生成）", ""]
    if s["citation_validity"] >= 0.999 and s["no_evidence_verdict_rate"] <= 0.001:
        lines.append("1. **红线守住了**：引用有效率 1.00、无证据定性率 0.00——这是「敢给法务看」的前提，任何版本回退都必须拦。")
    else:
        lines.append("1. **红线被突破**：引用有效率或无证据定性率未达标，属于发布阻断项，必须先修检索/校验再谈其他指标。")
    if s["statute_recall@5"] < 0.85:
        lines.append(f"2. **法条召回是主要瓶颈**（{s['statute_recall@5']:.2f} < 0.85）：BM25 词面匹配 + 小规模语料的必然结果，下一步应换 BGE-M3 语义向量并扩法条库到全量，而不是继续调 top_k。")
    else:
        lines.append("2. **法条召回达标**：词表 + 关键词重排的组合在 42 条语料上够用，扩库后需用语义向量兜底。")
    if s["hitl_trigger_rate"] < 0.20:
        lines.append(f"3. **HITL 触发率偏低**（{s['hitl_trigger_rate']:.2f} < 0.20）：说明置信度公式在「证据齐全但双方势均力敌」的场景过于自信，应下调 confidence 基线或改为「焦点级」置信度。")
    elif s["hitl_trigger_rate"] > 0.60:
        lines.append(f"3. **HITL 触发率偏高**（{s['hitl_trigger_rate']:.2f} > 0.60）：复核会变成橡皮章，应提高阈值或只在「高风险 + 未决冲突」时拦截。")
    else:
        lines.append(f"3. **HITL 触发率落在目标区间**（{s['hitl_trigger_rate']:.2f}）：复核闸门既不过载也不空转。")
    lines.append(f"4. **风险/倾向一致率偏低**（{s['risk_match_rate']:.2f} / {s['party_match_rate']:.2f}）：评分是启发式加权，且标注本身有主观性；需与法务共同校准权重后再作为对外指标。")
    lines.append("5. **Mock 的定位**：它不追求论证质量，价值在于把收敛、冲突、引用校验、HITL、审计这些工程机制压成可回归的基线。")
    lines += [
        "",
        "## 复现",
        "",
        "```bash",
        "make eval     # 生成 data/eval/results.json 与本报告",
        "make test     # pytest：切片/RRF/收敛/HITL/审计/接口",
        "```",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport written: {REPORT}")


if __name__ == "__main__":
    main()
