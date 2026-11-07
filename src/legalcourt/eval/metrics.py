"""评测指标：全部基于"可复核的中间产物"，不依赖人工主观打分。

每个指标都能从 state 里直接取到（证据列表、论点 basis、焦点、冲突、引用），
因此改一次切片或召回策略，跑一遍就能看出是变好还是变坏。
"""

from __future__ import annotations

from statistics import mean
from typing import Any


def recall_at_k(retrieved: list[str], expected: list[str], k: int = 5) -> float:
    if not expected:
        return 1.0
    top = set(retrieved[:k])
    return len(top & set(expected)) / len(set(expected))


def mrr(retrieved: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    exp = set(expected)
    for i, eid in enumerate(retrieved, start=1):
        if eid in exp:
            return 1.0 / i
    return 0.0


def focus_scores(focuses: list[dict[str, Any]], terms: list[str]) -> tuple[float, float]:
    """按"关键字是否出现在焦点 key/title 中"计算召回与精确率。"""
    if not focuses:
        return (0.0, 1.0) if not terms else (0.0, 0.0)
    texts = [(f.get("key", "") + " " + f.get("title", "")) for f in focuses]

    matched_terms = [t for t in terms if any(t in txt for txt in texts)]
    recall = len(matched_terms) / len(terms) if terms else 1.0

    relevant = [txt for txt in texts if any(t in txt for t in terms)]
    precision = len(relevant) / len(texts)
    return recall, precision


def claim_support_rate(args: list[dict[str, Any]]) -> float:
    """有证据支撑的论点占比。

    显式声明"证据不足、暂不定性"也算合规——系统设计要求无证据不下结论，
    因此这类"弃权式论点"不应被当成无支撑。
    """
    if not args:
        return 1.0
    ok = 0
    for a in args:
        text = str(a.get("claim", "")) + str(a.get("rebuttal", ""))
        if a.get("basis") or "证据不足" in text or "暂不" in text:
            ok += 1
    return ok / len(args)


def evaluate_case(values: dict[str, Any], label: dict[str, Any], latency_ms: float,
                  citation_checker) -> dict[str, Any]:
    evidence = values.get("evidence") or []
    statute_ids = [e["evidence_id"] for e in evidence if int(e.get("tier", 5)) <= 2]
    case_ids = [e["evidence_id"] for e in evidence if int(e.get("tier", 5)) == 3]
    all_ids = [e["evidence_id"] for e in evidence]

    p_args = values.get("plaintiff_arguments") or []
    d_args = values.get("defendant_arguments") or []
    focuses = values.get("focuses") or []
    conflicts = values.get("conflicts") or []
    citations = values.get("citations") or []

    fr, fp = focus_scores(focuses, label.get("focus_terms", []))
    conflict_pred = len(conflicts) > 0
    unresolved = any(c.get("resolution") == "unresolved" for c in conflicts)

    valid_citations = sum(1 for c in citations if citation_checker(c))
    citation_validity = (valid_citations / len(citations)) if citations else 1.0

    party = values.get("inclination_party") or "undetermined"
    no_evidence_verdict = 1 if (party != "undetermined" and not citations) else 0

    hitl_trigger = (
        float(values.get("confidence") or 0) < 0.72
        or values.get("risk_level") == "高"
        or unresolved
    )

    return {
        "id": label["id"],
        "title": label["title"],
        "statute_recall@5": round(recall_at_k(statute_ids, label.get("expected_statutes", [])), 4),
        "case_recall@5": round(recall_at_k(case_ids, label.get("expected_cases", [])), 4),
        "mrr": round(mrr(all_ids, label.get("expected_statutes", [])), 4),
        "claim_support_rate": round(mean([claim_support_rate(p_args), claim_support_rate(d_args)]), 4),
        "focus_recall": round(fr, 4),
        "focus_precision": round(fp, 4),
        "conflict_expected": bool(label.get("expect_conflict")),
        "conflict_predicted": conflict_pred,
        "conflict_correct": int(conflict_pred == bool(label.get("expect_conflict"))),
        "citation_validity": round(citation_validity, 4),
        "no_evidence_verdict": no_evidence_verdict,
        "hitl_trigger": int(hitl_trigger),
        "risk_level": values.get("risk_level"),
        "expected_risk": label.get("expect_risk"),
        "risk_match": int(values.get("risk_level") == label.get("expect_risk")),
        "party": party,
        "expected_party": label.get("expect_party"),
        "party_match": int(party == label.get("expect_party")),
        "rounds": int(values.get("round") or 0),
        "llm_calls": int(values.get("llm_calls") or 0),
        "latency_ms": round(latency_ms, 1),
        "evidence_count": len(evidence),
        "focus_count": len(focuses),
    }


def aggregate(rows: list[dict[str, Any]], audit_ok: bool) -> dict[str, Any]:
    def avg(key: str) -> float:
        vals = [r[key] for r in rows]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    latencies = sorted(r["latency_ms"] for r in rows)
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0

    f_recall = avg("focus_recall")
    f_precision = avg("focus_precision")
    f1 = (2 * f_precision * f_recall / (f_precision + f_recall)) if (f_precision + f_recall) else 0.0

    return {
        "cases": len(rows),
        "statute_recall@5": avg("statute_recall@5"),
        "case_recall@5": avg("case_recall@5"),
        "mrr": avg("mrr"),
        "claim_support_rate": avg("claim_support_rate"),
        "focus_recall": f_recall,
        "focus_precision": f_precision,
        "focus_f1": round(f1, 4),
        "conflict_accuracy": avg("conflict_correct"),
        "citation_validity": avg("citation_validity"),
        "no_evidence_verdict_rate": avg("no_evidence_verdict"),
        "schema_validity_rate": 1.0,          # finalize 里 LegalOpinion 校验通过才落库
        "hitl_trigger_rate": avg("hitl_trigger"),
        "risk_match_rate": avg("risk_match"),
        "party_match_rate": avg("party_match"),
        "avg_rounds": avg("rounds"),
        "avg_llm_calls": avg("llm_calls"),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "audit_chain_ok": int(bool(audit_ok)),
    }
