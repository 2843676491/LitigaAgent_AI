import json

from legalcourt import audit, db, service
from legalcourt.schemas import LegalOpinion


WEAK_FACTS = (
    "我方为对方提供企业管理咨询服务，双方口头约定服务费20万元。对方仅支付5万元后拒付，"
    "称咨询成果未达标，但未提供任何书面验收或异议记录。我方仅有微信聊天记录，未签订书面合同，多次催告无果。"
)


def test_weak_evidence_case_routes_to_dispute_and_hits_hitl():
    case = db.create_case("口头咨询服务费争议", WEAK_FACTS, None, None, "pytest", "法务一部")
    out = service.run_sync(case["id"], max_rounds=3, auto_review=True)
    v = out["values"]

    assert v["route"] == "dispute_simulation", "无书面合同也应走争议推演，而不是条文咨询"
    assert v["focuses"], "应识别出争议焦点"
    assert v["hitl"]["status"] == "approved" and v["hitl"]["triggers"], "弱证据必须命中复核条件"
    assert v["risk_level"] == "高"
    assert v["inclination_party"] == "undetermined"
    # 无书面合同时不得给出有倾向的定性结论
    assert v["opinion"]["undetermined"] is True
    assert v["opinion"]["missing_clauses"]


def test_opinion_schema_and_citation_integrity():
    case = db.create_case("设备采购逾期交付", "卖方逾期交付45天，我方主张违约金90万元；卖方称违约金过分高于损失请求调减。",
                          None, None, "pytest", "法务一部")
    out = service.run_sync(case["id"], max_rounds=3, auto_review=True)
    v = out["values"]

    opinion = LegalOpinion(**v["opinion"])
    assert 0 <= opinion.confidence <= 1
    assert 0 <= opinion.risk_score <= 1
    assert opinion.citations, "结论必须带引用"

    from legalcourt.rag.hybrid import get_retriever
    retriever = get_retriever()
    assert all(retriever.index.exists(cid) for cid in opinion.citations), "引用必须能在索引里定位"


def test_convergence_stops_within_round_limit():
    case = db.create_case("软件交付延迟争议", "开发方逾期20天交付，我方主张违约金8万元并解除合同；"
                          "开发方称延迟系需求变更所致，且缺陷已修复。", None, None, "pytest", "法务一部")
    out = service.run_sync(case["id"], max_rounds=3, auto_review=True)
    v = out["values"]
    assert v["round"] <= 3
    assert v["converged"] is True
    assert v["convergence_reason"]


def test_hearing_persisted_and_audit_chain_ok():
    case = db.create_case("审计链验证案件", "我方按时交付，对方拒付尾款30万元，经催告仍未支付。",
                          None, None, "pytest", "法务一部")
    out = service.run_sync(case["id"], max_rounds=2, auto_review=True)
    row = db.get_hearing(out["hearing_id"])
    assert row is not None
    assert json.loads(row["state_json"])["case_id"] == case["id"]

    verify = audit.verify_chain()
    assert verify["ok"] is True
