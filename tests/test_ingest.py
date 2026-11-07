from legalcourt.llm.topics import detect_focus_candidates, missing_clauses
from legalcourt.rag.ingest import md5_of, split_contract


CONTRACT = """第一条 标的与数量
全自动装配线一套，数量 2 台。

第二条 交付
卖方应于 2026 年 6 月 30 日前交付并完成调试。

第三条 违约责任
卖方逾期交付的，每逾期一日按合同总金额 1% 支付违约金。
"""


def test_split_contract_keeps_clause_structure():
    chunks = split_contract(CONTRACT, "case_abc123", "设备采购合同.md")
    assert len(chunks) >= 1
    assert all(c.chunk_id.startswith("CT-abc123-C") for c in chunks)
    # 3 条合一个 chunk，且 locator 能指回具体条款
    assert any("第一条" in c.locator for c in chunks)
    assert all(c.case_id == "case_abc123" for c in chunks)


def test_split_contract_short_text_falls_back_to_single_chunk():
    chunks = split_contract("只有一句话的合同摘要。", "case_x", None)
    assert len(chunks) == 1
    assert "#段落1" in chunks[0].locator


def test_split_contract_long_clause_is_truncated():
    nums = "一二三"
    clauses = "\n".join(f"第{nums[i]}条 约定\n" + ("内容" * 900) for i in range(3))
    chunks = split_contract(clauses, "case_long", "长合同.md")
    assert chunks
    assert all(len(c.text) <= 900 for c in chunks)


def test_md5_of_is_stable():
    assert md5_of("abc") == md5_of("abc")
    assert md5_of("abc") != md5_of("abd")


def test_detect_focus_from_facts():
    facts = "卖方逾期交付 45 天，我方主张违约金 90 万元；卖方称违约金过分高于损失，请求调减。"
    focuses = detect_focus_candidates(facts)
    keys = [f["key"] for f in focuses]
    assert "违约金" in keys
    assert len(focuses) <= 5


def test_missing_clauses_flags_absent_items():
    missing = missing_clauses(CONTRACT)
    assert "知识产权" in missing
    assert "不可抗力" in missing
    assert "违约责任" not in missing
