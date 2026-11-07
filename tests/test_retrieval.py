from legalcourt.rag.hybrid import get_retriever, rrf_fuse


def test_rrf_fuse_prefers_documents_hit_by_both_lists():
    fused = rrf_fuse([["a", "b", "c"], ["c", "a"]], k=60)
    # a 出现在两路且排名靠前；c 也出现在两路
    assert set(fused) == {"a", "b", "c"}
    assert fused["a"] > fused["b"]
    assert fused["c"] > fused["b"]


def test_search_returns_statute_for_penalty_query():
    hits = get_retriever().search("违约金过分高于实际损失 请求调减", top_k=5)
    ids = [h.evidence_id for h in hits]
    assert any(i in {"S-028", "J-001", "J-003"} for i in ids)
    # 每条证据都必须带可回溯的 locator 与原文
    assert all(h.locator and h.snippet for h in hits)


def test_search_isolates_other_cases_contract():
    hits = get_retriever().search("违约金 条款", top_k=8, case_id="case_none")
    assert all(h.source_type.value != "contract" or h.meta.get("case_id") in (None, "case_none") for h in hits)


def test_case_recall_brings_opposing_precedents():
    """定向类案召回：每个焦点额外取类案，否则冲突检测会因候选池缺件而失效。"""
    retriever = get_retriever()
    evs = retriever.search_multi(["违约金 过分高于 损失"], top_k=6, max_total=14)
    case_ids = [e.evidence_id for e in evs if e.tier == 3]
    assert case_ids, "类案定向召回没有返回任何类案"
