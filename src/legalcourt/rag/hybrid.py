"""混合检索：向量 + BM25 → RRF 融合 → 规则重排。

为什么是 RRF 而不是加权求和：cosine 与 BM25 分数量纲不同，加权需要归一化，
而归一化会扭曲"某一路完全失效"的情形；RRF 只用排名，鲁棒且无需调参。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from functools import lru_cache
from typing import Any

from ..config import settings
from ..llm.embeddings import tokenize
from ..schemas import Evidence, SourceType
from .store import EvidenceIndex

_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零〇两]+条")


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion。输入多个"按相关性降序的 id 列表"。"""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, eid in enumerate(lst, start=1):
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank)
    return scores


def _overlap(query_tokens: set[str], rec: dict[str, Any]) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = set(tokenize(f"{rec['title']} {rec['text']}"))
    if not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def _article_hit(query: str, locator: str) -> int:
    arts = _ARTICLE_RE.findall(query or "")
    if not arts:
        return 0
    return 1 if any(a in (locator or "") for a in arts) else 0


def rerank(query: str, rec: dict[str, Any], rrf_score: float, max_rrf: float) -> float:
    """规则重排：融合分 0.55 + 词面重合 0.30 + 效力层级 0.10 + 条文号命中 0.05。"""
    base = (rrf_score / max_rrf) if max_rrf > 0 else 0.0
    tier = int(rec.get("tier", 5))
    authority = (5 - tier) / 4.0
    return round(
        0.55 * base
        + 0.30 * _overlap(set(tokenize(query)), rec)
        + 0.10 * authority
        + 0.05 * _article_hit(query, rec.get("locator", "")),
        4,
    )


class HybridRetriever:
    def __init__(self, index: EvidenceIndex | None = None) -> None:
        self.index = index or get_index()
        self._cache: OrderedDict[tuple, list[Evidence]] = OrderedDict()
        self._cache_size = settings.search_cache_size

    # ---------- 缓存 ----------
    def _cache_get(self, key: tuple) -> list[Evidence] | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: tuple, value: list[Evidence]) -> None:
        self._cache[key] = value
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        self._cache.clear()

    # ---------- 检索 ----------
    def search(
        self,
        query: str,
        top_k: int | None = None,
        case_id: str | None = None,
        source_types: set[SourceType] | None = None,
    ) -> list[Evidence]:
        top_k = top_k or settings.top_k_final
        key = (query.strip(), top_k, case_id, tuple(sorted(st.name for st in source_types)) if source_types else None)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        vhits = self.index.vector_search(query, settings.top_k_vector)
        bhits = self.index.bm25_search(query, settings.top_k_bm25)
        fused = rrf_fuse([[e for e, _ in vhits], [e for e, _ in bhits]], settings.rrf_k)
        if not fused:
            return []

        max_rrf = max(fused.values())
        candidates: list[tuple[float, dict[str, Any]]] = []
        for eid, s in fused.items():
            rec = self.index.get(eid)
            if not rec:
                continue
            # 文档级隔离：只允许本案件条款 + 全局语料
            if rec.get("case_id") not in (None, case_id):
                continue
            if source_types and SourceType(rec["source_type"]) not in source_types:
                continue
            candidates.append((rerank(query, rec, s, max_rrf), rec))

        candidates.sort(key=lambda x: -x[0])
        out = [
            Evidence(
                evidence_id=rec["evidence_id"],
                source_type=SourceType(rec["source_type"]),
                title=rec["title"],
                locator=rec["locator"],
                snippet=(rec["text"][:260] if rec["text"] else ""),
                score=score,
                tier=int(rec["tier"]),
                tags=rec.get("tags", []),
                meta={k: v for k, v in rec.items() if k in {"favor", "court", "year", "code"}},
            )
            for score, rec in candidates[:top_k]
        ]
        self._cache_put(key, out)
        return out

    def search_multi(self, queries: list[str], top_k: int = 6, case_id: str | None = None,
                     max_total: int = 14, case_recall: int = 2,
                     statute_recall: int = 3) -> list[Evidence]:
        """多 query（每个争议焦点一个）取并集，去重后按最高分排序。

        case_recall：每个焦点额外做一次"仅类案"的定向召回。
        没有这一步，纯相关性排序会把"结论相反的类案"挤出候选池，
        导致冲突检测与质证环节形同虚设。
        """
        pool: dict[str, Evidence] = {}
        for q in queries:
            if not q.strip():
                continue
            for ev in self.search(q, top_k=top_k, case_id=case_id):
                if ev.evidence_id not in pool or pool[ev.evidence_id].score < ev.score:
                    pool[ev.evidence_id] = ev
            if statute_recall:
                for ev in self.search(q, top_k=statute_recall, case_id=case_id,
                                      source_types={SourceType.STATUTE, SourceType.JUDICIAL_INTERPRETATION}):
                    if ev.evidence_id not in pool or pool[ev.evidence_id].score < ev.score:
                        pool[ev.evidence_id] = ev
            if case_recall:
                for ev in self.search(q, top_k=case_recall, case_id=case_id, source_types={SourceType.CASE}):
                    if ev.evidence_id not in pool or pool[ev.evidence_id].score < ev.score:
                        pool[ev.evidence_id] = ev
        return sorted(pool.values(), key=lambda e: -e.score)[:max_total]


# ---------------- 单例 ----------------

@lru_cache(maxsize=1)
def get_index() -> EvidenceIndex:
    return EvidenceIndex()


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever(get_index())


def reset_retriever() -> None:
    get_retriever.cache_clear()
    get_index.cache_clear()
