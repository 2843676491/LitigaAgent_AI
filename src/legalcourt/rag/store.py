"""证据索引：向量索引（Milvus Lite / 远程 Milvus）+ BM25 内存索引。

同一份 `records` 是"证据真源"：向量库只存 eid 与向量，正文永远从 records 取，
因此不需要在向量库里存长文本，也保证引用校验能拿到原文。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .. import db
from ..config import settings
from ..llm.embeddings import get_embedder, tokenize
from .ingest import load_corpus_records, chunk_to_record

COLLECTION = "lc_evidence"


def _pk(eid: str) -> int:
    return int.from_bytes(hashlib.blake2b(eid.encode("utf-8"), digest_size=8).digest()[:7], "little")


class _MilvusVector:
    kind = "milvus"

    def __init__(self, target: str, dim: int) -> None:
        from pymilvus import MilvusClient

        self.client = MilvusClient(target)
        self.dim = dim
        try:
            exists = self.client.has_collection(COLLECTION)
        except Exception:  # noqa: BLE001
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=COLLECTION, dimension=dim, metric_type="COSINE", auto_id=False,
            )
        # milvus-lite 进程重连后 collection 可能处于 released 状态，检索前必须 load
        self._loaded = False
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            self.client.load_collection(COLLECTION)
            self._loaded = True
        except Exception:  # noqa: BLE001
            self._loaded = False

    def upsert(self, items: list[tuple[str, list[float]]]) -> None:
        if not items:
            return
        records = [{"id": _pk(eid), "vector": vec, "eid": eid} for eid, vec in items]
        self.client.upsert(COLLECTION, records)

    def search(self, vec: list[float], k: int) -> list[tuple[str, float]]:
        self._ensure_loaded()
        res = self.client.search(COLLECTION, data=[vec], limit=k, output_fields=["eid"])
        hits = res[0] if res else []
        out = []
        for h in hits:
            eid = (h.get("entity") or {}).get("eid") or h.get("eid")
            if eid:
                out.append((str(eid), float(h.get("distance", 0.0))))
        return out


class _NumpyVector:
    """Milvus 不可用时的兜底实现（接口一致，保证系统不因向量库故障整体挂掉）。"""

    kind = "numpy"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vecs: dict[str, np.ndarray] = {}

    def upsert(self, items: list[tuple[str, list[float]]]) -> None:
        for eid, vec in items:
            self._vecs[eid] = np.asarray(vec, dtype=np.float32)

    def search(self, vec: list[float], k: int) -> list[tuple[str, float]]:
        if not self._vecs:
            return []
        q = np.asarray(vec, dtype=np.float32)
        eids = list(self._vecs.keys())
        mat = np.stack([self._vecs[e] for e in eids])
        scores = mat @ q
        order = np.argsort(-scores)[:k]
        return [(eids[i], float(scores[i])) for i in order]


class EvidenceIndex:
    def __init__(self, target: str | None = None, dim: int | None = None) -> None:
        self.dim = dim or settings.embed_dim
        self.records: dict[str, dict[str, Any]] = {}
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_dirty = True
        self._embedder = get_embedder()
        self.vector: Any = None
        self._init_vector(target or settings.milvus_target)
        self.load()

    # ---------- 初始化 ----------
    def _init_vector(self, target: str) -> None:
        try:
            self.vector = _MilvusVector(target, self.dim)
        except Exception:  # noqa: BLE001 - 向量库故障时降级，不阻断业务
            self.vector = _NumpyVector(self.dim)

    def load(self) -> None:
        for rec in load_corpus_records():
            self.records[rec["evidence_id"]] = rec
        for chunk in db.load_chunks():
            rec = chunk_to_record(chunk)
            self.records[rec["evidence_id"]] = rec
        self._sync_vectors(list(self.records.values()))
        self._bm25_dirty = True

    def _sync_vectors(self, recs: list[dict[str, Any]]) -> None:
        if not recs:
            return
        texts = [f"{r['title']} {r['text']}" for r in recs]
        vecs = self._embedder.encode_batch(texts)
        if vecs and len(vecs[0]) != self.dim:
            self.dim = len(vecs[0])
        self.vector.upsert([(r["evidence_id"], v) for r, v in zip(recs, vecs)])

    # ---------- 写入 ----------
    def add_chunks(self, chunk_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        recs = [chunk_to_record(c) for c in chunk_dicts]
        for r in recs:
            self.records[r["evidence_id"]] = r
        self._sync_vectors(recs)
        self._bm25_dirty = True
        return recs

    # ---------- 查询 ----------
    def get(self, eid: str) -> dict[str, Any] | None:
        return self.records.get(eid)

    def exists(self, eid: str) -> bool:
        return eid in self.records

    def _ensure_bm25(self) -> None:
        if not self._bm25_dirty and self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        eids = list(self.records.keys())
        corpus_tokens = [tokenize(f"{self.records[e]['title']} {self.records[e]['text']}") for e in eids]
        # rank_bm25 不接受空文档，给一个占位 token
        corpus_tokens = [t if t else ["_empty_"] for t in corpus_tokens]
        self._bm25 = BM25Okapi(corpus_tokens)
        self._bm25_ids = eids
        self._bm25_dirty = False

    def vector_search(self, query: str, k: int) -> list[tuple[str, float]]:
        vec = self._embedder.encode(query)
        return self.vector.search(vec, k)

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        tokens = tokenize(query) or ["_empty_"]
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(-np.asarray(scores))[:k]
        return [(self._bm25_ids[i], float(scores[i])) for i in order]

    @property
    def size(self) -> int:
        return len(self.records)
