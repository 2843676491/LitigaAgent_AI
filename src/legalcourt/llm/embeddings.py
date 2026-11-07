"""Embedding：默认用确定性哈希向量（离线、零依赖、可复现）；
生产可切 BGE-M3（通过 TEXT EMBEDDINGS INFERENCE 服务）。
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

import jieba
import numpy as np
import httpx

from ..config import settings

_CJK = re.compile(r"[\u4e00-\u9fff]")
_ASCII = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """统一分词：jieba 词 + 中文二元组 + 英文/数字串。检索与 BM25 共用。"""
    text = (text or "").strip()
    if not text:
        return []
    tokens: list[str] = []
    for w in jieba.lcut(text):
        w = w.strip()
        if not w:
            continue
        if _CJK.search(w) and len(w) >= 2:
            tokens.append(w)
        elif _ASCII.fullmatch(w):
            tokens.append(w.lower())
    # 中文二元组：兜住 jieba 词典未收录的法律术语
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        for i in range(len(run) - 1):
            tokens.append(run[i : i + 2])
    return tokens


class HashEmbedder:
    """特征哈希：token -> (维度索引, 符号)，累加后 L2 归一化。确定性、无需训练。"""

    name = "hash"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            # 词长加权：长词（更可能是专业术语）权重更高
            vec[idx] += sign * (1.0 + min(len(tok), 6) / 6.0)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]


class BgeM3Embedder:
    """TEI 风格接口：POST {url} {"inputs": [...]} -> [[float]]。失败自动降级为哈希。"""

    name = "bge_m3"

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self._fallback = HashEmbedder(settings.embed_dim)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = httpx.post(f"{self.url}/embed", json={"inputs": texts}, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data and isinstance(data[0], list):
                return [[float(x) for x in row] for row in data]
        except Exception:  # noqa: BLE001 - 降级优先，不阻断主流程
            pass
        return self._fallback.encode_batch(texts)

    def encode(self, text: str) -> list[float]:
        return self.encode_batch([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> HashEmbedder | BgeM3Embedder:
    if settings.embedder == "bge_m3" and settings.bge_m3_url:
        return BgeM3Embedder(settings.bge_m3_url)
    return HashEmbedder(settings.embed_dim)


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _l2(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]
