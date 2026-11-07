"""文档治理：MD5 去重、条款级切片、语料装载。

切片的唯一要求：**切片必须能被精确引用**。
因此每条切片都带 clause_no 与 locator，前端可以回到原文的"第 7 条"。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import settings
from ..schemas import AUTHORITY_TIER, SourceType

# 第X条 / 第X条之一
CLAUSE_RE = re.compile(r"^[\s　]*第[一二三四五六七八九十百零〇两]+条(?:之[一二三四五六七八九十])?")
NUM_RE = re.compile(r"^[\s　]*([一二三四五六七八九十]+)、")
ARABIC_RE = re.compile(r"^[\s　]*(\d{1,2})[\.．、]\s*")

TAG_RULES: list[tuple[str, list[str]]] = [
    ("违约金", ["违约金", "违约责任"]),
    ("违约", ["违约责任"]),
    ("解除", ["合同解除"]),
    ("验收", ["验收", "质量瑕疵"]),
    ("质量", ["质量瑕疵", "质量标准"]),
    ("不可抗力", ["不可抗力"]),
    ("保密", ["保密"]),
    ("争议解决", ["争议解决", "管辖"]),
    ("仲裁", ["争议解决", "管辖"]),
    ("诉讼", ["争议解决", "管辖"]),
    ("付款", ["付款义务", "价款"]),
    ("价款", ["付款义务", "价款"]),
    ("交付", ["交付"]),
    ("知识产权", ["知识产权"]),
    ("免责", ["格式条款", "免责条款"]),
    ("工程款", ["建设工程", "工程款"]),
    ("情势变更", ["情势变更"]),
    ("分期", ["分期付款"]),
]


@dataclass
class Chunk:
    chunk_id: str
    case_id: str | None
    clause_no: str
    text: str
    locator: str
    tags: list[str] = field(default_factory=list)
    source_type: str = "contract"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "case_id": self.case_id,
            "clause_no": self.clause_no,
            "text": self.text,
            "locator": self.locator,
            "tags": self.tags,
            "source_type": self.source_type,
        }


def md5_of(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.md5(data).hexdigest()


def _tags_of(text: str, clause_no: str) -> list[str]:
    tags: list[str] = []
    for kw, ts in TAG_RULES:
        if kw in text or kw in clause_no:
            tags.extend(ts)
    return sorted(set(tags))


def split_contract(text: str, case_id: str, filename: str | None = None) -> list[Chunk]:
    """按"条"切片；无条款结构时回退为段落窗口切片。"""
    text = (text or "").strip()
    if not text:
        return []

    raw_lines = text.splitlines()
    blocks: list[tuple[str, list[str]]] = []
    current_no = "前言"
    current: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):   # 标题行不作为条款头
            continue
        if CLAUSE_RE.match(stripped):
            if current:
                blocks.append((current_no, current))
            current_no = stripped.split("　")[0][:12]
            current = [stripped]
            continue
        if NUM_RE.match(stripped) and len(stripped) <= 20:
            if current:
                blocks.append((current_no, current))
            current_no = stripped[:12]
            current = [stripped]
            continue
        current.append(stripped)
    if current:
        blocks.append((current_no, current))

    chunks: list[Chunk] = []
    short = (case_id or "case")[-6:]
    if len(blocks) < 3:
        # 回退：按段落窗口切片（保证有可引用单元，但 locator 精度下降）
        paras = [p for p in re.split(r"\n{2,}", text) if p.strip()]
        for i, p in enumerate(paras, 1):
            chunks.append(
                Chunk(
                    chunk_id=f"CT-{short}-P{i:02d}",
                    case_id=case_id,
                    clause_no=f"段落{i}",
                    text=p.strip()[:900],
                    locator=f"{filename or '合同'}#段落{i}",
                    tags=_tags_of(p, ""),
                )
            )
        return chunks

    step = 3  # 条款合并粒度：3 条一个 chunk，既保上下文又不过长
    for i in range(0, len(blocks), step):
        group = blocks[i : i + step]
        clause_no = group[0][0]
        body = "\n".join("\n".join(lines) for _, lines in group)
        if len(body) > 900:
            body = body[:900]
        chunks.append(
            Chunk(
                chunk_id=f"CT-{short}-C{i // step + 1:02d}",
                case_id=case_id,
                clause_no=clause_no,
                text=body,
                locator=f"{filename or '合同'}#{clause_no}",
                tags=_tags_of(body, clause_no),
            )
        )
    return chunks


def load_corpus_records(corpus_dir: Path | None = None) -> list[dict[str, Any]]:
    """读取 data/corpus/*.jsonl，统一成证据记录。"""
    corpus_dir = corpus_dir or settings.corpus_dir
    out: list[dict[str, Any]] = []
    if not corpus_dir.exists():
        return out
    for path in sorted(corpus_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{lineno} JSON 解析失败：{exc}") from exc
                out.append(normalize_record(rec))
    return out


def normalize_record(rec: dict[str, Any]) -> dict[str, Any]:
    st = SourceType(rec["source_type"])
    if st == SourceType.STATUTE:
        locator = f"{rec.get('code', '法条')}#{rec.get('article_no', '')}"
    elif st == SourceType.JUDICIAL_INTERPRETATION:
        locator = f"司法解释#{rec.get('article_no', '')}"
    else:
        locator = rec.get("case_no", rec.get("id", ""))
    return {
        "evidence_id": rec["id"],
        "source_type": st.value,
        "title": rec.get("title", ""),
        "text": rec.get("text", ""),
        "locator": locator,
        "tags": rec.get("tags", []),
        "tier": AUTHORITY_TIER[st],
        "favor": rec.get("favor"),
        "case_id": None,
        "meta": rec.get("meta") or {},
        "code": rec.get("code"),
        "court": rec.get("court"),
        "year": rec.get("year"),
    }


def chunk_to_record(chunk: Chunk | dict[str, Any]) -> dict[str, Any]:
    c = chunk.to_dict() if isinstance(chunk, Chunk) else chunk
    return {
        "evidence_id": c["chunk_id"],
        "source_type": SourceType.CONTRACT.value,
        "title": f"{c['locator']}",
        "text": c["text"],
        "locator": c["locator"],
        "tags": c.get("tags", []),
        "tier": AUTHORITY_TIER[SourceType.CONTRACT],
        "favor": None,
        "case_id": c.get("case_id"),
        "meta": {"clause_no": c.get("clause_no")},
    }


def fact_record(case_id: str, facts: str) -> dict[str, Any]:
    return {
        "evidence_id": f"FACT-{case_id[-6:]}",
        "source_type": SourceType.FACTS.value,
        "title": "争议事实陈述",
    "text": facts[:1200],
        "locator": "事实陈述",
        "tags": [],
        "tier": AUTHORITY_TIER[SourceType.FACTS],
        "favor": None,
        "case_id": case_id,
        "meta": {},
    }
