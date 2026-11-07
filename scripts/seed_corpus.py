"""语料体检与装载：校验 id 唯一、字段完整，可选触发一次索引重建。

用法：
  python scripts/seed_corpus.py            # 只体检
  python scripts/seed_corpus.py --reindex  # 体检 + 重建向量/BM25 索引
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FILES = {
    "statutes.jsonl": ("法条", {"id", "code", "article_no", "title", "text", "tags"}),
    "interpretations.jsonl": ("司法解释", {"id", "title", "text", "tags"}),
    "cases.jsonl": ("类案", {"id", "title", "text", "tags", "favor"}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reindex", action="store_true")
    args = ap.parse_args()

    corpus = ROOT / "data" / "corpus"
    total = 0
    ok = True
    for name, (label, required) in FILES.items():
        path = corpus / name
        if not path.exists():
            print(f"[缺失] {name}")
            ok = False
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        total += len(rows)
        ids = [r.get("id") for r in rows]
        dup = [k for k, v in Counter(ids).items() if v > 1]
        missing = [r.get("id") for r in rows if required - set(r.keys())]
        print(f"[{label}] {len(rows)} 条 · id 重复 {len(dup)} · 缺字段 {len(missing)}")
        if dup or missing:
            ok = False
            print("  重复：", dup[:5], "缺字段：", missing[:5])
        if name == "cases.jsonl":
            favor = Counter(r.get("favor") for r in rows)
            print(f"  类案立场分布：{dict(favor)}（冲突检测依赖 favor，三种立场都应有样本）")

    print(f"合计 {total} 条")
    if not ok:
        sys.exit(1)

    if args.reindex:
        from legalcourt.rag.hybrid import get_retriever

        retriever = get_retriever()
        retriever.clear_cache()
        print(f"索引重建完成：{retriever.index.size} 条证据")


if __name__ == "__main__":
    main()
