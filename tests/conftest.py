"""pytest 全局配置：独立测试库与独立索引，避免污染开发数据。"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_TEST_DB = ROOT / "data" / "runtime" / "test.db"
_TEST_INDEX = ROOT / "data" / "index_test"

os.environ["DB_PATH"] = str(_TEST_DB)
os.environ["INDEX_DIR"] = str(_TEST_INDEX)
os.environ["APP_ENV"] = "test"
os.environ["LLM_PROVIDER"] = "mock"

for suffix in ("", "-wal", "-shm"):
    p = Path(str(_TEST_DB) + suffix)
    if p.exists():
        p.unlink()
if _TEST_INDEX.exists():
    shutil.rmtree(_TEST_INDEX)
