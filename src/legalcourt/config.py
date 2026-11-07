"""全局配置：所有开关都走环境变量，业务代码不硬编码任何路径/模型。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]  # LegalCourtAI/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "LegalCourt AI"
    app_env: Literal["dev", "test", "prod"] = "dev"
    api_prefix: str = "/api/v1"

    # ---------- LLM ----------
    llm_provider: Literal["mock", "openai"] = "mock"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout: float = 30.0
    llm_max_calls_per_hearing: int = 14

    # ---------- 检索 ----------
    embedder: Literal["hash", "bge_m3"] = "hash"
    embed_dim: int = 384
    bge_m3_url: str = ""
    top_k_vector: int = 8
    top_k_bm25: int = 8
    top_k_final: int = 6
    rrf_k: int = 60
    search_cache_size: int = 128

    # ---------- 图与人工复核 ----------
    max_rounds: int = 3
    max_revisions: int = 2
    hitl_confidence_threshold: float = 0.72

    # ---------- 存储 ----------
    milvus_uri: str = ""
    storage_backend: Literal["local", "s3"] = "local"
    cache_backend: Literal["memory", "redis"] = "memory"
    audit_secret: str = "dev-only-change-me"

    # ---------- 路径 ----------
    db_path: Path = BASE_DIR / "data" / "runtime" / "legalcourt.db"
    index_dir: Path = BASE_DIR / "data" / "index"
    object_dir: Path = BASE_DIR / "data" / "runtime" / "objects"
    corpus_dir: Path = BASE_DIR / "data" / "corpus"
    sample_dir: Path = BASE_DIR / "data" / "samples"
    eval_dir: Path = BASE_DIR / "data" / "eval"
    web_dir: Path = BASE_DIR / "web"

    @property
    def milvus_target(self) -> str:
        """留空则用本地 milvus-lite 文件，生产填远程地址即可切换。"""
        return self.milvus_uri.strip() or str(self.index_dir / "milvus.db")

    def ensure_dirs(self) -> None:
        for p in (self.db_path.parent, self.index_dir, self.object_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
