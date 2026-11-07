"""LLM 适配层：统一的 generate(task, payload) 接口。

- MockLLM：确定性规则生成（默认），可离线回归、可评测基线；
- OpenAICompatLLM：任意 OpenAI 兼容端点（火山方舟 / DeepSeek / vLLM）。
业务代码只依赖本接口，切换模型不改节点逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from ..config import settings


@dataclass
class LLMResult:
    data: dict[str, Any] = field(default_factory=dict)
    calls: int = 1
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = "mock"


class BaseLLM(Protocol):
    name: str
    model: str

    def generate(self, task: str, payload: dict[str, Any]) -> LLMResult: ...


@lru_cache(maxsize=1)
def get_llm() -> Any:
    if settings.llm_provider == "openai":
        from .openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model or "gpt-4o-mini",
            timeout=settings.llm_timeout,
        )
    from .mock import MockLLM

    return MockLLM()


def reset_llm() -> None:
    get_llm.cache_clear()
