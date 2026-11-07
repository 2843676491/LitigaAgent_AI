"""OpenAI 兼容协议客户端（火山方舟 / DeepSeek / vLLM 通用）。"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .base import LLMResult
from .prompts import build_messages

_JSON_RE = re.compile(r"\{[\s\S]*\}")


class OpenAICompatLLM:
    name = "openai-compat"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0) -> None:
        if not base_url:
            raise ValueError("LLM_BASE_URL 未配置，无法使用 openai 模式")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def generate(self, task: str, payload: dict[str, Any]) -> LLMResult:
        messages = build_messages(task, payload)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1200,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self._endpoint(), headers=headers, json={**body, "response_format": {"type": "json_object"}})
            if resp.status_code >= 400:
                # 部分端点不支持 response_format，降级重试一次
                resp = client.post(self._endpoint(), headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()

        content = raw["choices"][0]["message"]["content"]
        data = self._parse(content)
        usage = raw.get("usage") or {}
        return LLMResult(
            data=data,
            calls=1,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            model=self.model,
        )

    @staticmethod
    def _parse(content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            m = _JSON_RE.search(content or "")
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        raise ValueError(f"LLM 输出无法解析为 JSON：{content[:200]}")
