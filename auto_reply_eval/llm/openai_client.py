"""基于 OpenAI 兼容协议的裁判模型客户端。"""

from __future__ import annotations

from typing import Any, Sequence

from openai import OpenAI

from auto_reply_eval.config import Settings
from auto_reply_eval.llm.base import ChatMessage


class OpenAICompatibleClient:
    """适配所有兼容 OpenAI Chat Completions 协议的服务。

    OpenAI / DeepSeek / 通义千问 / 智谱 等服务都可以通过 ``base_url`` 接入，
    因此不需要为每家厂商维护一套客户端。
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.api_key:
            raise ValueError(
                "未配置 EVAL_API_KEY，无法使用真实 LLM 模式；"
                "请复制 .env.example 为 .env 并填写，或改用 --mode mock"
            )
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
        )

    @property
    def name(self) -> str:
        return f"{self._settings.model} @ {self._settings.base_url}"

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [message.to_payload() for message in messages],
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
        }
        if self._settings.json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**payload)
        return response.choices[0].message.content or ""
