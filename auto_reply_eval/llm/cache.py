"""裁判响应缓存。

LLM 裁判的输出本身带有随机性，同一批数据重复运行会得到略有差异的报告，
这既不利于复核，也让"报告与 README 对不上"。

缓存装饰器把「模型 + 提示词」的哈希映射到原始响应：
* 同一次评估重跑时直接命中缓存，报告可以逐字复现；
* 只有提示词或模型变化时才会真正发起新的调用，避免重复烧钱。

它实现了与 :class:`LLMClient` 完全一致的最小接口，因此可以套在任意客户端外面。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from auto_reply_eval.llm.base import ChatMessage, LLMClient


class CachedLLMClient:
    """给任意裁判客户端加上磁盘缓存。"""

    def __init__(self, inner: LLMClient, cache_path: Path) -> None:
        self._inner = inner
        self._cache_path = cache_path
        self._cache: dict[str, str] = self._load()

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def hits(self) -> int:
        """本次进程内命中缓存的次数，用于在 CLI 中提示。"""

        return self._hits

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        key = self._key(messages)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            return cached

        response = self._inner.complete(messages)
        self._cache[key] = response
        self._flush()
        return response

    def _key(self, messages: Sequence[ChatMessage]) -> str:
        payload = "|".join(
            [self._inner.name]
            + [f"{message.role}:{message.content}" for message in messages]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _load(self) -> dict[str, str]:
        self._hits = 0
        if not self._cache_path.exists():
            return {}
        try:
            with self._cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): str(value) for key, value in payload.items()}

    def _flush(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("w", encoding="utf-8") as handle:
            json.dump(self._cache, handle, ensure_ascii=False, indent=2)
