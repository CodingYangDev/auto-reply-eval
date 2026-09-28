"""裁判模型的最小抽象。

评判层只依赖这个协议，因此可以在真实 LLM 与离线模拟器之间自由切换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence


@dataclass(frozen=True)
class ChatMessage:
    """一条对话消息。"""

    role: Literal["system", "user", "assistant"]
    content: str

    def to_payload(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMClient(Protocol):
    """给定对话消息、返回模型文本输出的最小接口。"""

    @property
    def name(self) -> str:
        """用于在报告中标识模型来源。"""
        ...

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        """执行一次补全，返回模型原始文本输出。"""
        ...
