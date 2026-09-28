"""离线裁判模拟器。

它不访问任何网络服务，而是从提示词里提取结构化输入，
再用 :mod:`auto_reply_eval.llm.heuristics` 中的规则生成一份
格式与真实裁判完全一致的结构化输出。

这样做的目的：让流水线、报告、校验、测试在无 API Key 的环境下也能端到端跑通，
并且结果是确定的、可复现的。生产环境应切换到 :class:`OpenAICompatibleClient`。
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from auto_reply_eval.llm.base import ChatMessage
from auto_reply_eval.llm.heuristics import HeuristicScorer

_INPUT_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


class MockLLMClient:
    """确定性的离线裁判模拟器。"""

    def __init__(self) -> None:
        self._scorer = HeuristicScorer()

    @property
    def name(self) -> str:
        return "mock-heuristic-judge (offline)"

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        payload = self._extract_payload(messages)
        result: dict[str, Any] = self._scorer.score(
            user_question=str(payload["user_question"]),
            reply_text=str(payload["reply_text"]),
        )
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _extract_payload(messages: Sequence[ChatMessage]) -> dict[str, Any]:
        for message in reversed(messages):
            if message.role != "user":
                continue
            match = _INPUT_BLOCK_PATTERN.search(message.content)
            if match:
                return json.loads(match.group(1))
        raise ValueError("提示词中未找到结构化输入块，无法模拟裁判打分")
