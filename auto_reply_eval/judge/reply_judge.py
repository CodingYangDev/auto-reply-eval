"""裁判：把单条回复送进模型，得到通过校验的结构化评分。"""

from __future__ import annotations

import json
import re
from typing import Sequence

from auto_reply_eval.judge.prompts import SYSTEM_PROMPT, build_user_prompt
from auto_reply_eval.llm.base import ChatMessage, LLMClient
from auto_reply_eval.metrics.definitions import all_definitions
from auto_reply_eval.models import (
    CaseJudgement,
    JudgeOutput,
    MetricDefinition,
    ReplySource,
)

_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class JudgeResponseError(RuntimeError):
    """模型输出无法解析成合法评分时抛出。"""


class ReplyJudge:
    """按评分卡对单条回复打分。"""

    def __init__(
        self,
        client: LLMClient,
        definitions: Sequence[MetricDefinition] | None = None,
        max_retries: int = 2,
    ) -> None:
        self._client = client
        self._definitions = tuple(definitions or all_definitions())
        self._max_retries = max_retries
        self._expected_metrics = {definition.name for definition in self._definitions}

    @property
    def client_name(self) -> str:
        return self._client.name

    def judge(
        self,
        case_id: str,
        user_question: str,
        text: str,
        source: ReplySource,
    ) -> CaseJudgement:
        """对一条回复打分；解析失败时自动重试。"""

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=build_user_prompt(
                    self._definitions,
                    case_id=case_id,
                    user_question=user_question,
                    reply_text=text,
                    source=source,
                ),
            ),
        ]

        last_error: Exception | None = None
        for _ in range(self._max_retries + 1):
            raw = self._client.complete(messages)
            try:
                output = self._parse(raw)
            except JudgeResponseError as error:
                last_error = error
                continue
            return CaseJudgement(
                case_id=case_id,
                user_question=user_question,
                source=source,
                text=text,
                output=output,
            )

        raise JudgeResponseError(f"{case_id} 裁判输出解析失败：{last_error}")

    def _parse(self, raw: str) -> JudgeOutput:
        match = _JSON_BLOCK_PATTERN.search(raw)
        if not match:
            raise JudgeResponseError("输出中未找到 JSON 对象")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise JudgeResponseError(f"JSON 解析失败：{error}") from error

        try:
            output = JudgeOutput.model_validate(payload)
        except Exception as error:  # noqa: BLE001 - 统一转换成可重试的解析异常
            raise JudgeResponseError(f"评分结构校验失败：{error}") from error

        produced = {verdict.metric for verdict in output.verdicts}
        missing = self._expected_metrics - produced
        if missing:
            names = "、".join(sorted(metric.value for metric in missing))
            raise JudgeResponseError(f"评分缺少指标：{names}")
        return output


def summarize_output(output: JudgeOutput) -> str:
    """把评分压成一行，便于日志与调试。"""

    parts = [
        f"{verdict.metric.value}={verdict.score}" for verdict in output.verdicts
    ]
    return ", ".join(parts)


__all__ = ["JudgeResponseError", "ReplyJudge", "summarize_output"]
