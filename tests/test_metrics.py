"""指标定义与裁判输出的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_reply_eval.judge.reply_judge import JudgeResponseError, ReplyJudge
from auto_reply_eval.llm.base import ChatMessage
from auto_reply_eval.llm.mock_client import MockLLMClient
from auto_reply_eval.metrics.definitions import all_definitions
from auto_reply_eval.models import MetricName, ReplySource


def test_weights_sum_to_one_and_priorities_are_unique() -> None:
    definitions = all_definitions()
    assert sum(item.weight for item in definitions) == pytest.approx(1.0)
    assert sorted(item.priority for item in definitions) == [1, 2, 3, 4]
    assert {item.name for item in definitions} == set(MetricName)


def test_priority_order_matches_policy() -> None:
    ordered = [item.name for item in all_definitions()]
    assert ordered == [
        MetricName.FAITHFULNESS,
        MetricName.ACCURACY,
        MetricName.HELPFULNESS,
        MetricName.TONE,
    ]


def test_mock_judge_penalises_self_service_reply() -> None:
    """把操作推回用户的回复，其有用性应明显低于主动代办的回复。"""

    judge = ReplyJudge(client=MockLLMClient())
    question = "我的快递放错快递柜了取不出来"

    self_service = judge.judge(
        case_id="t1",
        user_question=question,
        text="您好，请您在订单详情页查看取件码，或联系客服协助处理。",
        source=ReplySource.AUTO,
    )
    proactive = judge.judge(
        case_id="t2",
        user_question=question,
        text="我帮您联系快递员重新派送，请您提供订单号，我帮您处理。",
        source=ReplySource.AUTO,
    )

    assert self_service.score(MetricName.HELPFULNESS) < proactive.score(
        MetricName.HELPFULNESS
    )


def test_judge_retries_then_raises_on_broken_output() -> None:
    class BrokenClient:
        @property
        def name(self) -> str:
            return "broken"

        def complete(self, messages: list[ChatMessage]) -> str:
            return "这不是 JSON"

    judge = ReplyJudge(client=BrokenClient(), max_retries=1)
    with pytest.raises(JudgeResponseError):
        judge.judge("t3", "问题", "回复", ReplySource.AUTO)


def test_judge_rejects_incomplete_metric_set() -> None:
    class PartialClient:
        @property
        def name(self) -> str:
            return "partial"

        def complete(self, messages: list[ChatMessage]) -> str:
            payload = {"verdicts": [{"metric": "tone", "score": 5, "reason": "x"}]}
            return json.dumps(payload, ensure_ascii=False)

    judge = ReplyJudge(client=PartialClient(), max_retries=0)
    with pytest.raises(JudgeResponseError):
        judge.judge("t4", "问题", "回复", ReplySource.AUTO)


def test_mock_client_requires_structured_input_block() -> None:
    with pytest.raises(ValueError):
        MockLLMClient().complete([ChatMessage(role="user", content="没有输入块")])


def test_repository_rejects_missing_file(tmp_path: Path) -> None:
    from auto_reply_eval.data.repository import DatasetRepository

    repository = DatasetRepository(tmp_path / "nope.json", tmp_path / "nope2.json")
    with pytest.raises(FileNotFoundError):
        repository.load_cases()
