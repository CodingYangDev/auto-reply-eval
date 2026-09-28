"""裁判提示词构造。

提示词与代码分离，便于单独迭代评分口径而不改动流程逻辑。
"""

from __future__ import annotations

import json
from typing import Sequence

from auto_reply_eval.metrics.definitions import SCORE_MAX, SCORE_MIN, SCORE_STEP
from auto_reply_eval.models import MetricDefinition, ReplySource

SYSTEM_PROMPT = f"""你是一位资深的电商客服质量评估专家，负责评估"客服自动回复"的质量。

评分要求：
1. 只依据给定的用户问题与待评估回复，不要脑补外部信息。
2. 对每个指标独立打分，分数范围 {SCORE_MIN}-{SCORE_MAX}，以 {SCORE_STEP} 为最小粒度。
3. 严格对齐评分锚点，不要给人情分；也不要因为语气客气就抬高「有用性」。
4. 如果回复把操作成本推回给用户（让用户自己看详情页、自己联系其他客服），
   即使内容正确，「有用性」也应低于 3 分。
5. 每个指标的判据必须指向回复中的具体表述，不要写空泛的评价。
6. 「不瞎编」的判断尺度必须统一，避免把常规政策误判成编造：
   - 平台通用政策与法定时限（退款到账时长、七天无理由、三包质保期限等）属于常规表述，
     即使本次没有资料逐条核对，也只做轻微扣分，不要直接判为编造；
   - 针对用户具体商品的参数断言（材质、容量、成分、型号）若无商品库依据，
     属于高风险断言，需要明确指出「无法核实」；
   - 平台未必兑现的承诺（额外补偿、保证、免费赠送）才按编造处理，从重扣分。

输出要求：只输出一个 JSON 对象，不要输出任何解释性文字或 Markdown 代码块标记。
"""

_OUTPUT_SCHEMA = """{
  "verdicts": [
    {"metric": "faithfulness", "score": 5, "reason": "一句话说明判据"},
    {"metric": "accuracy", "score": 5, "reason": "一句话说明判据"},
    {"metric": "helpfulness", "score": 3, "reason": "一句话说明判据"},
    {"metric": "tone", "score": 5, "reason": "一句话说明判据"}
  ],
  "unsupported_claims": ["回复中无法核实或过度承诺的具体断言，没有则为空数组"],
  "missing_actions": ["为了真正解决问题、回复本应做到却没有做到的动作，没有则为空数组"]
}"""


def build_rubric_text(definitions: Sequence[MetricDefinition]) -> str:
    """把指标定义渲染成给模型看的评分卡。"""

    blocks: list[str] = []
    for definition in definitions:
        anchors = "\n".join(
            f"      {score} 分：{text}" for score, text in sorted(definition.anchors.items())
        )
        blocks.append(
            f"  - {definition.label}（metric={definition.name.value}，权重 {definition.weight}）\n"
            f"      定义：{definition.definition}\n"
            f"      量化方式：{definition.quantification}\n"
            f"      锚点：\n{anchors}"
        )
    return "\n".join(blocks)


def build_user_prompt(
    definitions: Sequence[MetricDefinition],
    case_id: str,
    user_question: str,
    reply_text: str,
    source: ReplySource,
) -> str:
    """构造单条 case 的评判请求。"""

    payload = {
        "case_id": case_id,
        "user_question": user_question,
        "reply_text": reply_text,
        "source": source.value,
    }
    source_hint = (
        "这条回复是线上自动回复，请从严评估其「有用性」。"
        if source is ReplySource.AUTO
        else "这条回复是人工客服的参考回复，用于建立评分基准。"
    )
    return f"""【评估任务】
请对下面这条客服回复，在评分卡的每个指标上打分，并给出判据。
{source_hint}

【评分卡】
{build_rubric_text(definitions)}

【输出格式】仅输出 JSON：
{_OUTPUT_SCHEMA}

【输入数据】
```json
{json.dumps(payload, ensure_ascii=False, indent=2)}
```"""
