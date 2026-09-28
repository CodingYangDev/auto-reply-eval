"""用人工参考回复与人工评语来检验评估方法本身。

这里做三类校验：

1. **基准对照**：把人工参考回复用同一套评分卡打分，正常情况下它应当在"有用性"上
   稳定高于自动回复；如果连这一点都不成立，说明评分卡本身失效。
2. **尾部区分度**：人工评语中问题最明确的几条 case，应当落在整体分布尾部。
3. **分组区分度**：按人工评语正负倾向把 case 分成两组，问题组的"有用性"均值
   应当低于认可组；若区分不开，说明指标对人工关注点的还原度不足。
"""

from __future__ import annotations

from math import ceil
from typing import Sequence

from auto_reply_eval.models import (
    CaseJudgement,
    CaseScore,
    HumanReference,
    MetricName,
    ValidationFinding,
)

# 人工评语中负面判断最明确的 5 条，作为"已知最差集"用于检查尾部区分度。
# 逐条附上评语依据，便于人工复核该集合是否成立。
ANNOTATED_WORST_CASES: dict[str, str] = {
    "case_01": "评语：把责任推给了用户，没有体现主动服务意识",
    "case_08": "评语：典型的『正确但没用』",
    "case_13": "评语：没有体现对用户特殊需求的关注",
    "case_17": "评语：没有主动帮用户查，而是让用户自己去查",
    "case_20": "评语：属于答非所问",
}

# 评语关键词：用于把人工评语粗分成"有问题的"和"认可的"两组，做分组区分度检验。
PROBLEMATIC_MARKERS: tuple[str, ...] = (
    "推给",
    "推走",
    "没用",
    "负担",
    "答非所问",
    "没有帮",
    "没有查",
    "没有体现",
    "没有主动",
    "没有立刻帮",
    "不够",
    "让用户自己",
    "需要用户自己",
    "自己去看",
)

ACCEPTABLE_MARKERS: tuple[str, ...] = (
    "不错",
    "基本正确",
    "可接受",
    "尚可",
    "也有价值",
    "基本合理",
    "处理得不错",
    "是准确的",
)


def classify_annotator_note(note: str) -> str:
    """把人工评语粗分为 problematic / acceptable / mixed。"""

    problematic = sum(1 for marker in PROBLEMATIC_MARKERS if marker in note)
    acceptable = sum(1 for marker in ACCEPTABLE_MARKERS if marker in note)
    if problematic > acceptable:
        return "problematic"
    if acceptable > problematic:
        return "acceptable"
    return "mixed"


class HumanReferenceValidator:
    """基于人工参考回复的评估方法校验器。"""

    def __init__(self, worst_case_count: int = 3) -> None:
        self._worst_case_count = worst_case_count

    def validate(
        self,
        references: dict[str, HumanReference],
        auto_judgements: dict[str, CaseJudgement],
        human_judgements: dict[str, CaseJudgement],
        case_scores: Sequence[CaseScore],
        ranked_case_ids: Sequence[str],
        overall_score: float,
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        findings.append(
            self._check_reference_win_rate(auto_judgements, human_judgements)
        )
        findings.append(
            self._check_reference_margin(auto_judgements, human_judgements)
        )
        findings.append(
            self._check_worst_case_separation(case_scores, ranked_case_ids, overall_score)
        )
        findings.append(self._check_annotator_group_separation(references, auto_judgements))
        return findings

    # ------------------------------------------------------------------ 各项校验

    @staticmethod
    def _check_reference_win_rate(
        auto_judgements: dict[str, CaseJudgement],
        human_judgements: dict[str, CaseJudgement],
    ) -> ValidationFinding:
        shared = [
            case_id
            for case_id in auto_judgements
            if case_id in human_judgements
        ]
        wins = sum(
            1
            for case_id in shared
            if human_judgements[case_id].score(MetricName.HELPFULNESS)
            > auto_judgements[case_id].score(MetricName.HELPFULNESS)
        )
        rate = wins / len(shared) if shared else 0.0
        return ValidationFinding(
            name="参考回复胜出率（有用性）",
            description=(
                "用同一套评分卡给人工参考回复打分，参考回复的有用性应稳定高于自动回复。"
                "若胜出率过低，说明指标口径无法区分人工与自动回复。"
            ),
            value=f"{wins}/{len(shared)} 条中参考回复更高，胜出率 {rate:.0%}（要求 >= 80%）",
            passed=rate >= 0.8,
        )

    @staticmethod
    def _check_reference_margin(
        auto_judgements: dict[str, CaseJudgement],
        human_judgements: dict[str, CaseJudgement],
    ) -> ValidationFinding:
        shared = [case_id for case_id in auto_judgements if case_id in human_judgements]
        deltas = [
            human_judgements[case_id].score(MetricName.HELPFULNESS)
            - auto_judgements[case_id].score(MetricName.HELPFULNESS)
            for case_id in shared
        ]
        margin = sum(deltas) / len(deltas) if deltas else 0.0
        return ValidationFinding(
            name="参考回复有用性领先幅度",
            description="平均差距过小意味着评分卡对『主动代办 vs 推回用户』不敏感。",
            value=f"平均领先 {margin:+.2f} 分（要求 >= +0.50 分）",
            passed=margin >= 0.5,
        )

    def _check_worst_case_separation(
        self,
        case_scores: Sequence[CaseScore],
        ranked_case_ids: Sequence[str],
        overall_score: float,
    ) -> ValidationFinding:
        score_by_id = {item.case_id: item.final_score for item in case_scores}
        gold_scores = {
            case_id: score_by_id[case_id]
            for case_id in ANNOTATED_WORST_CASES
            if case_id in score_by_id
        }
        total = len(ranked_case_ids)
        window = max(1, ceil(total * 0.4))
        tail = set(ranked_case_ids[:window])
        hits = [case_id for case_id in gold_scores if case_id in tail]
        missed = [case_id for case_id in gold_scores if case_id not in tail]
        hit_rate = len(hits) / len(gold_scores) if gold_scores else 0.0
        detail = "、".join(
            f"{case_id}={gold_scores[case_id]:.2f}" for case_id in ANNOTATED_WORST_CASES
            if case_id in gold_scores
        )
        value = (
            f"最差集得分 {detail}（整体均分 {overall_score:.2f}）；"
            f"{len(hits)}/{len(gold_scores)} 条落在最差 {window} 名内，"
            f"命中率 {hit_rate:.0%}（要求 >= 60%）"
        )
        if missed:
            value += f"；未命中：{'、'.join(missed)}"
        return ValidationFinding(
            name="人工标注最差集命中率",
            description=(
                "人工评语中问题最明确的 5 条 case 作为已知最差集，"
                "检查评估方法能否把它们排到整体分布的最差 40% 区间内。"
                "用命中率而非全量命中，是因为人工判断本身也存在尺度差异。"
            ),
            value=value,
            passed=hit_rate >= 0.6,
        )

    @staticmethod
    def _check_annotator_group_separation(
        references: dict[str, HumanReference],
        auto_judgements: dict[str, CaseJudgement],
    ) -> ValidationFinding:
        groups: dict[str, list[str]] = {"problematic": [], "acceptable": [], "mixed": []}
        for case_id, reference in references.items():
            groups[classify_annotator_note(reference.annotator_notes)].append(case_id)

        def mean_helpfulness(case_ids: Sequence[str]) -> float:
            scores = [
                auto_judgements[case_id].score(MetricName.HELPFULNESS)
                for case_id in case_ids
                if case_id in auto_judgements
            ]
            return sum(scores) / len(scores) if scores else 0.0

        problematic_mean = mean_helpfulness(groups["problematic"])
        acceptable_mean = mean_helpfulness(groups["acceptable"])
        separated = bool(groups["problematic"] and groups["acceptable"]) and (
            problematic_mean < acceptable_mean
        )
        return ValidationFinding(
            name="人工评语分组区分度",
            description=(
                "用评语关键词把人工评语粗分为『指出问题』与『总体认可』两组，"
                "问题组的有用性均值应更低。该检验依赖关键词代理指标，精度有限。"
            ),
            value=(
                f"问题组 {len(groups['problematic'])} 条均值 {problematic_mean:.2f}；"
                f"认可组 {len(groups['acceptable'])} 条均值 {acceptable_mean:.2f}；"
                f"未分组 {len(groups['mixed'])} 条"
            ),
            passed=separated,
        )
