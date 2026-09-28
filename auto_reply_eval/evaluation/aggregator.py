"""把逐条评分聚合成可汇报的结果。

总分口径 = 加权平均 × 闸门惩罚：

1. 加权平均反映四个指标对业务目标的相对重要性；
2. 闸门惩罚表达优先级——底线指标不达标时，不允许被其他指标的高分掩盖。
"""

from __future__ import annotations

from typing import Sequence

from auto_reply_eval.metrics.definitions import SCORE_MAX, SCORE_MIN, SCORE_STEP
from auto_reply_eval.models import (
    CaseJudgement,
    CaseScore,
    HumanReference,
    MetricDefinition,
    MetricName,
    MetricStatistics,
)


class ScoreAggregator:
    """负责单条打分、指标分布与总分计算。"""

    def __init__(
        self,
        definitions: Sequence[MetricDefinition],
        worst_case_count: int = 3,
        pass_threshold: float = 3.5,
    ) -> None:
        self._definitions = tuple(definitions)
        self._worst_case_count = worst_case_count
        self._pass_threshold = pass_threshold

    @property
    def worst_case_count(self) -> int:
        return self._worst_case_count

    @property
    def pass_threshold(self) -> float:
        return self._pass_threshold

    def build_case_score(
        self,
        judgement: CaseJudgement,
        reference: HumanReference | None = None,
        human_judgement: CaseJudgement | None = None,
    ) -> CaseScore:
        """计算单条 case 的明细分数。"""

        metric_scores = {
            definition.name: judgement.score(definition.name)
            for definition in self._definitions
        }
        metric_reasons = {
            definition.name: judgement.output.reason_of(definition.name)
            for definition in self._definitions
        }

        weighted = sum(
            definition.weight * metric_scores[definition.name]
            for definition in self._definitions
        )

        penalties: list[str] = []
        factor = 1.0
        for definition in self._definitions:
            if definition.penalty_factor is None:
                continue
            score = metric_scores[definition.name]
            if score < definition.penalty_threshold:
                factor *= definition.penalty_factor
                penalties.append(
                    f"{definition.label} {score:g} 分低于底线 {definition.penalty_threshold:g} 分"
                    f"，总分 ×{definition.penalty_factor:g}"
                )

        human_scores = (
            {
                definition.name: human_judgement.score(definition.name)
                for definition in self._definitions
            }
            if human_judgement is not None
            else None
        )

        return CaseScore(
            case_id=judgement.case_id,
            user_question=judgement.user_question,
            auto_reply=judgement.text,
            metric_scores=metric_scores,
            metric_reasons=metric_reasons,
            weighted_score=round(weighted, 3),
            final_score=round(weighted * factor, 3),
            applied_penalties=penalties,
            unsupported_claims=judgement.output.unsupported_claims,
            missing_actions=judgement.output.missing_actions,
            human_reference=reference.human_reference if reference else None,
            annotator_notes=reference.annotator_notes if reference else None,
            human_metric_scores=human_scores,
        )

    def build_metric_statistics(
        self, judgements: Sequence[CaseJudgement]
    ) -> list[MetricStatistics]:
        """计算每个指标的均值与分布。"""

        statistics: list[MetricStatistics] = []
        for definition in self._definitions:
            scores = [judgement.score(definition.name) for judgement in judgements]
            distribution: dict[float, int] = {}
            for bucket in self._buckets():
                distribution[bucket] = 0
            for score in scores:
                distribution[self._bucket_of(score)] += 1
            statistics.append(
                MetricStatistics(
                    metric=definition.name,
                    label=definition.label,
                    weight=definition.weight,
                    mean=round(sum(scores) / len(scores), 3) if scores else 0.0,
                    minimum=min(scores) if scores else 0.0,
                    maximum=max(scores) if scores else 0.0,
                    distribution=distribution,
                )
            )
        return statistics

    def overall_score(self, case_scores: Sequence[CaseScore]) -> float:
        if not case_scores:
            return 0.0
        return round(sum(item.final_score for item in case_scores) / len(case_scores), 3)

    def pass_rate(self, case_scores: Sequence[CaseScore]) -> float:
        if not case_scores:
            return 0.0
        passed = sum(1 for item in case_scores if item.final_score >= self._pass_threshold)
        return round(passed / len(case_scores), 3)

    def rank_worst(self, case_scores: Sequence[CaseScore], limit: int | None = None) -> list[CaseScore]:
        """按总分升序排名；同分时有用性更低者更差。"""

        count = limit if limit is not None else self._worst_case_count
        ordered = sorted(
            case_scores,
            key=lambda item: (
                item.final_score,
                item.metric_scores[MetricName.HELPFULNESS],
                item.case_id,
            ),
        )
        return list(ordered[:count])

    def rank_all(self, case_scores: Sequence[CaseScore]) -> list[str]:
        """返回从最差到最好的 case_id 序列。"""

        ordered = sorted(
            case_scores,
            key=lambda item: (
                item.final_score,
                item.metric_scores[MetricName.HELPFULNESS],
                item.case_id,
            ),
        )
        return [item.case_id for item in ordered]

    def tied_with_worst(
        self, case_scores: Sequence[CaseScore], worst_cases: Sequence[CaseScore]
    ) -> list[str]:
        """找出与最差档同分、但未被逐条展开的 case。

        本批数据中大量回复"同病相怜"，分数高度并列，
        显式列出并列项可以避免读者误以为只有这几条有问题。
        """

        if not worst_cases:
            return []
        cutoff = worst_cases[-1].final_score
        shown = {item.case_id for item in worst_cases}
        return sorted(
            item.case_id
            for item in case_scores
            if item.case_id not in shown and item.final_score == cutoff
        )

    @staticmethod
    def _buckets() -> list[float]:
        steps = int(round((SCORE_MAX - SCORE_MIN) / SCORE_STEP)) + 1
        return [round(SCORE_MIN + index * SCORE_STEP, 1) for index in range(steps)]

    @staticmethod
    def _bucket_of(score: float) -> float:
        snapped = round(score / SCORE_STEP) * SCORE_STEP
        return round(max(SCORE_MIN, min(SCORE_MAX, snapped)), 1)
