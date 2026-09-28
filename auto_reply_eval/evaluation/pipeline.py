"""评估流水线编排。"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Sequence

from auto_reply_eval.config import Settings
from auto_reply_eval.data.repository import DatasetRepository
from auto_reply_eval.evaluation.aggregator import ScoreAggregator
from auto_reply_eval.judge.reply_judge import ReplyJudge, summarize_output
from auto_reply_eval.llm.factory import create_llm_client
from auto_reply_eval.metrics.definitions import all_definitions
from auto_reply_eval.models import CaseJudgement, EvaluationResult, ReplySource
from auto_reply_eval.validation.human_reference import HumanReferenceValidator

ProgressCallback = Callable[[str], None]


class EvaluationPipeline:
    """串起"读数据 -> 逐条裁判 -> 聚合 -> 校验"的完整流程。"""

    def __init__(
        self,
        settings: Settings,
        repository: DatasetRepository,
        judge: ReplyJudge,
        aggregator: ScoreAggregator,
        validator: HumanReferenceValidator,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._judge = judge
        self._aggregator = aggregator
        self._validator = validator

    @classmethod
    def from_settings(cls, settings: Settings) -> EvaluationPipeline:
        """按配置装配整条流水线，依赖关系集中在这里。"""

        definitions = all_definitions()
        return cls(
            settings=settings,
            repository=DatasetRepository(settings.dataset_path, settings.human_ref_path),
            judge=ReplyJudge(
                client=create_llm_client(settings),
                definitions=definitions,
                max_retries=settings.max_retries,
            ),
            aggregator=ScoreAggregator(
                definitions=definitions,
                worst_case_count=settings.worst_case_count,
                pass_threshold=settings.pass_threshold,
            ),
            validator=HumanReferenceValidator(worst_case_count=settings.worst_case_count),
        )

    def run(
        self,
        limit: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> EvaluationResult:
        """执行一次完整评估。"""

        cases = self._repository.load_cases()
        if limit is not None:
            cases = cases[:limit]
        references = self._repository.load_human_references()

        auto_judgements: dict[str, CaseJudgement] = {}
        human_judgements: dict[str, CaseJudgement] = {}

        for case in cases:
            judgement = self._judge.judge(
                case_id=case.id,
                user_question=case.user_question,
                text=case.auto_reply,
                source=ReplySource.AUTO,
            )
            auto_judgements[case.id] = judgement
            if on_progress is not None:
                on_progress(f"[auto ] {case.id}  {summarize_output(judgement.output)}")

            reference = references.get(case.id)
            if reference is None:
                continue
            human_judgements[case.id] = self._judge.judge(
                case_id=case.id,
                user_question=case.user_question,
                text=reference.human_reference,
                source=ReplySource.HUMAN,
            )
            if on_progress is not None:
                on_progress(
                    f"[human] {case.id}  "
                    f"{summarize_output(human_judgements[case.id].output)}"
                )

        case_scores = [
            self._aggregator.build_case_score(
                judgement=auto_judgements[case.id],
                reference=references.get(case.id),
                human_judgement=human_judgements.get(case.id),
            )
            for case in cases
        ]

        overall_score = self._aggregator.overall_score(case_scores)
        ranked_case_ids = self._aggregator.rank_all(case_scores)
        worst_cases = self._aggregator.rank_worst(case_scores)
        metric_statistics = self._aggregator.build_metric_statistics(
            list(auto_judgements.values())
        )

        validations = self._validator.validate(
            references=references,
            auto_judgements=auto_judgements,
            human_judgements=human_judgements,
            case_scores=case_scores,
            ranked_case_ids=ranked_case_ids,
            overall_score=overall_score,
        )

        human_overall = (
            self._aggregator.overall_score(
                [
                    self._aggregator.build_case_score(judgement)
                    for judgement in human_judgements.values()
                ]
            )
            if human_judgements
            else None
        )
        human_metric_means = (
            {
                item.metric: item.mean
                for item in self._aggregator.build_metric_statistics(
                    list(human_judgements.values())
                )
            }
            if human_judgements
            else None
        )

        return EvaluationResult(
            mode=self._settings.mode.value,
            model=self._judge.client_name,
            generated_at=datetime.now(),
            case_count=len(case_scores),
            overall_score=overall_score,
            pass_threshold=self._aggregator.pass_threshold,
            pass_rate=self._aggregator.pass_rate(case_scores),
            metric_statistics=metric_statistics,
            case_scores=case_scores,
            worst_cases=worst_cases,
            validations=validations,
            human_overall_score=human_overall,
            human_metric_means=human_metric_means,
            tied_worst_case_ids=self._aggregator.tied_with_worst(case_scores, worst_cases),
        )
