"""评估流水线的端到端测试（mock 模式，无需 API Key）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_reply_eval.config import LLMMode, PROJECT_ROOT, Settings
from auto_reply_eval.data.repository import DatasetRepository
from auto_reply_eval.evaluation.pipeline import EvaluationPipeline
from auto_reply_eval.metrics.definitions import all_definitions
from auto_reply_eval.models import EvaluationResult, MetricName

DATASET_PATH = PROJECT_ROOT / "task3_auto_replies.json"
HUMAN_REF_PATH = PROJECT_ROOT / "task3_human_ref.json"


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> EvaluationResult:
    settings = Settings(
        mode=LLMMode.MOCK,
        dataset_path=DATASET_PATH,
        human_ref_path=HUMAN_REF_PATH,
        report_dir=tmp_path_factory.mktemp("reports"),
        cache_dir=tmp_path_factory.mktemp("cache"),
    )
    return EvaluationPipeline.from_settings(settings).run()


def test_dataset_and_references_are_aligned() -> None:
    repository = DatasetRepository(DATASET_PATH, HUMAN_REF_PATH)
    cases = repository.load_cases()
    references = repository.load_human_references()

    assert len(cases) == 20
    assert {case.id for case in cases} == set(references)


def test_every_case_gets_all_metric_scores(result: EvaluationResult) -> None:
    assert result.case_count == 20
    for case_score in result.case_scores:
        assert set(case_score.metric_scores) == set(MetricName)
        assert set(case_score.metric_reasons) == set(MetricName)
        for score in case_score.metric_scores.values():
            assert 1.0 <= score <= 5.0


def test_overall_score_is_within_range(result: EvaluationResult) -> None:
    assert 1.0 <= result.overall_score <= 5.0
    assert 0.0 <= result.pass_rate <= 1.0
    assert len(result.metric_statistics) == len(MetricName)


def test_metric_distribution_covers_all_cases(result: EvaluationResult) -> None:
    for statistics in result.metric_statistics:
        assert sum(statistics.distribution.values()) == result.case_count


def test_gate_penalty_lowers_final_score(result: EvaluationResult) -> None:
    penalised = [case for case in result.case_scores if case.applied_penalties]
    assert penalised, "至少应有一条回复触发底线闸门"
    for case in penalised:
        assert case.final_score < case.weighted_score


def test_worst_cases_are_sorted_ascending(result: EvaluationResult) -> None:
    scores = [case.final_score for case in result.worst_cases]
    assert scores == sorted(scores)
    assert len(result.worst_cases) == 3
    assert result.worst_cases[0].final_score == min(
        case.final_score for case in result.case_scores
    )


def test_validations_cover_all_four_checks(result: EvaluationResult) -> None:
    assert len(result.validations) == 4
    names = {finding.name for finding in result.validations}
    assert "参考回复胜出率（有用性）" in names
    assert "人工标注最差集整体靠后" in names


def test_reports_are_written(
    result: EvaluationResult, tmp_path: Path
) -> None:
    from auto_reply_eval.cli import write_reports

    paths = write_reports(result, tmp_path)
    assert len(paths) == 3
    for path in paths:
        assert path.exists() and path.stat().st_size > 0
    assert "客服自动回复质量评估报告" in (tmp_path / "eval_report.html").read_text(
        encoding="utf-8"
    )
