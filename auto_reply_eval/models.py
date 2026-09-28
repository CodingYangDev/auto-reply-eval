"""评估流水线使用的领域模型。

全部使用 Pydantic 建模，保证：类型安全、可校验、可序列化为报告产物。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MetricName(str, Enum):
    """四个可自动评估的指标。"""

    FAITHFULNESS = "faithfulness"
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    TONE = "tone"


class ReplySource(str, Enum):
    """被评估回复的来源。"""

    AUTO = "auto"
    HUMAN = "human"


class AutoReplyCase(BaseModel):
    """一条待评估的线上自动回复。"""

    id: str
    user_question: str
    auto_reply: str


class HumanReference(BaseModel):
    """人工标注的参考回复与评语。"""

    id: str
    human_reference: str
    annotator_notes: str


class MetricDefinition(BaseModel):
    """指标定义：把业务方的模糊要求翻译成可执行的评分口径。"""

    name: MetricName
    label: str
    priority: int = Field(ge=1, description="1 表示最高优先级")
    weight: float = Field(gt=0.0, le=1.0)
    definition: str
    quantification: str
    rationale: str
    anchors: dict[int, str] = Field(default_factory=dict)
    penalty_factor: float | None = Field(
        default=None,
        ge=0.1,
        le=1.0,
        description="低于阈值时对该条回复总分的乘性惩罚，None 表示不设闸门",
    )
    penalty_threshold: float = 3.0


class MetricVerdict(BaseModel):
    """裁判对单个指标的打分。"""

    metric: MetricName
    score: float = Field(ge=1.0, le=5.0)
    reason: str


class JudgeOutput(BaseModel):
    """裁判模型的完整结构化输出。"""

    verdicts: list[MetricVerdict]
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_actions: list[str] = Field(default_factory=list)

    def score_of(self, metric: MetricName) -> float:
        for verdict in self.verdicts:
            if verdict.metric is metric:
                return verdict.score
        raise KeyError(f"裁判输出缺少指标 {metric.value} 的评分")

    def reason_of(self, metric: MetricName) -> str:
        for verdict in self.verdicts:
            if verdict.metric is metric:
                return verdict.reason
        return ""


class CaseJudgement(BaseModel):
    """单条回复的裁判结果。"""

    case_id: str
    user_question: str
    source: ReplySource
    text: str
    output: JudgeOutput

    def score(self, metric: MetricName) -> float:
        return self.output.score_of(metric)


class MetricStatistics(BaseModel):
    """单个指标的分布统计。"""

    metric: MetricName
    label: str
    weight: float
    mean: float
    minimum: float
    maximum: float
    distribution: dict[float, int] = Field(default_factory=dict)


class CaseScore(BaseModel):
    """单条 case 的最终评分明细。"""

    case_id: str
    user_question: str
    auto_reply: str
    metric_scores: dict[MetricName, float]
    metric_reasons: dict[MetricName, str]
    weighted_score: float
    final_score: float
    applied_penalties: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_actions: list[str] = Field(default_factory=list)
    human_reference: str | None = None
    annotator_notes: str | None = None
    human_metric_scores: dict[MetricName, float] | None = None


class ValidationFinding(BaseModel):
    """一条与人工标注的一致性校验结论。"""

    name: str
    description: str
    value: str
    passed: bool


class EvaluationResult(BaseModel):
    """一次完整评估的产物，报告渲染的唯一数据源。"""

    mode: str
    model: str
    generated_at: datetime
    case_count: int
    overall_score: float
    pass_threshold: float
    pass_rate: float
    metric_statistics: list[MetricStatistics]
    case_scores: list[CaseScore]
    worst_cases: list[CaseScore]
    validations: list[ValidationFinding]
    human_overall_score: float | None = None
    human_metric_means: dict[MetricName, float] | None = None
    tied_worst_case_ids: list[str] = Field(default_factory=list)
