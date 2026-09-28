"""Markdown 报告渲染器。"""

from __future__ import annotations

from auto_reply_eval.models import EvaluationResult, MetricName, MetricStatistics
from auto_reply_eval.reporting.notes import (
    IMPROVEMENTS,
    LIMITATIONS,
    build_conclusions,
)

_BAR_UNIT = "█"


def _bar(count: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return ""
    filled = round(count / total * width)
    return _BAR_UNIT * filled


def render_metric_block(statistics: MetricStatistics, total: int) -> list[str]:
    lines = [
        f"### {statistics.label}（均值 {statistics.mean:.2f} / 权重 {statistics.weight:g}）",
        "",
        "| 分值 | 条数 | 分布 |",
        "| --- | ---: | --- |",
    ]
    for score in sorted(statistics.distribution):
        count = statistics.distribution[score]
        lines.append(f"| {score:g} | {count} | {_bar(count, total)} |")
    lines.append("")
    return lines


def render_markdown(result: EvaluationResult) -> str:
    """把评估结果渲染成 Markdown 报告。"""

    lines: list[str] = [
        "# 客服自动回复质量评估报告",
        "",
        f"- 生成时间：{result.generated_at:%Y-%m-%d %H:%M:%S}",
        f"- 运行模式：`{result.mode}`",
        f"- 裁判模型：`{result.model}`",
        f"- 样本量：{result.case_count} 条自动回复",
        "",
        "## 1. 整体结论",
        "",
        f"- **整体得分：{result.overall_score:.2f} / 5.00**",
        f"- 及格率（单条得分 >= {result.pass_threshold:g}）：{result.pass_rate:.0%}",
    ]
    if result.human_overall_score is not None:
        gap = result.human_overall_score - result.overall_score
        lines.append(
            f"- 人工参考回复整体得分：{result.human_overall_score:.2f} / 5.00"
            f"（自动回复落后 {gap:.2f} 分）"
        )
    lines += ["", "| 指标 | 权重 | 均值 | 最低 | 最高 |", "| --- | ---: | ---: | ---: | ---: |"]
    for statistics in result.metric_statistics:
        lines.append(
            f"| {statistics.label} | {statistics.weight:g} | {statistics.mean:.2f} "
            f"| {statistics.minimum:g} | {statistics.maximum:g} |"
        )

    lines += ["", "**结论与建议**", ""]
    lines += [f"- {item}" for item in build_conclusions(result)]

    lines += ["", "## 2. 各指标分布", ""]
    for statistics in result.metric_statistics:
        lines += render_metric_block(statistics, result.case_count)

    lines += [
        f"## 3. 最差 {len(result.worst_cases)} 条 case 及分析",
        "",
    ]
    for rank, case in enumerate(result.worst_cases, start=1):
        lines += [
            f"### {rank}. {case.case_id} （总分 {case.final_score:.2f}）",
            "",
            f"**用户问题**：{case.user_question}",
            "",
            f"**自动回复**：{case.auto_reply}",
            "",
            "| 指标 | 得分 | 裁判判据 |",
            "| --- | ---: | --- |",
        ]
        for metric, score in case.metric_scores.items():
            reason = case.metric_reasons.get(metric, "").replace("|", "／")
            lines.append(f"| {metric.value} | {score:g} | {reason} |")
        lines.append("")
        if case.applied_penalties:
            lines.append("**闸门惩罚**：")
            lines += [f"- {item}" for item in case.applied_penalties]
            lines.append("")
        if case.missing_actions:
            lines.append("**缺失的关键动作**：")
            lines += [f"- {item}" for item in case.missing_actions]
            lines.append("")
        if case.human_reference:
            lines += [
                f"**人工参考回复**：{case.human_reference}",
                "",
                f"**人工评语**：{case.annotator_notes}",
                "",
            ]

    if result.tied_worst_case_ids:
        lines += [
            f"> 说明：与最差档（{result.worst_cases[-1].final_score:.2f} 分）并列的还有 "
            f"{'、'.join(result.tied_worst_case_ids)}，属于同一类病灶（只告知、不办理），"
            "因此实际需要整改的 case 不止上面这几条。",
            "",
        ]

    lines += ["## 4. 与人工标注的一致性校验", "", "| 校验项 | 结果 | 结论 |", "| --- | --- | --- |"]
    for finding in result.validations:
        status = "通过" if finding.passed else "未通过"
        lines.append(f"| {finding.name} | {finding.value} | {status} |")
    lines.append("")
    for finding in result.validations:
        lines.append(f"- **{finding.name}**：{finding.description}")
    lines.append("")

    lines += ["## 5. 局限性", ""]
    lines += [f"- {item}" for item in LIMITATIONS]
    lines += ["", "## 6. 改进方向", ""]
    lines += [f"- {item}" for item in IMPROVEMENTS]

    lines += [
        "",
        "## 7. 全部 case 明细（按总分升序）",
        "",
        "| 排名 | case | 不瞎编 | 准确 | 有用 | 语气 | 总分 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(result.case_scores, key=lambda item: item.final_score)
    for rank, case in enumerate(ordered, start=1):
        lines.append(
            f"| {rank} | {case.case_id} "
            f"| {case.metric_scores[MetricName.FAITHFULNESS]:g} "
            f"| {case.metric_scores[MetricName.ACCURACY]:g} "
            f"| {case.metric_scores[MetricName.HELPFULNESS]:g} "
            f"| {case.metric_scores[MetricName.TONE]:g} "
            f"| {case.final_score:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)
