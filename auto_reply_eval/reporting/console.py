"""控制台报告渲染器。

只输出可在普通终端里直接阅读的文本，便于在 CI 日志里留痕或截图。
"""

from __future__ import annotations

from auto_reply_eval.models import EvaluationResult
from auto_reply_eval.reporting.notes import build_conclusions

_LINE = "=" * 78
_SUB = "-" * 78


def _bar(count: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return ""
    return "#" * round(count / total * width)


def render_console(result: EvaluationResult) -> str:
    """渲染终端可读的评估摘要。"""

    lines: list[str] = [
        _LINE,
        "客服自动回复质量评估报告",
        _LINE,
        f"运行模式 : {result.mode}",
        f"裁判模型 : {result.model}",
        f"样本量   : {result.case_count} 条自动回复",
        f"生成时间 : {result.generated_at:%Y-%m-%d %H:%M:%S}",
        "",
        f"整体得分 : {result.overall_score:.2f} / 5.00"
        f"    及格率(>= {result.pass_threshold:g}) : {result.pass_rate:.0%}",
    ]
    if result.human_overall_score is not None:
        lines.append(
            f"人工参考 : {result.human_overall_score:.2f} / 5.00"
            f"    （自动回复落后 {result.human_overall_score - result.overall_score:.2f} 分）"
        )

    lines += ["", _SUB, "指标总览", _SUB, f"{'指标':<18}{'权重':>6}{'均值':>8}{'最低':>8}{'最高':>8}"]
    for statistics in result.metric_statistics:
        lines.append(
            f"{statistics.label:<18}{statistics.weight:>6g}{statistics.mean:>8.2f}"
            f"{statistics.minimum:>8g}{statistics.maximum:>8g}"
        )

    lines += ["", _SUB, "结论与建议", _SUB]
    for conclusion in build_conclusions(result):
        lines.append(f"- {conclusion}")

    lines += ["", _SUB, "指标分布（分值 : 条数）", _SUB]
    for statistics in result.metric_statistics:
        lines.append(f"[{statistics.label}]")
        for score in sorted(statistics.distribution, reverse=True):
            count = statistics.distribution[score]
            lines.append(f"  {score:>4g} | {_bar(count, result.case_count):<30} {count}")
        lines.append("")

    lines += [_SUB, f"最差 {len(result.worst_cases)} 条 case", _SUB]
    for rank, case in enumerate(result.worst_cases, start=1):
        scores = "  ".join(
            f"{metric.value}={score:g}" for metric, score in case.metric_scores.items()
        )
        lines.append(f"{rank}. {case.case_id}  总分 {case.final_score:.2f}")
        lines.append(f"   问题   : {case.user_question}")
        lines.append(f"   得分   : {scores}")
        for reason in case.metric_reasons.values():
            lines.append(f"   判据   : {reason}")
        for action in case.missing_actions:
            lines.append(f"   缺失   : {action}")
        lines.append("")

    if result.tied_worst_case_ids:
        lines.append(
            f"注：与最差档（{result.worst_cases[-1].final_score:.2f} 分）并列的还有 "
            f"{'、'.join(result.tied_worst_case_ids)}，属于同一类病灶。"
        )
        lines.append("")

    lines += [_SUB, "与人工标注的一致性校验", _SUB]
    for finding in result.validations:
        status = "通过  " if finding.passed else "未通过"
        lines.append(f"[{status}] {finding.name}")
        lines.append(f"         {finding.value}")

    return "\n".join(lines)
