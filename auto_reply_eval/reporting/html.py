"""HTML 报告渲染器。

产出单文件、零依赖的 HTML，便于直接在浏览器中查看、截图，或作为静态页面部署。
"""

from __future__ import annotations

from html import escape

from auto_reply_eval.models import EvaluationResult, MetricDefinition, MetricName, MetricStatistics
from auto_reply_eval.reporting.notes import (
    IMPROVEMENTS,
    LIMITATIONS,
    SCORE_BANDS,
    build_conclusions,
    describe_band,
)

_STYLE = """
:root {
  --bg: #0f1115; --panel: #171a21; --panel-2: #1e222b; --line: #2a2f3a;
  --text: #e8eaed; --muted: #9aa3b2; --good: #35c26b; --warn: #e0a020; --bad: #e05a5a;
  --accent: #4a9eff;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 20px 64px; background: var(--bg); color: var(--text);
  font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", system-ui, sans-serif;
  line-height: 1.7;
}
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 8px; }
h2 { font-size: 19px; margin: 36px 0 12px; padding-left: 10px; border-left: 4px solid var(--accent); }
h3 { font-size: 16px; margin: 22px 0 8px; }
.meta { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
.meta code { background: var(--panel-2); padding: 2px 6px; border-radius: 4px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 18px 0 8px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; }
.card .k { color: var(--muted); font-size: 13px; }
.card .v { font-size: 30px; font-weight: 700; margin-top: 4px; }
.card .v small { font-size: 14px; color: var(--muted); font-weight: 400; }
table { width: 100%; border-collapse: collapse; margin: 10px 0 6px; font-size: 14px; background: var(--panel); }
th, td { border: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }
th { background: var(--panel-2); color: var(--muted); font-weight: 600; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.bar { display: flex; align-items: center; gap: 8px; }
.bar .fill { height: 12px; border-radius: 6px; background: var(--accent); min-width: 1px; }
.bar .fill.low { background: var(--bad); }
.bar .fill.mid { background: var(--warn); }
.bar .fill.high { background: var(--good); }
.case { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 18px 20px; margin: 14px 0; }
.case .score { float: right; font-size: 22px; font-weight: 700; }
.case .score.bad { color: var(--bad); }
.case .score.mid { color: var(--warn); }
.case .score.good { color: var(--good); }
.q { color: var(--muted); font-size: 13px; }
.reply { background: var(--panel-2); border-radius: 8px; padding: 10px 12px; margin: 6px 0 12px; }
ul { margin: 6px 0 6px 20px; padding: 0; }
li { margin: 4px 0; }
.pass { color: var(--good); font-weight: 600; }
.fail { color: var(--bad); font-weight: 600; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; border: 1px solid var(--line); color: var(--muted); }
.band { background: var(--panel); border: 1px solid var(--line); border-left: 4px solid var(--warn);
        border-radius: 8px; padding: 12px 16px; margin: 14px 0 4px; font-size: 15px; }
.band b { color: var(--warn); }
.footer { color: var(--muted); font-size: 12px; margin-top: 40px; border-top: 1px solid var(--line); padding-top: 14px; }
"""


def _tone_class(score: float) -> str:
    if score >= 3.5:
        return "good"
    return "mid" if score >= 2.5 else "bad"


def _fill_class(score: float) -> str:
    if score >= 4.0:
        return "high"
    return "mid" if score >= 3.0 else "low"


def _render_metric_distribution(statistics: MetricStatistics, total: int) -> str:
    rows: list[str] = []
    for score in sorted(statistics.distribution, reverse=True):
        count = statistics.distribution[score]
        width = round(count / total * 100) if total else 0
        rows.append(
            f'<tr><td class="num">{score:g}</td><td class="num">{count}</td>'
            f'<td><div class="bar"><div class="fill {_fill_class(score)}" style="width:{width}%"></div>'
            f"</div></td></tr>"
        )
    return (
        f"<h3>{escape(statistics.label)}"
        f'<span class="pill" style="margin-left:8px">均值 {statistics.mean:.2f} · 权重 {statistics.weight:g}'
        f"</span></h3>"
        '<table><thead><tr><th class="num">分值</th><th class="num">条数</th><th>分布</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_worst_cases(result: EvaluationResult) -> str:
    blocks: list[str] = []
    for rank, case in enumerate(result.worst_cases, start=1):
        metric_rows = "".join(
            f"<tr><td>{escape(metric.value)}</td><td class='num'>{score:g}</td>"
            f"<td>{escape(case.metric_reasons.get(metric, ''))}</td></tr>"
            for metric, score in case.metric_scores.items()
        )
        extra: list[str] = []
        if case.applied_penalties:
            extra.append(
                "<p><b>闸门惩罚</b></p><ul>"
                + "".join(f"<li>{escape(item)}</li>" for item in case.applied_penalties)
                + "</ul>"
            )
        if case.missing_actions:
            extra.append(
                "<p><b>缺失的关键动作</b></p><ul>"
                + "".join(f"<li>{escape(item)}</li>" for item in case.missing_actions)
                + "</ul>"
            )
        if case.human_reference:
            extra.append(
                f'<p class="q">人工参考回复</p><div class="reply">{escape(case.human_reference)}</div>'
                f'<p class="q">人工评语：{escape(case.annotator_notes or "")}</p>'
            )
        blocks.append(
            f'<div class="case">'
            f'<span class="score {_tone_class(case.final_score)}">{case.final_score:.2f}</span>'
            f"<h3>{rank}. {escape(case.case_id)}</h3>"
            f'<p class="q">用户问题：{escape(case.user_question)}</p>'
            f'<div class="reply">{escape(case.auto_reply)}</div>'
            "<table><thead><tr><th>指标</th><th class='num'>得分</th><th>裁判判据</th></tr></thead>"
            f"<tbody>{metric_rows}</tbody></table>"
            f"{''.join(extra)}</div>"
        )
    return "".join(blocks)


def _render_rubric(definitions: list[MetricDefinition]) -> str:
    rows = "".join(
        f"<tr><td>{definition.priority}</td><td>{escape(definition.label)}</td>"
        f"<td class='num'>{definition.weight:g}</td><td>{escape(definition.definition)}</td>"
        f"<td>{escape(definition.quantification)}</td></tr>"
        for definition in definitions
    )
    return (
        "<table><thead><tr><th>优先级</th><th>指标</th><th class='num'>权重</th>"
        "<th>定义</th><th>量化方式</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_html(result: EvaluationResult, definitions: list[MetricDefinition]) -> str:
    """渲染单文件 HTML 报告。"""

    human_card = ""
    if result.human_overall_score is not None:
        human_card = (
            f'<div class="card"><div class="k">人工参考回复整体分</div>'
            f'<div class="v">{result.human_overall_score:.2f}<small> / 5</small></div></div>'
        )

    summary_rows = "".join(
        f"<tr><td>{escape(item.label)}</td><td class='num'>{item.weight:g}</td>"
        f"<td class='num'>{item.mean:.2f}</td><td class='num'>{item.minimum:g}</td>"
        f"<td class='num'>{item.maximum:g}</td></tr>"
        for item in result.metric_statistics
    )

    validation_rows = "".join(
        f"<tr><td>{escape(item.name)}</td><td>{escape(item.value)}</td>"
        f"<td class='{'pass' if item.passed else 'fail'}'>{'通过' if item.passed else '未通过'}</td></tr>"
        for item in result.validations
    )

    detail_rows = "".join(
        f"<tr><td class='num'>{rank}</td><td>{escape(case.case_id)}</td>"
        f"<td class='num'>{case.metric_scores[MetricName.FAITHFULNESS]:g}</td>"
        f"<td class='num'>{case.metric_scores[MetricName.ACCURACY]:g}</td>"
        f"<td class='num'>{case.metric_scores[MetricName.HELPFULNESS]:g}</td>"
        f"<td class='num'>{case.metric_scores[MetricName.TONE]:g}</td>"
        f"<td class='num'>{case.final_score:.2f}</td></tr>"
        for rank, case in enumerate(
            sorted(result.case_scores, key=lambda item: item.final_score), start=1
        )
    )

    distributions = "".join(
        _render_metric_distribution(item, result.case_count)
        for item in result.metric_statistics
    )
    limitations = "".join(f"<li>{escape(item)}</li>" for item in LIMITATIONS)
    improvements = "".join(f"<li>{escape(item)}</li>" for item in IMPROVEMENTS)
    conclusions = "".join(f"<li>{escape(item)}</li>" for item in build_conclusions(result))
    band_rows = "".join(
        f"<tr><td class='num'>&gt;= {threshold:g}</td><td>{escape(label)}</td>"
        f"<td>{escape(advice)}</td></tr>"
        for threshold, label, advice in SCORE_BANDS
    )
    tie_note = ""
    if result.tied_worst_case_ids:
        tie_note = (
            f'<p class="q">与最差档（{result.worst_cases[-1].final_score:.2f} 分）并列的还有 '
            f"{escape('、'.join(result.tied_worst_case_ids))}，属于同一类病灶（只告知、不办理），"
            "因此实际需要整改的 case 不止上面这几条。</p>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>客服自动回复质量评估报告</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="wrap">
  <h1>客服自动回复质量评估报告</h1>
  <p class="meta">
    生成时间 {result.generated_at:%Y-%m-%d %H:%M:%S} ·
    运行模式 <code>{escape(result.mode)}</code> ·
    裁判模型 <code>{escape(result.model)}</code> ·
    样本量 {result.case_count} 条
  </p>

  <div class="cards">
    <div class="card"><div class="k">整体得分</div>
      <div class="v">{result.overall_score:.2f}<small> / 5</small></div></div>
    <div class="card"><div class="k">及格率（>= {result.pass_threshold:g} 分）</div>
      <div class="v">{result.pass_rate:.0%}</div></div>
    {human_card}
  </div>

  <p class="band">得分解读：<b>{escape(describe_band(result.overall_score))}</b></p>

  <h2>1. 结论与建议</h2>
  <ul>{conclusions}</ul>
  <table><thead><tr><th class="num">总分区间</th><th>结论</th><th>建议动作</th></tr></thead>
  <tbody>{band_rows}</tbody></table>

  <h2>2. 指标总览</h2>
  <table><thead><tr><th>指标</th><th class="num">权重</th><th class="num">均值</th>
  <th class="num">最低</th><th class="num">最高</th></tr></thead>
  <tbody>{summary_rows}</tbody></table>

  <h2>3. 各指标分布</h2>
  {distributions}

  <h2>4. 最差 {len(result.worst_cases)} 条 case 及分析</h2>
  {_render_worst_cases(result)}
  {tie_note}

  <h2>5. 与人工标注的一致性校验</h2>
  <table><thead><tr><th>校验项</th><th>结果</th><th>结论</th></tr></thead>
  <tbody>{validation_rows}</tbody></table>

  <h2>6. 评分卡（指标定义与优先级）</h2>
  {_render_rubric(definitions)}

  <h2>7. 局限性</h2>
  <ul>{limitations}</ul>

  <h2>8. 改进方向</h2>
  <ul>{improvements}</ul>

  <h2>9. 全部 case 明细（按总分升序）</h2>
  <table><thead><tr><th class="num">排名</th><th>case</th><th class="num">不瞎编</th>
  <th class="num">准确</th><th class="num">有用</th><th class="num">语气</th>
  <th class="num">总分</th></tr></thead>
  <tbody>{detail_rows}</tbody></table>

  <p class="footer">
    本报告由 auto_reply_eval 自动生成。指标定义、量化方式与权重均可通过
    <code>auto_reply_eval/metrics/definitions.py</code> 调整。
  </p>
</div>
</body>
</html>"""
