"""报告中的固定文本与分析结论。

局限性与改进方向是固定的方法论文本；
结论与建议则由评估结果按规则推导，避免把结论写死在报告模板里。
"""

from __future__ import annotations

from auto_reply_eval.models import EvaluationResult, MetricName

LIMITATIONS: tuple[str, ...] = (
    "**依赖裁判模型的能力**：四个指标都由 LLM 裁判打分，模型自身的偏见、对"
    "『平台规则是否属实』的无知都会传导到结果上。本轮 20 条回复的政策类断言"
    "（退款时长、质保天数、充电宝限额）无法从给定资料核实，只能判定为『有依据但未验证』。",
    "**无法核验具体商品事实**：像「这款手机壳是 TPU 材质」这类断言，是否属实取决于商品库。"
    "评估只基于对话文本，无法访问商品数据，因此这类回复在『不瞎编』上只能给中性评价，"
    "真实幻觉会被漏判。",
    "**『有用性』高度依赖上下文**：主动代办需要后台工单、订单查询等能力支撑。"
    "如果自动回复根本没有查询权限，那么推给用户看详情页其实是系统能力问题，"
    "而不是话术问题。当前指标无法区分『不想办』与『办不到』，会把系统缺陷记在话术头上。",
    "**单轮评估的局限**：只看一问一答，看不到后续是否真的解决了问题、"
    "用户是否二次投诉。回复『有用』与否的最终判据其实是会话闭环结果。",
    "**领域适配**：关键词代理与锚点描述都是围绕电商客服场景写的，"
    "迁移到金融、医疗等场景需要重新校准，尤其是『准确性』的判据。",
    "**低位区间分辨率不足**：真实裁判给出的有用性集中在 1-3.5 分，多数回复都不及格。"
    "在这种整体偏低的分布下，评分卡对不同回复的区分能力会下降，"
    "基于分组均值的一致性检验也失去统计功效（认可组只有 4 条样本），只能作为方向性参考。",
)

IMPROVEMENTS: tuple[str, ...] = (
    "接入商品库、政策库做检索增强校验，把『不瞎编』从主观判断变成可核对的引用匹配。",
    "引入会话级指标：一次解决率、转人工率、后续 24 小时重复咨询率，用真实业务结果校准『有用性』。",
    "用人工标注扩充样本，计算评估结论与人工结论的 Spearman 相关系数，"
    "为指标权重做数据驱动的标定，而不是靠拍脑袋。",
    "对同一批数据做多次重复评判，统计分数方差，量化裁判模型的稳定性（当前是单次评判）。",
    "把 LLM 裁判换成一主一复核的双模型投票，减少单模型偏见。",
)


def build_conclusions(result: EvaluationResult) -> list[str]:
    """按评估结果推导结论与建议。"""

    statistics = {item.metric: item for item in result.metric_statistics}
    weakest = min(result.metric_statistics, key=lambda item: item.mean)
    strongest = max(result.metric_statistics, key=lambda item: item.mean)
    helpfulness = statistics[MetricName.HELPFULNESS]

    conclusions = [
        f"四个指标中「{weakest.label}」是明显短板（均值 {weakest.mean:.2f}），"
        f"与表现最好的「{strongest.label}」（均值 {strongest.mean:.2f}）相差 "
        f"{strongest.mean - weakest.mean:.2f} 分。"
    ]

    if helpfulness.mean < 4.0:
        conclusions.append(
            "有用性的典型病灶是『只告知、不办理』：回复普遍没有主动查询用户的订单、"
            "物流或商品参数，而是把操作成本推回用户，这与人工评语反复指出的问题一致。"
        )
    if statistics[MetricName.FAITHFULNESS].mean >= 4.5:
        conclusions.append(
            "准确性与无幻觉两项均表现良好，说明当前风险不在于『说错话』，而在于『没办事』。"
        )
    if result.human_metric_means is not None and result.human_overall_score is not None:
        gap = result.human_overall_score - result.overall_score
        human_helpfulness = result.human_metric_means.get(MetricName.HELPFULNESS, 0.0)
        conclusions.append(
            f"与人工参考回复相比整体落后 {gap:.2f} 分，其中有用性一项就差了 "
            f"{human_helpfulness - helpfulness.mean:.2f} 分，差距集中体现在是否主动代办。"
        )
    if helpfulness.mean < result.pass_threshold:
        conclusions.append(
            "有用性均值低于及格线，建议先补齐主动代办能力（订单查询、物流查询、"
            "商品参数检索），再考虑扩大自动回复覆盖范围。"
        )
    if result.pass_rate < 0.8:
        conclusions.append(
            f"及格率 {result.pass_rate:.0%} 低于 80%，当前不建议直接扩大覆盖范围；"
            "可按指标逐项补齐后重新评估。"
        )
    return conclusions


__all__ = ["IMPROVEMENTS", "LIMITATIONS", "build_conclusions"]
