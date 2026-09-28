"""指标注册表。

业务方的原始要求只有四个词："准确、有用、语气好、不能瞎编"。
本模块把这四个词翻译成可打分、可复现的评分口径，并显式定义它们之间的优先级关系。

优先级通过两种机制落地：
1. ``weight``  —— 反映各指标对"是否值得扩大自动回复覆盖范围"这一业务决策的影响权重；
2. ``penalty_factor`` —— 硬闸门，指标低于 ``penalty_threshold`` 时对该条回复总分做乘性惩罚，
   用来表达"这是底线，不能靠其他指标补回来"。
"""

from __future__ import annotations

from auto_reply_eval.models import MetricDefinition, MetricName

SCORE_MIN = 1.0
SCORE_MAX = 5.0
SCORE_STEP = 0.5

METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        name=MetricName.FAITHFULNESS,
        label="不瞎编（无幻觉）",
        priority=1,
        weight=0.25,
        definition=(
            "回复中出现的所有事实性断言——政策条款、时限、金额、商品参数、赔付承诺——"
            "是否都有依据，没有凭空编造，也没有平台无法兑现的过度承诺。"
        ),
        quantification=(
            "裁判先抽取回复中的事实性断言，逐条判定『有依据 / 需人工核实 / 无依据编造』；"
            "1-5 分制，出现无依据的具体承诺或虚构的商品参数直接压到 2 分以下，"
            "无法从现有资料核实的具体数值按条轻微扣分。"
        ),
        rationale=(
            "业务方点名要求『不能瞎编』。编造信息会直接造成客诉、赔付与合规风险，"
            "是四类风险中代价最高的一类，因此设为最高优先级，并作为一票否决项："
            "低于 3 分时总分乘 0.70，不允许被其他指标的高分掩盖。"
        ),
        anchors={
            1: "存在明显编造，例如虚构商品参数、虚构平台赔付承诺",
            3: "通篇表述有依据，但个别具体数值无法从现有资料核实",
            5: "所有断言均可追溯，无过度承诺、无未经验证的具体参数",
        },
        penalty_factor=0.70,
        penalty_threshold=3.0,
    ),
    MetricDefinition(
        name=MetricName.ACCURACY,
        label="准确（对题且无误导）",
        priority=2,
        weight=0.25,
        definition=(
            "回复是否直接命中用户的真实诉求，且陈述正确、没有遗漏关键前提导致用户误解。"
            "它同时覆盖『答到点上』（相关性）与『说得对』（事实正确性）两面。"
        ),
        quantification=(
            "裁判对照用户问题与回复，分别判定『是否答到点上』与『陈述是否有误』；"
            "答非所问、或遗漏关键条件（例如把已过期的规则说成通用规则）逐项扣分。"
        ),
        rationale=(
            "『准确』必须先定义清楚才能量化：一个回复可以每句话都对，却完全没回答用户的问题。"
            "典型反例是用户说『退货流程太复杂搞不懂』，回复却把流程原样再念一遍——"
            "事实无误，但用户的困惑一点没解决。准确性是仅次于幻觉的底线，低于 3 分时总分乘 0.80。"
        ),
        anchors={
            1: "答非所问，或给出的信息会明确误导用户",
            3: "方向正确，但遗漏了关键前提或未覆盖用户的核心疑问",
            5: "正面命中用户诉求，信息正确且没有歧义",
        },
        penalty_factor=0.80,
        penalty_threshold=3.0,
    ),
    MetricDefinition(
        name=MetricName.HELPFULNESS,
        label="有用（主动推进解决）",
        priority=3,
        weight=0.35,
        definition=(
            "回复是否真正推动问题被解决：主动给出确定性结论、主动代办（查订单/查物流/查商品参数），"
            "或主动追问办成事所需的关键信息（订单号、具体商品）；"
            "而不是把操作成本转移回用户——『请看商品详情页』『请联系客服』『请您自行确认』。"
        ),
        quantification=(
            "裁判识别回复中的推进动作与推诿动作，按净倾向打分："
            "主动代办 / 追问关键信息 / 给出确定结论加分，引导自助 / 甩给其他渠道减分；"
            "当用户问的是『我的』具体问题时，推诿的扣分更重。"
        ),
        rationale=(
            "这是业务方『有用』的可操作化定义，也是本批数据里区分度最大的指标："
            "20 条自动回复大多事实正确、语气礼貌，但普遍停留在『告知』层面，没有『办理』。"
            "权重设为最高（0.35），因为它直接决定自动回复能不能替代人工、"
            "以及业务方能否放心扩大覆盖范围。有用性低于 3 分意味着用户仍需自己动手，"
            "这条回复几乎没有产生业务价值，因此总分乘 0.80。"
        ),
        anchors={
            1: "纯自助引导，用户仍需自己完成全部操作",
            3: "给出了正确信息与可执行路径，但没有主动代办，也未追问关键信息",
            5: "主动代办或主动追问关键信息，用户只需配合极少量动作即可",
        },
        penalty_factor=0.80,
        penalty_threshold=3.0,
    ),
    MetricDefinition(
        name=MetricName.TONE,
        label="语气好（共情与礼貌）",
        priority=4,
        weight=0.15,
        definition=(
            "语气是否礼貌、专业，并且在用户带有情绪时（投诉、着急、焦虑、涉及敏感信息）"
            "给出针对性的共情与安抚，而不是只输出冷冰冰的条款。"
        ),
        quantification=(
            "裁判检查两点：①基础礼貌（问候、致歉、感谢）；②对用户情绪的针对性回应。"
            "用户明显带情绪而回复完全没有安抚时显著扣分。"
        ),
        rationale=(
            "语气对情绪化场景的满意度影响很大，但它可以通过话术模板快速改善，"
            "业务风险低于前三项，因此权重最低（0.15）且不设一票否决，避免『态度好』掩盖实质问题。"
        ),
        anchors={
            1: "语气冷漠或让用户感觉被敷衍、被指责",
            3: "礼貌但模板化，用户的情绪没有被接住",
            5: "礼貌专业，且对用户的具体情绪做了针对性安抚",
        },
        penalty_factor=None,
    ),
)

_DEFINITION_BY_NAME: dict[MetricName, MetricDefinition] = {
    definition.name: definition for definition in METRIC_DEFINITIONS
}


def all_definitions() -> tuple[MetricDefinition, ...]:
    """按优先级顺序返回全部指标定义。"""

    return tuple(sorted(METRIC_DEFINITIONS, key=lambda item: item.priority))


def get_definition(metric: MetricName) -> MetricDefinition:
    """按名称取指标定义。"""

    return _DEFINITION_BY_NAME[metric]


def weight_of(metric: MetricName) -> float:
    """取指标在总分中的权重。"""

    return _DEFINITION_BY_NAME[metric].weight
