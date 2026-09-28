"""离线裁判的启发式评分逻辑。

这组规则把人工评语中反复出现的信号抽象成可解释的特征：

* 主动代办（"我帮您…"）与追问关键信息      -> 有用性加分
* 引导用户自助（"请看详情页""请联系客服"） -> 有用性减分，用户问的是"自己的"具体问题时惩罚更重
* 用户已表达困惑、回复却只是复述流程     -> 准确性与有用性同时减分（答非所问）
* 情绪场景缺少安抚                        -> 语气减分
* 过度承诺、无法核实的具体断言            -> 无幻觉减分

它的作用是让整条流水线在没有 API Key 的环境下也能完整跑通；
它模拟的是"一个严格按评分卡打分的裁判"，不能替代真实模型的语义理解。
"""

from __future__ import annotations

import re

from auto_reply_eval.models import MetricName

PROACTIVE_MARKERS = (
    "现在就帮您",
    "马上帮您",
    "我来帮您",
    "我直接帮",
    "我帮您",
    "帮您处理",
    "帮您查",
    "我为您",
    "我帮",
)

ASK_INFO_MARKERS = (
    "请问您的订单号",
    "请您提供",
    "请提供",
    "请问是哪",
    "告诉我是哪",
    "方便告诉我",
    "请告诉我",
    "告诉我",
    "请问您的",
)

DEFLECTION_MARKERS = (
    "商品详情页",
    "物流追踪页面",
    "关注商品页面",
    "加入购物车",
    "请联系客服",
    "客服协助",
    "联系客服",
    "请联系",
    "请咨询",
    "自己去",
    "自行",
    "详情页",
)

EMPATHY_MARKERS = (
    "非常抱歉",
    "抱歉",
    "对不起",
    "感谢您的",
    "感谢您",
    "给您带来",
)

INTERNAL_MARKERS = (
    "加强客服团队的培训",
    "反馈给产品团队",
    "转达给产品团队",
    "反馈给品控部门",
    "产品团队",
)

OVERPROMISE_MARKERS = (
    "额外的补偿优惠券",
    "一定",
    "保证",
    "免费",
)

EMOTION_CUES = (
    "投诉",
    "太差",
    "没人理",
    "等了",
    "搞半天",
    "复杂",
    "敏感",
    "害怕",
    "真的吗",
    "没声音",
    "不工作",
    "没更新",
    "取不出来",
    "搞不懂",
    "不知道怎么",
    "坏的",
    "坏了",
)

SPECIFIC_INTENT_MARKERS = (
    "我要买的",
    "我上次",
    "我买",
    "我那个",
    "我收到",
    "我的",
    "买了",
    "那两款",
    "这两款",
    "这个",
)

CONFUSION_MARKERS = (
    "搞不懂",
    "不知道怎么",
    "不会操作",
    "看不懂",
    "太复杂",
    "搞半天",
)

DECISIVE_MARKERS = (
    "属于质量",
    "由我们承担",
    "由买家承担",
    "可以退货退款",
    "可以随身携带",
    "可以申请退货",
    "可以申请取消",
    "可以取消",
    "退款到账时间",
)

PROCESS_STEP_PATTERN = re.compile(r"[1-9][.、]")
NUMERIC_CLAIM_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:个工作日|工作日|个小时|小时|个月|天|年|Wh|mAh|元|%)"
)
PRODUCT_CLAIM_PATTERN = re.compile(r"(采用的是|材质是|容量是|额定容量|参数是)")


def _match(text: str, markers: tuple[str, ...], cap: int) -> list[str]:
    """按"长标记优先"的方式匹配，避免同一个片段被重复计数。"""

    remaining = text
    hits: list[str] = []
    for marker in sorted(markers, key=len, reverse=True):
        if marker in remaining:
            hits.append(marker)
            remaining = remaining.replace(marker, " ")
            if len(hits) >= cap:
                break
    return hits


def _clamp(value: float) -> float:
    return max(1.0, min(5.0, round(value * 2) / 2))


class HeuristicScorer:
    """把启发式规则封装成可复用的评分器。"""

    def score(self, user_question: str, reply_text: str) -> dict[str, object]:
        """返回与真实裁判一致的结构化输出。"""

        features = self._extract_features(user_question, reply_text)
        faithfulness, unsupported = self._faithfulness(reply_text, features)
        accuracy = self._accuracy(features)
        helpfulness = self._helpfulness(features)
        tone = self._tone(reply_text, features)

        return {
            "verdicts": [
                {
                    "metric": MetricName.FAITHFULNESS.value,
                    "score": faithfulness,
                    "reason": self._faithfulness_reason(features, unsupported),
                },
                {
                    "metric": MetricName.ACCURACY.value,
                    "score": accuracy,
                    "reason": self._accuracy_reason(features),
                },
                {
                    "metric": MetricName.HELPFULNESS.value,
                    "score": helpfulness,
                    "reason": self._helpfulness_reason(features),
                },
                {
                    "metric": MetricName.TONE.value,
                    "score": tone,
                    "reason": self._tone_reason(features),
                },
            ],
            "unsupported_claims": unsupported,
            "missing_actions": self._missing_actions(features),
        }

    # ------------------------------------------------------------------ 特征抽取

    def _extract_features(self, user_question: str, reply_text: str) -> dict[str, object]:
        proactive = _match(reply_text, PROACTIVE_MARKERS, 2)
        ask_info = _match(reply_text, ASK_INFO_MARKERS, 1)
        deflection = _match(reply_text, DEFLECTION_MARKERS, 4)
        empathy = _match(reply_text, EMPATHY_MARKERS, 2)
        internal = _match(reply_text, INTERNAL_MARKERS, 1)
        overpromise = _match(reply_text, OVERPROMISE_MARKERS, 2)
        emotion = _match(user_question, EMOTION_CUES, 2)
        decisive = _match(reply_text, DECISIVE_MARKERS, 1)

        specific_intent = any(marker in user_question for marker in SPECIFIC_INTENT_MARKERS)
        has_steps = bool(PROCESS_STEP_PATTERN.search(reply_text))
        user_confused = any(marker in user_question for marker in CONFUSION_MARKERS)
        # 用户说自己"搞不懂流程"，回复却把流程再念一遍：典型答非所问
        confusion_mismatch = user_confused and has_steps and not ask_info

        return {
            "proactive": proactive,
            "ask_info": ask_info,
            "deflection": deflection,
            "empathy": empathy,
            "internal": internal,
            "overpromise": overpromise,
            "emotion": emotion,
            "decisive": decisive,
            "specific_intent": specific_intent,
            "has_steps": has_steps,
            "user_confused": user_confused,
            "confusion_mismatch": confusion_mismatch,
        }

    # ------------------------------------------------------------------ 指标打分

    def _faithfulness(self, reply_text: str, features: dict[str, object]) -> tuple[float, list[str]]:
        unsupported: list[str] = []
        score = 5.0

        for marker in features["overpromise"]:  # type: ignore[union-attr]
            score -= 1.5
            unsupported.append(f"过度承诺：『{marker}』——平台是否必然兑付无法核实")

        product_claims = PRODUCT_CLAIM_PATTERN.findall(reply_text)
        if product_claims and features["specific_intent"]:
            score -= 0.5
            unsupported.append("断言了具体商品的材质/参数，但缺少商品库依据可追溯")

        numeric_claims = NUMERIC_CLAIM_PATTERN.findall(reply_text)
        if numeric_claims:
            score -= min(0.2 * len(numeric_claims), 0.6)
            unsupported.append(
                "包含需人工核实的时效/数值断言："
                + "、".join(dict.fromkeys(numeric_claims))
            )

        return _clamp(score), unsupported

    def _accuracy(self, features: dict[str, object]) -> float:
        score = 5.0
        if features["confusion_mismatch"]:
            # 用户表达的是"我看不懂流程"，回复却复述流程，属于答非所问
            score -= 2.0
        return _clamp(score)

    def _helpfulness(self, features: dict[str, object]) -> float:
        score = 3.0
        score += 0.7 * len(features["proactive"])  # type: ignore[arg-type]
        score += 0.4 * len(features["ask_info"])  # type: ignore[arg-type]

        deflection_count = len(features["deflection"])  # type: ignore[arg-type]
        if features["specific_intent"]:
            # 用户问的是"我的"具体问题，推给自助渠道的代价更高
            score -= 0.4 * deflection_count
        else:
            score -= 0.2 * deflection_count
        if not features["specific_intent"] and deflection_count == 0:
            score += 0.6  # 通用问题回答得干净利落，值得加分
        if features["decisive"]:
            score += 0.5  # 直接给出确定性结论
        if features["has_steps"]:
            score += 0.3
        if features["confusion_mismatch"]:
            score -= 0.8
        return _clamp(score)

    def _tone(self, reply_text: str, features: dict[str, object]) -> float:
        score = 5.0
        if features["emotion"] and not features["empathy"]:
            score -= 2.0
        if features["internal"] and not features["proactive"]:
            score -= 0.5  # 把内部整改当答复，用户并不关心
        return _clamp(score)

    # ------------------------------------------------------------------ 理由生成

    @staticmethod
    def _faithfulness_reason(features: dict[str, object], unsupported: list[str]) -> str:
        if not unsupported:
            return "未发现无依据的事实性断言，也没有超出平台能力的承诺。"
        return "存在需核实的断言：" + "；".join(unsupported) + "。"

    @staticmethod
    def _accuracy_reason(features: dict[str, object]) -> str:
        if features["confusion_mismatch"]:
            return "用户已说明看不懂流程，回复却把流程原样复述，属于答非所问。"
        return "回复方向与用户诉求一致，未发现事实性错误或误导性表述。"

    @staticmethod
    def _helpfulness_reason(features: dict[str, object]) -> str:
        parts: list[str] = []
        if features["proactive"]:
            parts.append("有主动代办动作（" + "、".join(features["proactive"]) + "）")  # type: ignore[arg-type]
        if features["ask_info"]:
            parts.append("追问了关键信息（" + "、".join(features["ask_info"]) + "）")  # type: ignore[arg-type]
        if features["deflection"]:
            parts.append(
                "存在自助引导（" + "、".join(features["deflection"]) + "）"  # type: ignore[arg-type]
            )
        if features["decisive"]:
            parts.append("给出了确定性结论")
        if features["confusion_mismatch"]:
            parts.append("未追问用户卡在哪一步")
        if not parts:
            return "给出了通用信息，但没有任何推进动作。"
        return "；".join(parts) + "。"

    @staticmethod
    def _tone_reason(features: dict[str, object]) -> str:
        if features["emotion"] and not features["empathy"]:
            return "用户带有明显情绪，回复未做任何安抚，仅陈述规则。"
        if features["internal"] and not features["proactive"]:
            return "礼貌到位，但用内部整改事项回应了用户诉求，安抚力度不足。"
        return "语气礼貌专业，情绪场景的回应也基本到位。"

    @staticmethod
    def _missing_actions(features: dict[str, object]) -> list[str]:
        actions: list[str] = []
        if features["deflection"] and features["specific_intent"]:
            actions.append("未主动协助查询用户的具体订单/商品，把操作成本转移给了用户")
        if features["confusion_mismatch"]:
            actions.append("未追问用户具体卡在哪一步，而是重复流程说明")
        if not features["proactive"] and not features["ask_info"]:
            actions.append("未主动代办，也未追问办成事所需的关键信息（订单号等）")
        return actions
