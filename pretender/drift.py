"""Attention drift — MaiBot's ``src/maisaka/attention_drift.py``, ported.

A bot that tracks every topic with equal, unwavering attention reads as
software. A person gets briefly hijacked by the funniest detail in the
backlog, says something about it, and comes back. This module renders that
tendency as a prompt block injected into BOTH the planner and the replyer.

Injecting it into the replyer alone would demote it to a prose-style knob:
the drift decides WHICH pending message gets latched onto, and that is a
stage-one decision.

The block is a 4x3x3 matrix — ``level`` (how far attention wanders) x
``anchor`` (how hard it comes back) x ``reaction`` (how often it opens with a
short interjection) — over four guard clauses that are always present. The
fourth guard is the load-bearing one: without "do not simulate real
inefficiency", drift prompts reliably degrade into "oops, what was I saying"
theatre, which is the opposite of the intended effect.

Rules are module constants rather than a data file, as in MaiBot: the config
picks one cell of the matrix, and the text is not a per-deployment knob.
Chinese only — Pretender has no i18n framework by design (PLAN.md §Locale).
"""

from __future__ import annotations

from pretender.config import DriftConfig

#: How far attention is allowed to wander from the current topic.
DRIFT_LEVEL_RULES: dict[str, str] = {
    "subtle": (
        "漂移档位：轻微漂移。只在最近消息里出现非常自然的触发点时，轻轻联想一句；大多数时候继续当前话题。"
    ),
    "active": (
        "漂移档位：活跃联想。可以主动抓住新鲜、好笑、反差强或熟悉的细节接话，但回复仍要清楚、短促、能被最近消息解释。"
    ),
    "scattered": (
        "漂移档位：明显发散。你可以明显地被支线、关键词、熟人语气或反差点勾走，先接住那个点再回到正题；回复里允许出现一次可理解的突然拐弯。"
    ),
    "wild": (
        "漂移档位：强烈跳跃。你可以先被最有趣的细节劫走一下，出现短促插话、突然联想或半路拐弯；但每轮最多一次明显跳跃，不能无视明确提问，最后要让人看得出你在接哪条消息。"
    ),
}


#: How firmly the reply must return to what was being discussed.
ANCHOR_POLICY_RULES: dict[str, str] = {
    "strict": (
        "回钩策略：严格回钩。联想或短反应之后，要立刻回到当前正在聊的主题或被回复对象。"
    ),
    "balanced": (
        "回钩策略：自然回钩。可以短暂沿着支线说一句，但通常要让结尾或主要意思回到当前聊天。"
    ),
    "loose": (
        "回钩策略：宽松关联。可以保留更自由的相关联想，但不能凭空换话题，也不能无视明确提问。"
    ),
}


#: How readily a reply opens with a short interjection.
REACTION_STYLE_RULES: dict[str, str] = {
    "reserved": (
        "短反应风格：少量短反应。只有特别适合接话时，才用一句很短的反应开头。"
    ),
    "natural": (
        "短反应风格：自然短反应。可以偶尔先用短句、吐槽或语气词接住话题，再继续正常回复。"
    ),
    "lively": (
        "短反应风格：活泼短反应。更容易先用短促反应开头，但不要把回复拆得太碎，也不要每次都这样。"
    ),
}


#: The four guards, always emitted. Every drift needs a traceable hook in the
#: recent messages; drift is not topic-switching; liveliness is not
#: inefficiency; and the style is never named or medicalised.
_PREAMBLE = (
    "注意力漂移风格：\n"
    "- 你可以短暂被聊天里新鲜、好笑、反差强或熟悉的人和梗吸引，"
    "但每次漂移都必须能从最近消息里找到明确触发点。\n"
    "- 漂移不是单纯换话题，而是先抓一个突出的触发点，"
    "再用很短的拐弯制造\u201c脑子突然亮了一下\u201d的感觉。\n"
    "- 表现活跃联想，不要真的降效；不要为了显得分心而故意拖延、忽略任务或打散工具调用。\n"
    "- 不要医学化描述这种风格，不要自称 ADHD，也不要主动声明自己分心。\n"
)


def build_drift_block(config: DriftConfig | None) -> str:
    """The drift prompt block for ``config``, or ``""`` when it is unusable.

    An unknown level/anchor/reaction degrades to an empty block rather than
    raising: a typo in one config key must not silence the bot.
    """
    if config is None:
        return ""
    level = DRIFT_LEVEL_RULES.get(config.level)
    anchor = ANCHOR_POLICY_RULES.get(config.anchor)
    reaction = REACTION_STYLE_RULES.get(config.reaction)
    if not (level and anchor and reaction):
        return ""
    return f"{_PREAMBLE}- {level}\n- {anchor}\n- {reaction}\n"
