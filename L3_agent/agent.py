"""L3 智能体层。

定位：决策层。**不负责**把 L1 数据写进 L2 短期记忆——
那是 L2 ``Cognition.process`` 的职责，由调用方（core/workflow）
在 L1 → L3 之间显式执行。本层只借助 L2 提供的"检索 / 查询 / 写长期"
能力做决策。

当前阶段的 ``act`` 主流程：

    0. Trigger                              ────  判断要不要走决策
    1. Reason.recognize_intent_and_plan     ────  让模型识别意图并给候选动作（仅作信息收集）
    # 下一步：Reason.reasoning_and_decision ────  综合推理 + 拍板（暂未接入 act）
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from L2_cognition.cognition import Cognition

from .reason import Reason
from .trigger import Trigger

__all__ = ["Agent"]


class Agent:
    """L3 智能体。"""

    def __init__(
        self,
        cognition: Cognition,
        trigger: Optional[Trigger] = None,
        reason: Optional[Reason] = None,
        reason_url: Optional[str] = None,
        reason_model: Optional[str] = None,
        reason_timeout: Optional[float] = None,
        experience_topk: int = 5,  # noqa: ARG002 - 暂存，等 Retriever 重建后启用
    ) -> None:
        self._cognition = cognition
        self._experience_topk = experience_topk
        self._trigger: Trigger = trigger if trigger is not None else Trigger()
        # 没传完整的 Reason 时，用可选 kwarg 构造一个；都缺就走 Reason 的默认配置
        if reason is not None:
            self._reason: Reason = reason
        else:
            self._reason = Reason(
                url=reason_url or Reason.DEFAULT_URL,
                model=reason_model or Reason.DEFAULT_MODEL,
                timeout=Reason.DEFAULT_TIMEOUT if reason_timeout is None else float(reason_timeout),
            )

    # ------------------------------------------------------------------ 入口

    def act(self, perception_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """基于感知 + 认知做一次决策。

        Parameters
        ----------
        perception_snapshot : dict
            L1 ``Perception.get()`` 返回的完整感知快照。

        Returns
        -------
        dict
            当前阶段直接把 ``Reason.recognize_intent_and_plan`` 的回复原样返回；
            结构为 ``{"status": "ok"|"error", "text"|"error": ..., "model": ...}``。

        Note
        ----
        本方法**不**调用 ``Cognition.process`` —— 落短期记忆
        是 L2 的职责，由调用方在 L1 → L3 之间显式执行。
        """
        # 0. 触发判断：未触发则本轮跳过决策
        if not self._trigger.should_act(perception_snapshot):
            return {
                "action": "skip",
                "reason": "trigger not fired",
                "critical": False,
                "inputs": {},
            }

        # 1. 让模型先做"意图识别 + 候选动作规划"（两个推理入口中的第一个）
        return self._reason.recognize_intent_and_plan(perception_snapshot)
