"""L3 触发器（Trigger）。

当前职责：判断当前感知快照是否需要触发智能体的决策流程。

骨架阶段：``Trigger.should_act`` 直接返回 ``True``，
不做任何条件判断 —— 等上层规则 / LLM 决策准备好后再补阈值逻辑。
"""

from __future__ import annotations

from typing import Any, Dict


__all__ = ["Trigger"]


class Trigger:
    """L3 智能体触发器。"""

    def should_act(self, perception_snapshot: Dict[str, Any]) -> bool:
        """是否触发本轮决策。

        Parameters
        ----------
        perception_snapshot : dict
            L1 ``Perception.get()`` 返回的完整感知快照。

        Returns
        -------
        bool
            ``True`` 表示触发；``False`` 表示本轮跳过。

        Note
        ----
        骨架阶段：暂时直接返回 ``True``，后续会替换为阈值 / 事件驱动逻辑。
        """
        # TODO: 替换为基于 battery / connectivity / mission_phase 等条件的真触发逻辑
        return True
