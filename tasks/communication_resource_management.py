"""通信资源管理任务定义。

高层目标：管理无线资源（信道 / 功率 / 带宽），降低冲突与能耗。
"""

from __future__ import annotations

from .enums import SubCapability, TaskType
from .spec import TaskSpec


# 子能力列表：通信资源管理任务对外可执行的细粒度动作单元
SUB_CAPABILITIES = (
    SubCapability.SPECTRUM_ALLOCATION,
    SubCapability.CHANNEL_SELECTION,
    SubCapability.BANDWIDTH_ALLOCATION,
    SubCapability.TIMESLOT_ADJUSTMENT,
)


# 任务规格：被 ``tasks/__init__.py`` 收集进 ``TASK_REGISTRY``
SPEC = TaskSpec(
    type=TaskType.COMMUNICATION_RESOURCE_MANAGEMENT,
    description="管理无线资源（信道 / 功率 / 带宽），降低冲突与能耗",
    sub_capabilities=SUB_CAPABILITIES,
    default_tools=("ResourceKnowledge",),
)


__all__ = ["SPEC", "SUB_CAPABILITIES"]