"""拓扑控制任务定义。

高层目标：维护邻居关系与链路质量，控制节点动态拓扑。
"""

from __future__ import annotations

from .enums import SubCapability, TaskType
from .spec import TaskSpec


# 子能力列表：拓扑控制任务对外可执行的细粒度动作单元
SUB_CAPABILITIES = (
    SubCapability.NEIGHBOR_DISCOVERY,
    SubCapability.LINK_PREDICTION,
    SubCapability.NETWORK_RECONSTRUCTION,
)


# 任务规格：被 ``tasks/__init__.py`` 收集进 ``TASK_REGISTRY``
SPEC = TaskSpec(
    type=TaskType.TOPOLOGY_CONTROL,
    description="维护邻居关系与链路质量，控制节点动态拓扑",
    sub_capabilities=SUB_CAPABILITIES,
    default_tools=("ExperienceRetriever", "KnowledgeGraph"),
)


__all__ = ["SPEC", "SUB_CAPABILITIES"]