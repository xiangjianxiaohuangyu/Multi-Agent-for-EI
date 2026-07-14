"""路由优化任务定义。

高层目标：优化 FANET 节点之间的路由选择，提升端到端可达性与时延。
"""

from __future__ import annotations

from .enums import SubCapability, TaskType
from .spec import TaskSpec


# 子能力列表：路由优化任务对外可执行的细粒度动作单元
SUB_CAPABILITIES = (
    SubCapability.PROTOCOL_SELECTION_AND_SWITCHING,
    SubCapability.ROUTING_PARAMETER_OPTIMIZATION,
    SubCapability.PATH_DECISION_AND_MAINTENANCE,
)


# 任务规格：被 ``tasks/__init__.py`` 收集进 ``TASK_REGISTRY``
SPEC = TaskSpec(
    type=TaskType.ROUTING_OPTIMIZATION,
    description="优化 FANET 节点之间的路由选择，提升端到端可达性与时延",
    sub_capabilities=SUB_CAPABILITIES,
    default_tools=("ExperienceRetriever", "KnowledgeGraph"),
)


__all__ = ["SPEC", "SUB_CAPABILITIES"]