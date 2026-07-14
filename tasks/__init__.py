"""FANET 任务定义包。

按高层任务类型拆分为独立子模块；本 ``__init__`` 负责把它们聚合成
``TASK_REGISTRY``，对外只暴露稳定 API。

模块布局::

    tasks/
    ├── __init__.py                          (本文件：聚合 + 公共 API)
    ├── enums.py                             TaskType / SubCapability 枚举
    ├── spec.py                              TaskSpec dataclass
    ├── routing_optimization.py              路由优化任务（含子能力）
    ├── topology_control.py                  拓扑控制任务（含子能力）
    └── communication_resource_management.py 通信资源管理任务（含子能力）

使用示例::

    from tasks import TaskType, TASK_REGISTRY, get_task_spec
    spec = get_task_spec(TaskType.ROUTING_OPTIMIZATION)
    print(spec.sub_capabilities)
"""

from __future__ import annotations

from typing import Any, Dict, List

from .communication_resource_management import (
    SPEC as COMMUNICATION_RESOURCE_MANAGEMENT_SPEC,
)
from .enums import SubCapability, TaskType
from .routing_optimization import SPEC as ROUTING_OPTIMIZATION_SPEC
from .spec import TaskSpec
from .topology_control import SPEC as TOPOLOGY_CONTROL_SPEC


# ---------------------------------------------------------------------- 任务注册表

TASK_REGISTRY: Dict[TaskType, TaskSpec] = {
    TaskType.ROUTING_OPTIMIZATION: ROUTING_OPTIMIZATION_SPEC,
    TaskType.TOPOLOGY_CONTROL: TOPOLOGY_CONTROL_SPEC,
    TaskType.COMMUNICATION_RESOURCE_MANAGEMENT: COMMUNICATION_RESOURCE_MANAGEMENT_SPEC,
}


def get_task_spec(task_type: TaskType) -> TaskSpec:
    """按枚举值查表，找不到时抛 ``KeyError``。"""
    return TASK_REGISTRY[task_type]


def valid_task_types() -> List[str]:
    """给 prompt 用：所有合法 ``type`` 字符串。"""
    return [t.value for t in TaskType]


def describe_all_tasks() -> List[Dict[str, Any]]:
    """用于 LLM 提示词 / 调试：所有任务的简短描述。"""
    return [
        {
            "type": spec.type.value,
            "description": spec.description,
            "sub_capabilities": [c.value for c in spec.sub_capabilities],
            "default_tools": list(spec.default_tools),
        }
        for spec in TASK_REGISTRY.values()
    ]


__all__ = [
    "TaskType",
    "SubCapability",
    "TaskSpec",
    "TASK_REGISTRY",
    "get_task_spec",
    "valid_task_types",
    "describe_all_tasks",
]