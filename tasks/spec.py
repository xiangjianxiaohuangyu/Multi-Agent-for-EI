"""任务规格 dataclass：``TaskSpec``。

每个任务（``TaskType`` 枚举值之一）对应一份 ``TaskSpec``，记录
其内部子能力、默认工具集和一句话描述。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .enums import SubCapability, TaskType


@dataclass(frozen=True)
class TaskSpec:
    """一个高层任务的"内部规格"。

    Attributes
    ----------
    type : TaskType
        任务类型枚举值。
    description : str
        一句话描述，供人 / prompt 阅读。
    sub_capabilities : tuple[SubCapability, ...]
        这个任务具备的子能力列表（子能力是真正执行动作的细粒度单元）。
    default_tools : tuple[str, ...]
        默认推荐工具集；LLM 输出的 ``need_tools`` 缺省时回退到它。
    """

    type: TaskType
    description: str
    sub_capabilities: Tuple[SubCapability, ...]
    default_tools: Tuple[str, ...] = ()


__all__ = ["TaskSpec"]