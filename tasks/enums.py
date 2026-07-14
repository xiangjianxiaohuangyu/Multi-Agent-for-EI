"""任务枚举：``TaskType``（高层任务类型）+ ``SubCapability``（内部子能力）。

LLM 输出的 JSON 里只填 ``type`` 字段（``TaskType`` 值），
子能力走 ``TaskSpec.sub_capabilities`` 由本包自动补全。
"""

from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    """FANET 高层任务类型枚举。

    与 LLM 输出 JSON 中 ``type`` 字段一一对应。
    """

    ROUTING_OPTIMIZATION = "routing_optimization"                # 路由优化
    TOPOLOGY_CONTROL = "topology_control"                        # 拓扑控制
    COMMUNICATION_RESOURCE_MANAGEMENT = "communication_resource_management"  # 通信资源管理


class SubCapability(str, Enum):
    """每个 TaskType 内部的"子能力"枚举。

    任务粒度上对外不可见（LLM 不需要输出）；由各任务子模块维护自己的子能力清单。
    """

    # ---- 路由优化 ----
    PROTOCOL_SELECTION_AND_SWITCHING = "protocol_selection_and_switching"
    ROUTING_PARAMETER_OPTIMIZATION = "routing_parameter_optimization"
    PATH_DECISION_AND_MAINTENANCE = "path_decision_and_maintenance"

    # ---- 拓扑控制 ----
    NEIGHBOR_DISCOVERY = "neighbor_discovery"
    LINK_PREDICTION = "link_prediction"
    NETWORK_RECONSTRUCTION = "network_reconstruction"

    # ---- 通信资源管理 ----
    SPECTRUM_ALLOCATION = "spectrum_allocation"
    CHANNEL_SELECTION = "channel_selection"
    BANDWIDTH_ALLOCATION = "bandwidth_allocation"
    TIMESLOT_ADJUSTMENT = "timeslot_adjustment"


__all__ = ["TaskType", "SubCapability"]