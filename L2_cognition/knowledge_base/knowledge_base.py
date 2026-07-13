"""L2 认知层 - 专业知识库（Knowledge Base）。

存储 FANET 领域的"事实 / 规则 / 经验知识"，供 Cognition 调用查询。
典型内容：
- 路由协议特性（GPSR / AODV / OLSR / B.A.T.M.A.N. 的适用条件）
- 任务类型与 QoS 要求的对应关系
- 拥塞 / 分区 / 恶劣天气的判定阈值表
- 应急策略（低电量、链路断裂等）

骨架阶段：以键值表 + 查询方法为主，后续可替换为向量库 / LLM 提示词。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["KnowledgeBase"]


class KnowledgeBase:
    """FANET 专业知识库。"""

    # 任务类型 -> 推荐 QoS
    _TASK_QOS_HINTS: Dict[str, str] = {
        "Search":   "HighReliability",
        "Track":    "LowDelay",
        "Relay":    "HighBandwidth",
        "Delivery": "HighReliability",
    }

    # 路由协议 -> 适用场景描述
    _PROTOCOL_HINTS: Dict[str, str] = {
        "GPSR":       "高移动性 + 位置可知；适合稀疏网络",
        "AODV":       "按需建路；适合低流量 + 偶发连接",
        "OLSR":       "表驱动；适合密集 + 低移动性",
        "B.A.T.M.A.N.":"mesh 友好；适合中等规模",
    }

    def task_qos_hint(self, mission_type: str) -> Optional[str]:
        """根据任务类型返回推荐的 QoS 类别。"""
        return self._TASK_QOS_HINTS.get(mission_type)

    def protocol_hint(self, protocol: str) -> Optional[str]:
        """根据路由协议返回适用场景描述。"""
        return self._PROTOCOL_HINTS.get(protocol)

    def list_protocols(self) -> List[str]:
        """返回知识库中所有已注册的路由协议。"""
        return list(self._PROTOCOL_HINTS.keys())