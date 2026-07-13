"""L2 认知层入口。

定位：薄编排层，**不做复杂推理**，只把 L1 感知数据落进短期记忆，
并通过三个独立接口把"检索经验 / 查询约束 / 写长期经验"的能力
暴露给上层（L3 智能体）调用。

数据流：

    L1 perception.get()  ─▶  Cognition.process(snapshot)
                                  │
                                  └─▶ Memory.remember_short
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
        retrieve_experience(topk)   query_constraints()    save_long_term(event)
        (查短期记忆 top-k 条)        (查知识图谱约束)         (写长期记忆)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .knowledge_base.knowledge_base import KnowledgeBase
from .knowledge_graph.knowledge_graph import KnowledgeGraph
from .memory.memory import Memory

__all__ = ["Cognition"]


class Cognition:
    """L2 认知层入口类。"""

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        memory: Optional[Memory] = None,
    ) -> None:
        self._kb = knowledge_base if knowledge_base is not None else KnowledgeBase()
        self._kg = knowledge_graph if knowledge_graph is not None else KnowledgeGraph()
        self._memory = memory if memory is not None else Memory()

    # ------------------------------------------------------------------ 访问器

    def get_knowledge_base(self) -> KnowledgeBase:
        return self._kb

    def get_knowledge_graph(self) -> KnowledgeGraph:
        return self._kg

    def get_memory(self) -> Memory:
        return self._memory

    # ------------------------------------------------------------------ 入口

    def process(self, perception_snapshot: Dict[str, Any]) -> None:
        """L2 认知处理入口。

        骨架阶段**只做一件事**：把感知快照写入短期记忆。
        其余能力（经验检索 / 约束查询 / 长期经验落盘）由
        下面的独立接口按需调用。
        """
        self._memory.remember_short(perception_snapshot)

    # ------------------------------------------------------------------ 三大能力

    def retrieve_experience(self, topk: int = 5) -> List[Dict[str, Any]]:
        """从短期记忆中检索最近 topk 条经验。

        Parameters
        ----------
        topk : int, default 5
            返回的最大条数；<=0 或超过容量时返回全部。

        Returns
        -------
        list of dict
            最近的 topk 条感知快照（按时间从旧到新）。
        """
        if topk is None or topk <= 0:
            return self._memory.recent()
        return self._memory.recent(topk)

    def query_constraints(self) -> Dict[str, Any]:
        """查询知识图谱中当前已知的约束 / 关系。

        Returns
        -------
        dict
            - entities: 所有已知实体列表
            - relations: 形如 [{"src": ..., "dst": ..., "relation": ...}, ...]
            - bfs_from: 可选 BFS 起点（当前返回最近一次 process 涉及的 mission 实体）
        """
        relations: List[Dict[str, str]] = []
        for src in self._kg.entities():
            for dst, rels in self._kg._adjacency[src].items():  # noqa: SLF001 (内部访问用于导出)
                for r in rels:
                    relations.append({"src": src, "dst": dst, "relation": r})
        return {
            "entities": list(self._kg.entities()),
            "relations": relations,
        }

    def save_long_term(self, event: Dict[str, Any]) -> None:
        """把一条重要事件写入长期记忆（不会自动淘汰）。

        典型用法：上层在检测到异常 / 关键决策时调用此接口
        把事件落盘，事后可由 ``Memory.long_term()`` 检索。
        """
        self._memory.remember_long(event)