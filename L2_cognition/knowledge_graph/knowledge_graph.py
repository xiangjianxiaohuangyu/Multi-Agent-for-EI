"""L2 认知层 - 知识图谱（Knowledge Graph）。

以"实体 + 关系"的形式组织 FANET 领域的概念网络，
支持基于关系的多跳推理。

骨架阶段：用 dict 存图，提供基本的 add_edge / query_neighbors 接口，
后续可替换为 networkx / neo4j / 专用图数据库。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

__all__ = ["KnowledgeGraph"]


class KnowledgeGraph:
    """简单的有向知识图谱。"""

    def __init__(self) -> None:
        # adjacency: {src: {dst: [relation, ...]}}
        self._adjacency: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    # ------------------------------------------------------------------ 写入

    def add_entity(self, entity: str) -> None:
        """注册一个实体（不存在则创建空邻接表）。"""
        self._adjacency.setdefault(entity, defaultdict(list))

    def add_relation(self, src: str, dst: str, relation: str) -> None:
        """添加一条有向关系 src -[relation]-> dst。"""
        self.add_entity(src)
        self.add_entity(dst)
        self._adjacency[src][dst].append(relation)

    # ------------------------------------------------------------------ 查询

    def neighbors(self, entity: str) -> List[Tuple[str, str]]:
        """返回 (邻居, 关系) 列表。"""
        if entity not in self._adjacency:
            return []
        result: List[Tuple[str, str]] = []
        for dst, rels in self._adjacency[entity].items():
            for r in rels:
                result.append((dst, r))
        return result

    def entities(self) -> Iterable[str]:
        """返回所有已知实体。"""
        return self._adjacency.keys()

    def relations_of(self, src: str, dst: str) -> List[str]:
        """返回 src -> dst 的所有关系。"""
        return list(self._adjacency.get(src, {}).get(dst, []))

    # ------------------------------------------------------------------ 推理

    def bfs(self, start: str, max_depth: int = 2) -> Dict[str, int]:
        """从 start 出发做 BFS，返回 {entity: depth}。"""
        depths: Dict[str, int] = {start: 0}
        frontier: Set[str] = {start}
        for _ in range(max_depth):
            next_frontier: Set[str] = set()
            for node in frontier:
                for dst, _ in self.neighbors(node):
                    if dst not in depths:
                        depths[dst] = depths[node] + 1
                        next_frontier.add(dst)
            frontier = next_frontier
            if not frontier:
                break
        return depths