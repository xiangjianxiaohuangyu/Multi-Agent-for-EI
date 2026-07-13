"""L2 认知层 - 记忆管理（Memory）。

负责跨时间步保存 / 检索感知快照与认知结论。
骨架阶段提供：
- 短期记忆：最近 N 条感知快照（用于趋势检测）
- 长期记忆：重要事件列表（用于事后分析）

后续可扩展为摘要压缩 / 重要性评分 / 向量检索。
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

__all__ = ["Memory"]


class Memory:
    """短期 + 长期记忆。"""

    def __init__(self, short_term_capacity: int = 100) -> None:
        self._short_term: Deque[Dict[str, Any]] = deque(maxlen=short_term_capacity)
        self._long_term: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ 写入

    def remember_short(self, snapshot: Dict[str, Any]) -> None:
        """写入一条短期记忆（容量满则自动覆盖最旧的）。"""
        self._short_term.append(snapshot)

    def remember_long(self, event: Dict[str, Any]) -> None:
        """写入一条长期记忆（不会自动淘汰）。"""
        self._long_term.append(event)

    # ------------------------------------------------------------------ 查询

    def recent(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """返回最近 n 条短期记忆，默认全部。"""
        if n is None:
            return list(self._short_term)
        return list(self._short_term)[-n:]

    def long_term(self) -> List[Dict[str, Any]]:
        """返回全部长期记忆。"""
        return list(self._long_term)

    def short_term_size(self) -> int:
        return len(self._short_term)

    def long_term_size(self) -> int:
        return len(self._long_term)

    def clear_short(self) -> None:
        """清空短期记忆。"""
        self._short_term.clear()