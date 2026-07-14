"""L3 层工具定义。

对应 ``planner_agent.ToolTypeLiteral`` 的三类工具：

- :class:`ExperienceRetriever` —— 经验检索（短期记忆 top-k）
- :class:`KnowledgeGraph`     —— 知识图谱（实体 / 关系 / 约束）
- :class:`Memory`             —— 长期记忆（历史上发生过的重要事件，**只读**）

每个工具暂时只暴露一个核心方法 ``run(...)``；**当前阶段不实现真实逻辑**，
函数体里只 ``print`` 一行示意输出，方便上层 (Agent / PlannerAgent) 做接口联调。
后续把 L2 ``Cognition`` 的接口接上即可。

可扩充性（与 :mod:`tasks` 同型）：
- 新增一个工具 = 写一个类（带 ``name`` 类属性）+ 在 :data:`TOOL_REGISTRY`
  加一条 :class:`ToolSpec`。
- ``planner_agent.py`` 的工具提示词块**自动**反映 ``TOOL_REGISTRY`` 内容，
  不需要再回去改 ``planner_agent.py`` 的字符串字面量。

执行器（agent loop 调用工具用的桥）：
- :class:`ToolExecutor` —— 抽象接口（``execute(tool, args, snapshot) -> result``）
- :class:`DefaultToolExecutor` —— 调本文件 ``run(...)`` 骨架实现
- 后续可加 :class:`CognitionToolExecutor` 之类对接 L2 ``Cognition`` 真实记忆
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = [
    "ExperienceRetriever",
    "KnowledgeGraph",
    "Memory",
    "ToolSpec",
    "TOOL_REGISTRY",
    "ToolExecutor",
    "DefaultToolExecutor",
]


# ---------------------------------------------------------------------- 工具实现（骨架）

class ExperienceRetriever:
    """经验检索工具 —— 对应 ``Cognition.retrieve_experience(topk)``。"""

    name: str = "ExperienceRetriever"

    def run(self, topk: int = 5) -> List[Dict[str, Any]]:
        """从短期记忆里取 top-k 条经验。骨架阶段只打印。"""
        print(f"[ExperienceRetriever] run topk={topk}")
        return []


class KnowledgeGraph:
    """知识图谱工具 —— 对应 ``Cognition.query_constraints()``。"""

    name: str = "KnowledgeGraph"

    def run(self) -> Dict[str, Any]:
        """查询当前已知的实体 / 关系 / 约束。骨架阶段只打印。"""
        print("[KnowledgeGraph] run")
        return {"entities": [], "relations": []}


class Memory:
    """记忆工具 —— 对应 ``Memory.long_term()``（**只读**，不是写入）。"""

    name: str = "Memory"

    def run(self) -> List[Dict[str, Any]]:
        """获取长期记忆里历史上发生过的重要事件。骨架阶段只打印。"""
        print("[Memory] run")
        return []


# ---------------------------------------------------------------------- 工具元数据

@dataclass(frozen=True)
class ToolSpec:
    """工具元数据（与 :class:`tasks.spec.TaskSpec` 同型思路），用于：

    - ``planner_agent.py`` 动态构建工具清单提示词
    - 后续 Agent 端按工具名查找对应执行类的扩展点

    字段：

    - ``name``：工具字符串枚举值（与 ``planner_agent.ToolTypeLiteral`` 三个字面量一致）
    - ``description``：注入到提示词里的中文说明（单行，无前导编号）
    """

    name: str
    description: str


# ---------------------------------------------------------------------- 工具注册表

# 维护一份"按工具名 → ToolSpec"的注册表（与 :data:`tasks.TASK_REGISTRY` 同型）。
# 新增工具时只需：1) 在本文件加一个同名类（带 ``name`` 类属性），
#                  2) 在 ``TOOL_REGISTRY`` 里加一条 ``ToolSpec`` 登记。
# ``planner_agent.py`` 的工具清单提示词会自动反映。
TOOL_REGISTRY: Dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            name="ExperienceRetriever",
            description="经验检索，从短期记忆中检索最相关的 top-k 条经验（读）。",
        ),
        ToolSpec(
            name="KnowledgeGraph",
            description="知识图谱，查询网络中实体之间的关系 / 拓扑约束（读）。",
        ),
        ToolSpec(
            name="Memory",
            description="记忆，从长期记忆里获取历史上发生过的重要事件（**只读**，不是写入）。",
        ),
    )
}


# ---------------------------------------------------------------------- 工具执行器

class ToolExecutor:
    """工具执行器抽象接口。

    Agent loop 每一步拿到 LLM 的 :class:`planner_agent.ToolInvocation` 之后，
    把 ``tool / args`` 转成实际执行结果。不同的执行器实现可以对接不同后端：

    - :class:`DefaultToolExecutor` —— 调本文件 :meth:`run` 骨架
    - 后续 :class:`CognitionToolExecutor` —— 把 args 转给 :class:`Cognition` 接口

    返回值必须是 JSON 可序列化的 ``dict`` / ``list`` / 标量，让上层能直接 ``json.dumps``
    后塞回历史。
    """

    def execute(
        self,
        tool: str,
        args: Dict[str, Any],
        snapshot: Dict[str, Any],
    ) -> Any:
        """执行一次工具调用，返回结果。"""
        raise NotImplementedError


class DefaultToolExecutor(ToolExecutor):
    """默认执行器 —— 把工具名 dispatch 到本文件三个 ``run()`` 方法。

    骨架阶段所有工具都返回空结构；后续接入 ``Cognition`` 时换实现即可。
    """

    def execute(
        self,
        tool: str,
        args: Dict[str, Any],
        snapshot: Dict[str, Any],  # noqa: ARG002 - 骨架阶段不用
    ) -> Any:
        if tool == "ExperienceRetriever":
            return ExperienceRetriever().run(topk=int(args.get("topk", 5)))
        if tool == "KnowledgeGraph":
            return KnowledgeGraph().run()
        if tool == "Memory":
            return Memory().run()
        return {"error": f"unknown tool: {tool}"}
