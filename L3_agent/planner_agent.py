"""L3 规划智能体模块（PlannerAgent）。

负责调本地 Ollama（OpenAI 兼容入口）/ 其它 OpenAI 兼容服务，
用 ``instructor.from_openai(...)`` + Pydantic 强制模型返回结构化 JSON，
再反序列化为 :class:`TaskPlan` 对象。

采用 **agent 范式**：维护一份 :class:`AgentState`，最多调用
``MAX_PLANNING_STEPS`` 次大模型，每次循环由 LLM 决定是"调一个工具"
还是"定稿"，循环结束后返回结构化计划。核心入口：

- :meth:`PlannerAgent.plan` —— 任务识别 + 初步规划
  （先验：飞行节点想做什么；输出：候选任务集合 + 所需 L3 工具）

结构化输出由 :class:`TaskPlan` / :class:`TaskPlanItem` 约束，
字段枚举（任务大类、工具名）经过 ``Literal`` 二次校验。

调用约定：

    >>> p = PlannerAgent()
    >>> p.plan(snapshot)
    {'status': 'ok', 'plan': {...}, 'text': '...', 'model': 'qwen2:7b'}

错误一律返回 ``status='error'``，**不抛异常**，由 Agent 按状态分流。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from tasks import TASK_REGISTRY

from .tools import TOOL_REGISTRY


__all__ = [
    "PlannerAgent",
    "TaskPlan",
    "TaskPlanItem",
    "ToolInvocation",
    "PlanStep",
    "Observation",
    "AgentState",
]


# Ollama / LM Studio 等本地 OpenAI 兼容服务不校验 api_key，传一个占位即可。
OPENAI_PLACEHOLDER_API_KEY = "ollama"


# ---------------------------------------------------------------------- 响应模型

# 把 LLM 必须挑的三大类硬编码成 Literal —— 配合 Pydantic 校验，
# 模型一旦输出别的字符串，会被 instructor 拒掉并触发重试/抛错。
TaskTypeLiteral = Literal[
    "routing_optimization",
    "topology_control",
    "communication_resource_management",
]

# L3 层对 LLM 暴露的"三类工具"枚举（与 ``L2_cognition.Cognition`` 的三个能力一一对应）：
# - ExperienceRetriever：调 ``Cognition.retrieve_experience(topk)`` 查短期记忆 top-k 经验（读）
# - KnowledgeGraph：调 ``Cognition.query_constraints()`` 查知识图谱的实体 / 关系 / 约束（读）
# - Memory：调 ``Memory.long_term()`` 获取长期记忆里历史上发生过的重要事件（**读**，不是写入）
# 三者都是"只读"工具 —— L3 不通过 LLM 直接写状态，写入由工作流 / Cognition 内部完成。
# 不要把 SubCapability（如 ``spectrum_allocation``）塞进来 —— 它们是任务内部的子能力，不是工具。
ToolTypeLiteral = Literal[
    "ExperienceRetriever",
    "KnowledgeGraph",
    "Memory",
]


class TaskPlanItem(BaseModel):
    """单条任务识别结果（定稿产物，只保留核心三字段）。

    字段语义与 LLM 输出 JSON 中的 ``tasks[i]`` 一一对应：

    - ``id`` / ``type`` / ``priority`` —— 描述"要做什么任务、优先级多高"。

    工具在 agent loop 循环期间已按需调用（见 :class:`ToolInvocation`），
    调用轨迹保留在 ``AgentState.observations`` 里；定稿只描述任务本身，
    不再重复声明用过哪些工具或检索参数。
    """

    id: str = Field(..., description="任务 id，如 'task_1'")
    type: TaskTypeLiteral = Field(..., description="必须取自三大枚举之一")
    priority: int = Field(..., description="1 = 最高优先级，越大越低")


class TaskPlan(BaseModel):
    """``PlannerAgent.plan`` 的整体响应模型。

    如果当前不需要任何任务，``tasks`` 为空列表。
    """

    tasks: List[TaskPlanItem] = Field(
        default_factory=list,
        description="识别出的任务列表；空列表表示无任务",
    )


class ToolInvocation(BaseModel):
    """Agent loop 中 LLM 决定"调一个工具"时的单步决策。"""

    tool: ToolTypeLiteral = Field(..., description="要调的工具名")
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="工具参数；按各工具类 ``run(...)`` 签名自由传",
    )


class PlanStep(BaseModel):
    """Agent loop 中**每一步** LLM 的输出 schema。

    LLM 在每次循环里决定下一步动作（互斥三选一）：

    1. **调工具**：填 ``tool_invocation``，不填 ``final_plan``
    2. **定稿**：填 ``final_plan``，不填 ``tool_invocation``
    3. 都为空（异常）→ 视为本轮失败，继续下一轮

    每次必填 :attr:`reasoning` 给一段中文决策说明，便于事后审计。
    """

    reasoning: str = Field(..., description="本步推理；中文一句话说明为什么这么做")
    tool_invocation: Optional[ToolInvocation] = Field(
        default=None,
        description="本步要调的工具；调工具时填",
    )
    final_plan: Optional[TaskPlan] = Field(
        default=None,
        description="本步给出的最终任务计划；定稿时填",
    )


class Observation(BaseModel):
    """Agent loop 中一次工具调用的"结果快照"。"""

    iteration: int = Field(..., description="对应的循环计数；与 AgentState.iteration 一致")
    tool: str = Field(..., description="被调用的工具名")
    args: Dict[str, Any] = Field(default_factory=dict, description="传给工具的参数")
    result: Any = Field(default=None, description="工具返回值，JSON 序列化即可")


class AgentState(BaseModel):
    """Agent loop 的运行期状态。

    字段语义：

    - :attr:`perception`：L1 感知快照，进入循环时定下，循环期间不变。
    - :attr:`thoughts`：每轮 LLM 的 ``reasoning`` 字符串累积（语言痕迹）。
    - :attr:`observations`：每轮工具调用的执行结果（结构化）。
    - :attr:`task_plan`：LLM 定稿时填入的 :class:`TaskPlan`；收敛前一直为 ``None``。
    - :attr:`iteration`：当前已执行过的循环计数；进入循环时 0，每跑完一轮 ``+1``。

    每一轮 loop 必须：1) 推一个 ``reasoning`` 进 ``thoughts``；
                       2) 如有工具调用，推一个 :class:`Observation` 进 ``observations``；
                       3) LLM 定稿时填 :attr:`task_plan`，并 break 出去。
    """

    perception: Dict[str, Any] = Field(..., description="L1 感知快照，循环期间不变")
    thoughts: List[str] = Field(default_factory=list, description="每轮 LLM reasoning 累积")
    observations: List[Observation] = Field(
        default_factory=list,
        description="工具调用的执行结果快照，按 step 累积",
    )
    task_plan: Optional[TaskPlan] = Field(
        default=None,
        description="LLM 定稿时填充；循环结束前一直为 None",
    )
    iteration: int = Field(default=0, description="当前已执行的循环计数")


# ---------------------------------------------------------------------- 提示词构造

def _build_planning_step_prompt(state: AgentState) -> str:
    """动态构建"agent loop 单步"的系统约束。

    本函数是 agent loop 的**核心叙事函数**——每次循环都基于当前
    :class:`AgentState`（含 perception / thoughts / observations /
    task_plan / iteration）拼一条新的 prompt，让 LLM 看到到目前为止的
    完整上下文，再决定下一步动作。

    数据来源：

    - 任务大类块：读 :data:`tasks.TASK_REGISTRY`
    - L3 工具块：读 :data:`L3_agent.tools.TOOL_REGISTRY`
    - 历史 / 状态块：来自 ``state.observations`` 与 ``state.thoughts``
    """
    if state.observations:
        obs_lines = []
        for o in state.observations:
            obs_lines.append(
                f"  - [step {o.iteration}] tool={o.tool}, "
                f"args={json.dumps(o.args, ensure_ascii=False)}, "
                f"result={json.dumps(o.result, ensure_ascii=False, default=str)[:600]}"
            )
        obs_block = "\n".join(obs_lines)
    else:
        obs_block = "  （尚无 —— 这是第 1 步）"

    return (
        "你是一名 FANET（Flying Ad-Hoc Network）任务识别与初步规划助手。"
        "本任务采用 **agent 范式**：每次循环你会基于最新 "
        "``AgentState`` 决定下一步动作，循环上限 "
        f"{'MAX_PLANNING_STEPS' }（见 PlannerAgent 类）；每轮结束后状态会被系统更新。\n\n"

        "【每次循环你会输出以下三选一】\n"
        "1. **调用工具**：填 ``tool_invocation = {tool, args}``，把 ``final_plan`` 留空。\n"
        "   系统会执行该工具，并把结果作为新一条 ``Observation`` 推入 state。\n"
        "   args 自由传（如 Memory 可传 time_window / scene_tags 等），按工具需要填。\n"
        "2. **定稿**：填 ``final_plan = {tasks: [...]}``，把 ``tool_invocation`` 留空。\n"
        "   每条任务只需 ``{id, type, priority}`` 三个字段；\n"
        "   系统会设置 ``state.task_plan = final_plan`` 并退出循环。\n"
        "3. 都为空（异常）→ 系统会再问一次，``reasoning`` 仍要写，但 step 算无效。\n\n"

        "每次必填 ``reasoning``：用中文一句话说清这一步为什么这么做（便于事后审计）。\n\n"

        f"{_build_task_types_block()}"
        f"{_build_l3_tools_block()}"

        f"【前 {len(state.observations)} 轮的 observations（state.observations）】\n"
        f"{obs_block}\n\n"

        "【约束】\n"
        "- 候选任务必须从上面【可用任务大类】里挑，不要新增类别。\n"
        "- 子能力由系统根据所选大类自动补全，你不需要单独列出。\n"
        "- 需要外部信息时，用 ``tool_invocation`` 调上面【L3 层可用工具】里的工具，不要写 sub_capability 名。\n"
        "- 不是每步都必须调工具；信息够了就直接 ``finalize``。\n"
        "- 定稿的每条任务只输出 ``{id, type, priority}``，不要再输出用过哪些工具或检索参数。\n"
        "- 如果已收集到足够信息就尽早 ``finalize``，不要把所有工具都跑一遍。\n"
        "- 如果接近 loop 上限仍不能定稿，直接给一个你认为最合理的 ``final_plan``。"
    )


def _build_task_types_block() -> str:
    """从 :data:`tasks.TASK_REGISTRY` 拼装"可用任务大类"段落。

    返回形如：

        【可用任务大类（TaskType，必须从下列 N 类中挑选）】
          - routing_optimization (ROUTING_OPTIMIZATION)
            描述: ...
            子能力: ...
    """
    type_lines = []
    for spec in TASK_REGISTRY.values():
        sub_caps = ", ".join(c.value for c in spec.sub_capabilities)
        type_lines.append(
            f"  - {spec.type.value} ({spec.type.name})\n"
            f"    描述: {spec.description}\n"
            f"    子能力: {sub_caps}"
        )
    types_block = "\n".join(type_lines)
    return (
        f"【可用任务大类（TaskType，必须从下列 {len(TASK_REGISTRY)} 类中挑选）】\n"
        f"{types_block}\n\n"
    )


def _build_l3_tools_block() -> str:
    """从 :data:`L3_agent.tools.TOOL_REGISTRY` 拼装"L3 层可用工具"段落。

    返回形如：

        【L3 层可用工具（tool_invocation.tool 只能从下列 N 类中挑选）】
          - ExperienceRetriever：经验检索，...
          - KnowledgeGraph：知识图谱，...
          - Memory：记忆，...

    新增工具只用在 :data:`TOOL_REGISTRY` 里加一条 :class:`ToolSpec`，
    本函数会自动反映（含标题里的数量）—— 不需要再回 ``planner_agent.py``
    改字符串字面量。
    """
    tool_lines = [
        f"  - {spec.name}：{spec.description}"
        for spec in TOOL_REGISTRY.values()
    ]
    return (
        f"【L3 层可用工具（tool_invocation.tool 只能从下列 {len(TOOL_REGISTRY)} 类中挑选）】\n"
        + "\n".join(tool_lines)
        + "\n\n"
    )


class PlannerAgent:
    """调用本地 OpenAI 兼容端点（默认 Ollama）的规划智能体客户端。

    当前接口 :meth:`plan` 实现 **agent 范式**：
    维护一份 :class:`AgentState`，最多调用 ``MAX_PLANNING_STEPS`` 次大模型，
    每次循环由 LLM 决定是"调一个工具"还是"定稿"，循环结束后返回结构化计划。
    """

    # 默认指向本地 Ollama（OpenAI 兼容入口）；改成官方 OpenAI / LM Studio /
    # vLLM 等时只换 url / model 即可，SDK 会自动适配。
    # 注意：base_url 必须显式带上 ``/v1`` 前缀，SDK 不会再自动追加。
    DEFAULT_URL: str = "http://localhost:11434/v1"   # Ollama OpenAI 兼容入口
    DEFAULT_MODEL: str = "qwen2.5:3b-instruct"       # 本地 Ollama 上跑的模型名
    DEFAULT_TIMEOUT: float = 30.0
    # Agent loop 上限 —— 超过就强制把当时状态丢出去（status=error）
    DEFAULT_MAX_PLANNING_STEPS: int = 5

    def __init__(
        self,
        url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        api_key: str = OPENAI_PLACEHOLDER_API_KEY,
        max_planning_steps: int = DEFAULT_MAX_PLANNING_STEPS,
        executor: Optional[Any] = None,
    ) -> None:
        self._url: str = url.rstrip("/")
        self._model: str = model
        self._timeout: float = float(timeout)
        self._api_key: str = api_key
        self._max_planning_steps: int = int(max_planning_steps)
        # OpenAI SDK 客户端：base_url 指向本地 Ollama（OpenAI 兼容入口），
        # 因此实际跑的仍然是本地 qwen2:7b 等 Ollama 模型 —— SDK 只负责 HTTP 传输。
        # api_key 对 Ollama / LM Studio 无校验意义，传占位即可。
        self._client: OpenAI = OpenAI(
            base_url=self._url,
            api_key=self._api_key,
            timeout=self._timeout,
        )

        # 结构化输出客户端：在 self._client 之上套一层 instructor，
        # 用 Pydantic 模型（见模块顶部 ``TaskPlan`` / ``TaskPlanItem``）约束模型输出。
        # - mode=JSON：走 ``response_format={"type": "json_object"}`` 路线，
        #   对 Ollama 这种对 tools/function-calling 支持不完整的本地服务更稳；
        # - schema 由 Pydantic 自动注入到提示词 + 由 instructor 校验 / 反序列化。
        self._structured_client = instructor.from_openai(
            self._client, mode=instructor.Mode.JSON
        )

        # 工具执行器：默认走 ``DefaultToolExecutor`` 调 tools.py 的骨架 ``run()``；
        # 接入 L2 ``Cognition`` 时换成自定义 ``ToolExecutor`` 即可。
        if executor is None:
            from .tools import DefaultToolExecutor  # 避免循环引用
            executor = DefaultToolExecutor()
        self._executor = executor

    # ------------------------------------------------------------------ 配置访问

    def get_url(self) -> str:
        return self._url

    def get_model(self) -> str:
        return self._model

    # ------------------------------------------------------------------ 业务入口：agent 范式入口

    def plan(
        self,
        perception_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """任务识别 + 初步规划（**agent 范式**）。

        维护一份 :class:`AgentState`，最多调用 ``MAX_PLANNING_STEPS`` 次大模型，
        每次循环由 LLM 决定：

        - **调工具** → 在 ``state.observations`` 追加执行结果，state.iteration ``+1``；
        - **定稿** → 把 :class:`TaskPlan` 写到 ``state.task_plan``，break 出循环。

        任何 LLM 异常都被 instructor 重试；连续失败、模型返回空决策、
        或达到 ``MAX_PLANNING_STEPS`` 上限，都会被兜底成 ``status=error``
        并把当前 state 一起返回。

        Returns
        -------
        dict
            ``{"status", "plan"|"error", "text", "model", "state", "iterations"}``
        """
        # 1. 初始化 AgentState
        state = AgentState(perception=perception_snapshot)

        # 2. Agent loop
        last_error: Optional[str] = None
        for _ in range(self._max_planning_steps):
            step = self._ask_planning_step(state)
            # 状态：thoughts 累积 reasoning
            state.thoughts.append(step.reasoning)
            state.iteration += 1

            # 2a. 定稿分支：LLM 直接给出最终计划，退出循环
            if step.final_plan is not None:
                state.task_plan = step.final_plan
                return self._format_result(state, status="ok")

            # 2b. 工具调用分支：执行 → 把 Observation 推进 state
            if step.tool_invocation is not None:
                ti = step.tool_invocation
                try:
                    result = self._executor.execute(
                        ti.tool, ti.args, state.perception,
                    )
                except Exception as exc:  # noqa: BLE001
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                    last_error = f"tool {ti.tool} failed: {exc}"

                state.observations.append(
                    Observation(
                        iteration=state.iteration,
                        tool=ti.tool,
                        args=ti.args,
                        result=result,
                    )
                )
                continue

            # 2c. 都为空：LLM 决策无效，把这条 reasoning 也算上，跳过
            last_error = "model returned neither tool_invocation nor final_plan"

        # 3. 跳出循环 ⇒ 没收敛
        return self._format_result(
            state,
            status="error",
            error=(
                f"agent loop did not converge in {state.iteration} steps"
                + (f"; last_error={last_error}" if last_error else "")
            ),
        )

    def _ask_planning_step(self, state: AgentState) -> PlanStep:
        """单步调用 LLM：把当前 state 拼成 prompt，结构化输出 :class:`PlanStep`。

        与 ``_post_structured_chat`` 不同：本方法把 instructor 反序列化出来的
        原生 :class:`PlanStep` 对象直接返回，**不二次打包**成 dict，
        方便上层循环同步访问 ``step.final_plan`` / ``step.tool_invocation``。
        出错时抛 :class:`RuntimeError` 到外层 try / except。
        """
        user_content = self._build_user_content(
            system_prompt=_build_planning_step_prompt(state),
            perception_snapshot=state.perception,
            task_label=(
                f"agent loop 第 {state.iteration + 1} 步决策"
                f"（已有 {len(state.observations)} 条 observations）"
            ),
        )
        try:
            return self._structured_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": user_content}],
                response_model=PlanStep,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"planner step failed: {exc}") from exc

    def _format_result(
        self,
        state: AgentState,
        status: str,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """把 AgentState + 状态打包成对外 dict（保留旧 contract 的 plan/text 字段）。"""
        result: Dict[str, Any] = {
            "status": status,
            "model": self._model,
            "iterations": state.iteration,
            "state": state.model_dump(mode="json"),
        }
        if status == "ok" and state.task_plan is not None:
            plan_dict = state.task_plan.model_dump()
            result["plan"] = plan_dict
            result["text"] = json.dumps(plan_dict, ensure_ascii=False)
        else:
            result["error"] = error or "unknown error"
        return result

    # ------------------------------------------------------------------ 内部辅助

    def _build_user_content(
        self,
        system_prompt: str,
        perception_snapshot: Dict[str, Any],
        task_label: str,
    ) -> str:
        """拼成一条 user message：任务标签 + 简要 system 约束 + 快照。

        这里走"全部塞进 user.content"的妥协写法 —— 部分 OpenAI 兼容端点
        对 ``role=system`` 处理不一致，统一只走 user 最稳。
        """
        snapshot_json = json.dumps(perception_snapshot, ensure_ascii=False, default=str)
        return (
            f"[任务]\n{task_label}\n\n"
            f"[系统约束]\n{system_prompt}\n\n"
            f"[感知快照]\n{snapshot_json}"
        )

    def _post_structured_chat(
        self,
        user_content: str,
        response_model: type,
    ) -> Dict[str, Any]:
        """通过 instructor 调一次结构化 chat completion。

        走 ``self._structured_client``（``instructor.from_openai(...)``），
        用 ``response_model`` 约束输出的 JSON schema，并自动反序列化为 Pydantic 对象。

        任何连接 / 校验失败 / 解析异常都被兜底为 ``status=error``，不抛。
        """
        try:
            plan = self._structured_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": user_content},
                ],
                response_model=response_model,
            )
            # plan 是 TaskPlan 实例；同时返回 dict（便于上游消费）和 JSON 串（便于打印）。
            plan_dict = plan.model_dump()
            plan_json = json.dumps(plan_dict, ensure_ascii=False)
            return {
                "status": "ok",
                "plan": plan_dict,
                "text": plan_json,
                "model": self._model,
            }
        except Exception as exc:  # noqa: BLE001 - 调用方按 status 处理，不抛
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "model": self._model,
            }
