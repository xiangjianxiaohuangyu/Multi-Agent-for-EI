"""核心工作流：用 L0 模拟器灌数据到 L1 感知层，
经 L2 认知层处理后，由 L3 智能体做出决策。

该工作流负责"调度"：
1. 实例化 L0 ``EnvironmentSimulator``、L1 ``Perception``、L2 ``Cognition``、L3 ``Agent``；
2. L0 → L1 灌入随机状态；
3. ``perception.get()`` 拉快照；
4. ``agent.act(snapshot)`` 触发 L2 记忆/检索/约束 + L3 决策；
5. 打印 L1 快照 + L3 决策结果。
"""

from __future__ import annotations

import json
import os
import sys

# 允许直接运行本文件（`python core/workflow.py`）：
# 将项目根目录加入 sys.path，以便导入同级包。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from L0_environment.environment import EnvironmentSimulator  # noqa: E402
from L1_perception.perception import Perception  # noqa: E402
from L2_cognition.cognition import Cognition  # noqa: E402
from L3_agent.agent import Agent  # noqa: E402


def run(seed: int = 42) -> dict:
    """执行一次"L0 → L1 → L2 → L3"完整流程。

    Parameters
    ----------
    seed : int, default 42
        传给 L0 模拟器的随机种子，便于复现。

    Returns
    -------
    dict
        包含 L1 snapshot 与 L3 decision。
    """
    # 1. 构造 L0 数据源 + L1 感知 + L2 认知 + L3 智能体
    simulator = EnvironmentSimulator(seed=seed)
    perception = Perception()
    cognition = Cognition()
    agent = Agent(cognition=cognition, experience_topk=5)

    # 2. 模拟实际运行环境下，物理设备传输数据给系统L1
    simulator.push_data(perception)

    # 3. 取完整快照并打印
    snapshot = perception.get()
    print("=== L1 perception snapshot ===")
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))

    # 4. L2 认知层：把 L1 数据交给 Cognition.process（落短期记忆）
    print("=== L2 cognition result ===")
    cognition_result = cognition.process(snapshot)
    print(json.dumps(cognition_result, indent=2, ensure_ascii=False))

    # 5. 智能体基于感知 + 认知做决策
    print("=== L3 agent decision ===")
    decision = agent.act(snapshot)
    print(json.dumps(decision, indent=2, ensure_ascii=False))

    return {
        "snapshot": snapshot,
        "decision": decision,
    }


if __name__ == "__main__":
    # 作为脚本直接运行时，执行一次完整流程并正常退出
    run()
    sys.exit(0)