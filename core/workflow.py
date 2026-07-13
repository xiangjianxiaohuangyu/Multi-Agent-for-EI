"""核心工作流：用 L0 模拟器灌数据到 L1 感知层，并以 JSON 形式输出。

该工作流负责"调度"：
1. 实例化 L0 ``EnvironmentSimulator``，拿到随机数据生成能力；
2. 实例化 L1 ``Perception`` 聚合器；
3. 调用 simulator 的 ``random_*_state`` 方法生成随机状态，
   再用 Perception 各子容器的 ``update_*_state`` 写入；
4. 调用 ``perception.get()`` 拉取完整快照并打印为 JSON。
"""

from __future__ import annotations

import os
import sys

# 允许直接运行本文件（`python core/workflow.py`）：
# 将项目根目录加入 sys.path，以便导入同级包 `L0_environment` / `L1_perception`。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from L0_environment.environment import EnvironmentSimulator  # noqa: E402
from L1_perception.perception import Perception  # noqa: E402


def run(seed: int = 42) -> str:
    """执行一次"L0 生成 -> L1 灌入 -> 输出 JSON"流程。

    Parameters
    ----------
    seed : int, default 42
        传给 L0 模拟器的随机种子，便于复现。
    """
    # 1. 构造 L0 数据源与 L1 感知聚合器
    simulator = EnvironmentSimulator(seed=seed)
    perception = Perception()

    # 2. 模拟实际运行环境下，物理设备传输数据给系统L1
    simulator.push_data(perception)

    # 3. 取完整快照并以 JSON 形式输出
    snapshot_json = perception.get(as_json=True, indent=2)
    print(snapshot_json)
    return snapshot_json


if __name__ == "__main__":
    # 作为脚本直接运行时，执行一次完整流程并正常退出
    run()
    sys.exit(0)