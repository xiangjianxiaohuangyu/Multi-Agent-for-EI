# Multi-Agent-for-EI

FANET（Flying Ad-Hoc Network）多智能体仿真项目骨架，从 L0 数据生成到 L5 学习自下而上分层组织。

## 目录结构

```
Multi-Agent-for-EI/
├── L0_environment/      # 模拟数据源（EnvironmentSimulator）
├── L1_perception/       # 感知层（Cloud / Onboard / Wireless + Perception 聚合器）
├── L2_cognition/        # 认知层
├── L3_agent/            # 智能体层
├── L4_network/          # 网络层
├── L5_learning/         # 学习层
├── core/
│   └── workflow.py      # 主工作流入口：L0 -> L1 -> JSON
├── models/              # 状态数据类（StateBase + 各 State 子类）
└── tests/               # 单元测试
```

## 数据流

```
L0_environment  ──生成随机状态──▶  L1_perception
   (EnvironmentSimulator)              (Perception.update)
                                            │
                                            ▼
                                       Perception.get
                                            │
                                            ▼
                                       JSON 快照
                                            │
                                            ▼
                                       L2 / L3 / ...
```

## 环境要求

- Python ≥ 3.9
- 仅使用标准库（`random`, `json`, `dataclasses`），无第三方依赖

## 运行命令

> 以下命令均假设当前工作目录为项目根目录
> `e:\Project\【FANET_Project】\Multi-Agent-for-EI\`

### 1. 运行主工作流

执行一次完整流程：L0 模拟器生成随机状态 → 灌入 L1 Perception → 输出 JSON 快照到 stdout。

```bash
python core/workflow.py
```

可选参数（修改随机种子，便于复现实验）：

```bash
python -c "from core.workflow import run; run(seed=123)"
```

也可在交互式 REPL / 别的脚本里调用：

```python
from core.workflow import run
json_str = run(seed=42)
```

### 2. 直接调用各层 API

**L0：构造模拟数据**

```python
from L0_environment.environment import EnvironmentSimulator

sim = EnvironmentSimulator(seed=42)
mission = sim.random_mission_state()
network = sim.random_network_state()
# ... random_self_state / random_environment_state / random_neighbor_state / random_routing_state
```

**L1：感知聚合器**

```python
from L1_perception.perception import Perception

perception = Perception()

# 整体灌入（L0 push_data 内部走的就是这条路）
perception.update(
    mission_state=sim.random_mission_state(),
    network_state=sim.random_network_state(),
    self_state=sim.random_self_state(),
    environment_state=sim.random_environment_state(),
    neighbor_state=sim.random_neighbor_state(),
    routing_state=sim.random_routing_state(),
)

# 读快照
snapshot_dict = perception.get()                          # -> dict
snapshot_json = perception.get(as_json=True, indent=2)    # -> str

# 部分更新（只覆盖指定字段，其余保持不变）
perception.update(mission_state=sim.random_mission_state())
```

**L0 便捷封装**

```python
# 如果只想一行灌满 L1，可以用 EnvironmentSimulator.push_data
sim.push_data(perception)
```

### 3. 运行单元测试

```bash
python -m unittest discover -s tests -v
```

或单独跑某个测试文件：

```bash
python -m unittest tests.test_l1_perception -v
```

或在项目根目录直接：

```bash
python tests/test_l1_perception.py
```

## 关于 sys.path

`core/workflow.py` 和 `tests/test_l1_perception.py` 启动时会把项目根目录加入 `sys.path`，因此：

- ✅ 从项目根目录运行任意入口都能 `import L1_perception.* / L0_environment.* / models.*`
- ✅ 不需要在每个子目录放 `__init__.py`
- ✅ 不需要 `pip install -e .`

如果在 IDE / Notebook 中导入失败，把项目根目录加入 Python 解释器的搜索路径即可。

## 复现实验

由于 `EnvironmentSimulator` 使用 `random.Random(seed)`，**相同 seed 一定输出相同 JSON**：

```bash
python core/workflow.py > run1.json
python core/workflow.py > run2.json
diff run1.json run2.json    # 无输出 = 完全一致
```

## 后续扩展

- L2 / L3 / L4 / L5 目前只有占位文件，可以按相同模式（`__init__.py` 风格 + 公共访问器 + `update` / `get` 接口）逐层填入
- `Perception.update()` 支持增量更新——只传非 `None` 字段即可
- `Perception.get()` 既返回 dict 也返回 JSON，便于喂给 LLM agent