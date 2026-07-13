"""L0 environment: 随机数据生成器，为 L1 感知层提供模拟输入。

该模块只负责"造随机数 / 造随机状态对象"——为 FANET 仿真场景生成
合理的 MissionState / NetworkState / SelfState / EnvironmentState /
NeighborState / RoutingState。不负责把它们写入 L1 的 Perception
容器，灌数据由调用方（例如 core / workflow）自行调度。

典型用法：

    from L0_environment.environment import EnvironmentSimulator
    from L1_perception.perception import Perception

    sim = EnvironmentSimulator(seed=42)
    perception = Perception()
    perception.get_cloud().update_mission_state(sim.random_mission_state())
    perception.get_onboard().update_self_state(sim.random_self_state())
    ...
"""

from __future__ import annotations

import random
from typing import List, Optional

from L1_perception.perception import Perception
from models import (
    Battery,
    EnvironmentState,
    MissionState,
    NeighborInfo,
    NeighborState,
    NeighborSummary,
    NetworkState,
    Position,
    RoutingState,
    SelfState,
    Velocity,
)

__all__ = ["EnvironmentSimulator"]


class EnvironmentSimulator:
    """生成随机状态对象的工厂。

    Parameters
    ----------
    seed : int, optional
        随机种子，便于复现实验。
    node_id : int, default 1
        当前 UAV 的节点 ID，会写入 SelfState / NeighborInfo。
    """

    # 离散枚举值，方便制造合理分布
    _MISSION_TYPES = ["Search", "Track", "Relay", "Delivery"]
    _PRIORITIES = ["Low", "Normal", "High", "Critical"]
    _FORMATIONS = ["Leader", "Follower", "Relay", "Scout"]
    _QOS = ["BestEffort", "LowDelay", "HighReliability", "HighBandwidth"]
    _TRAFFIC = ["Data", "Video", "Telemetry", "Voice"]
    _TERRAINS = ["open", "urban", "forest", "mountain"]
    _PROTOCOLS = ["GPSR", "AODV", "OLSR", "B.A.T.M.A.N."]

    def __init__(self, seed: Optional[int] = None, node_id: int = 1) -> None:
        self._rng = random.Random(seed)
        self._node_id = node_id

    # ------------------------------------------------------------------ 写入

    def push_data(self, perception: Perception) -> None:
        """把生成的随机状态一次性灌入 Perception 的所有子容器。

        委托给 Perception.update()，避免在外部了解其内部子容器结构。
        """
        perception.update(
            mission_state=self.random_mission_state(),
            network_state=self.random_network_state(),
            self_state=self.random_self_state(),
            environment_state=self.random_environment_state(),
            neighbor_state=self.random_neighbor_state(),
            routing_state=self.random_routing_state(),
        )

    # ------------------------------------------------------------------ 基础随机数

    def rand_float(self, lo: float, hi: float) -> float:
        """返回 [lo, hi) 之间的随机浮点数。"""
        return self._rng.uniform(lo, hi)

    def rand_int(self, lo: int, hi: int) -> int:
        """返回 [lo, hi] 之间的随机整数。"""
        return self._rng.randint(lo, hi)

    def rand_choice(self, options) -> object:
        """从可迭代对象里随机挑一个元素。"""
        return self._rng.choice(list(options))

    def rand_bool(self, prob_true: float = 0.5) -> bool:
        """以 prob_true 的概率返回 True。"""
        return self._rng.random() < prob_true

    # ------------------------------------------------------------------ 状态对象

    def random_mission_state(self) -> MissionState:
        """随机生成一个 MissionState。"""
        return MissionState(
            mission_type=self.rand_choice(self._MISSION_TYPES),
            priority=self.rand_choice(self._PRIORITIES),
            destination=[self.rand_float(0, 1000) for _ in range(3)],
            remaining_distance=self.rand_float(0, 2000),
            deadline=self.rand_float(10, 300),
            formation_role=self.rand_choice(self._FORMATIONS),
            current_stage="Cruise",
            required_QoS=self.rand_choice(self._QOS),
            traffic_type=self.rand_choice(self._TRAFFIC),
            task_progress=self.rand_float(0, 1),
        )

    def random_network_state(self) -> NetworkState:
        """随机生成一个 NetworkState。"""
        return NetworkState(
            PDR=self.rand_float(0.7, 0.99),
            average_delay=self.rand_float(5, 150),
            throughput=self.rand_float(1, 50),
            routing_overhead=self.rand_float(0.05, 0.3),
            network_density=self.rand_float(0.2, 0.8),
            connectivity=self.rand_float(0.7, 1.0),
            collision_rate=self.rand_float(0.0, 0.1),
            channel_busy_ratio=self.rand_float(0.1, 0.7),
            active_flows=self.rand_int(0, 20),
            hop_count_mean=self.rand_float(1, 5),
            packet_loss_rate=self.rand_float(0.0, 0.1),
        )

    def random_self_state(self) -> SelfState:
        """随机生成一个 SelfState。"""
        return SelfState(
            node_id=self._node_id,
            position=Position(
                x=self.rand_float(-500, 500),
                y=self.rand_float(-500, 500),
                z=self.rand_float(20, 120),
            ),
            velocity=Velocity(
                vx=self.rand_float(-15, 15),
                vy=self.rand_float(-15, 15),
                vz=self.rand_float(-2, 2),
            ),
            speed=self.rand_float(5, 20),
            heading=self.rand_float(0, 360),
            battery=Battery(
                remaining=self.rand_float(20, 100),
                voltage=self.rand_float(20.0, 25.2),
            ),
            cpu_usage=self.rand_float(5, 90),
            memory_usage=self.rand_float(10, 80),
            queue_length=self.rand_int(0, 40),
            current_load=self.rand_float(0.0, 1.0),
            neighbor_num=self.rand_int(0, 8),
            communication_range=250.0,
            packet_send_rate=self.rand_int(0, 50),
            packet_receive_rate=self.rand_int(0, 50),
            current_route_hops=self.rand_int(1, 5),
        )

    def random_environment_state(self) -> EnvironmentState:
        """随机生成一个 EnvironmentState。"""
        return EnvironmentState(
            wind_speed=self.rand_float(0, 15),
            wind_direction=self.rand_float(0, 360),
            temperature=self.rand_float(-10, 35),
            humidity=self.rand_float(20, 95),
            visibility=self.rand_float(500, 10000),
            rain=self.rand_bool(0.2),
            obstacle_density=self.rand_float(0.0, 0.6),
            terrain_type=self.rand_choice(self._TERRAINS),
            gps_quality=self.rand_float(0.7, 1.0),
            interference_level=self.rand_float(0.0, 0.5),
        )

    def random_neighbor_state(self) -> NeighborState:
        """随机生成一个 NeighborState（含邻居表与聚合摘要）。"""
        count = self.rand_int(0, 6)
        neighbors: List[NeighborInfo] = [
            NeighborInfo(
                node_id=self._node_id + i + 1,
                distance=self.rand_float(20, 240),
                relative_speed=self.rand_float(0, 10),
                link_quality=self.rand_float(0.3, 1.0),
                RSSI=self.rand_float(-90, -40),
                SINR=self.rand_float(0, 30),
                packet_loss=self.rand_float(0, 0.2),
                queue_length=self.rand_int(0, 30),
                battery=self.rand_float(20, 100),
                LET=self.rand_float(5, 60),
            )
            for i in range(count)
        ]
        summary = NeighborSummary.from_neighbors(neighbors)
        return NeighborState(neighbors=neighbors, neighbor_summary=summary)

    def random_routing_state(self) -> RoutingState:
        """随机生成一个 RoutingState。"""
        return RoutingState(
            protocol=self.rand_choice(self._PROTOCOLS),
            hello_interval=self.rand_choice([0.5, 1.0, 2.0]),
            path_num=self.rand_int(0, 5),
            neighbor_table_size=self.rand_int(0, 8),
            routing_table_size=self.rand_int(0, 10),
            greedy_success_rate=self.rand_float(0.6, 0.99),
            perimeter_usage_rate=self.rand_float(0.0, 0.3),
            route_break_times=self.rand_int(0, 10),
            link_failure_rate=self.rand_float(0.0, 0.15),
            last_update_time=self.rand_float(0, 1000),
        )