"""Mission State model: high-level task assigned to a UAV / swarm.

This is the most decision-critical state for a mission-aware agent.
It describes *what* the swarm is doing, *where* it needs to go, and
*what quality of service* (QoS) the traffic requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .base import StateBase


@dataclass
class MissionState(StateBase):
    """Task / mission descriptor for a UAV.

    Field groups
    ------------
    Identity:   mission_type, priority
    Geometry:   destination (x, y, z), remaining_distance
    Timing:     deadline (seconds remaining)
    Role:       formation_role, current_stage
    QoS:        required_QoS, traffic_type
    Progress:   task_progress (0-1)
    """

    # --- identity ---
    mission_type: str = "Idle"        # 'Search' | 'Track' | 'Relay' | 'Delivery' | ...
    priority: str = "Normal"          # 'Low' | 'Normal' | 'High' | 'Critical'

    # --- geometry ---
    destination: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [x, y, z]
    remaining_distance: float = 0.0   # meters

    # --- timing ---
    deadline: float = 0.0             # seconds remaining

    # --- role / formation ---
    formation_role: str = "None"      # 'Leader' | 'Follower' | 'Relay' | 'Scout' | ...
    current_stage: str = "Idle"       # free-form stage descriptor

    # --- QoS / traffic ---
    required_QoS: str = "BestEffort"  # 'LowDelay' | 'HighReliability' | 'HighBandwidth' | ...
    traffic_type: str = "Data"        # 'Video' | 'Telemetry' | 'Voice' | 'Data' | ...

    # --- progress ---
    task_progress: float = 0.0        # 0-1

    # ------------------------------------------------------------------ helpers

    def is_realtime(self) -> bool:
        return self.required_QoS in ("LowDelay", "HighReliability") \
               or self.traffic_type in ("Video", "Voice")

    def is_critical(self) -> bool:
        return self.priority in ("High", "Critical")

    def is_overdue(self, current_time: float, start_time: float = 0.0) -> bool:
        elapsed = current_time - start_time
        return elapsed > self.deadline
