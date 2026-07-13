"""Self State model: current drone's own status (kinematics, energy, compute, comm).

This is the most frequently updated state in the FANET control loop and
is consumed by virtually every agent (routing, mission, MAC, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .base import StateBase


@dataclass
class Position(StateBase):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Velocity(StateBase):
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


@dataclass
class Battery(StateBase):
    """Energy state of the UAV (percentage + raw voltage)."""
    remaining: float = 100.0   # percentage 0-100
    voltage: float = 22.2      # volts (e.g. 6S LiPo nominal ~22.2V)


@dataclass
class SelfState(StateBase):
    """Complete self-state of a UAV node.

    Field groups
    ------------
    Identity
        node_id
    Kinematics
        position (x,y,z), velocity (vx,vy,vz), speed, heading
    Energy
        battery
    Compute
        cpu_usage, memory_usage
    Buffer
        queue_length
    Workload
        current_load
    Communication
        neighbor_num, communication_range, packet_send_rate, packet_receive_rate
    Routing
        current_route_hops
    """

    # --- identity ---
    node_id: int = 0

    # --- kinematics ---
    position: Position = field(default_factory=Position)
    velocity: Velocity = field(default_factory=Velocity)
    speed: float = 0.0            # m/s, scalar magnitude
    heading: float = 0.0          # degrees, 0 = +x, 90 = +y

    # --- energy ---
    battery: Battery = field(default_factory=Battery)

    # --- compute ---
    cpu_usage: float = 0.0        # percentage 0-100
    memory_usage: float = 0.0     # percentage 0-100

    # --- buffer ---
    queue_length: int = 0         # packets waiting in interface queue

    # --- workload ---
    current_load: float = 0.0     # normalized 0-1, fraction of capacity in use

    # --- communication ---
    neighbor_num: int = 0
    communication_range: float = 250.0   # meters
    packet_send_rate: int = 0            # pkts/s
    packet_receive_rate: int = 0         # pkts/s

    # --- routing ---
    current_route_hops: int = 0

    # ------------------------------------------------------------------ helpers

    def is_low_battery(self, threshold: float = 20.0) -> bool:
        return self.battery.remaining < threshold

    def is_overloaded(self, cpu_threshold: float = 85.0,
                      queue_threshold: int = 50) -> bool:
        return (self.cpu_usage > cpu_threshold
                or self.queue_length > queue_threshold)

    def kinematic_snapshot(self) -> Dict[str, Any]:
        return {
            "x": self.position.x,
            "y": self.position.y,
            "z": self.position.z,
            "speed": self.speed,
            "heading": self.heading,
        }
