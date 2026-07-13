"""Routing State model: live state of the running routing protocol.

The network-wide `NetworkState` answers *"how is the FANET performing?"*,
but a routing-optimization agent also needs to know *"how is the local
routing protocol behaving?"*.  This module captures that protocol-level
view, e.g. for GPSR / AODV / OLSR / B.A.T.M.A.N.-style protocols.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import StateBase


@dataclass
class RoutingState(StateBase):
    """Local view of the running routing protocol.

    Field groups
    ------------
    Protocol:   protocol name, hello_interval
    Topology:   path_num, neighbor_table_size, routing_table_size
    Behavior:   greedy_success_rate, perimeter_usage_rate
    Failures:   route_break_times, link_failure_rate
    Timing:     last_update_time
    """

    # --- protocol identity ---
    protocol: str = "GPSR"            # 'GPSR' | 'AODV' | 'OLSR' | 'B.A.T.M.A.N.' | ...
    hello_interval: float = 1.0       # seconds between HELLO beacons

    # --- topology caches ---
    path_num: int = 0                 # number of active paths
    neighbor_table_size: int = 0      # entries in neighbor table
    routing_table_size: int = 0       # entries in routing table

    # --- protocol-specific behavior (e.g. GPSR) ---
    greedy_success_rate: float = 0.0  # 0-1, fraction of greedy-mode successes
    perimeter_usage_rate: float = 0.0 # 0-1, fraction of routes using recovery mode

    # --- failure tracking ---
    route_break_times: int = 0        # cumulative count since startup
    link_failure_rate: float = 0.0    # 0-1, link failures per attempt

    # --- timing ---
    last_update_time: float = 0.0     # seconds, sim time of last routing update

    # ------------------------------------------------------------------ helpers

    def is_unstable(self,
                    break_threshold: int = 5,
                    failure_rate_threshold: float = 0.1) -> bool:
        return (self.route_break_times > break_threshold
                or self.link_failure_rate > failure_rate_threshold)

    def is_recovery_heavy(self, perimeter_threshold: float = 0.2) -> bool:
        """True when the protocol is frequently falling back to recovery mode."""
        return self.perimeter_usage_rate > perimeter_threshold
