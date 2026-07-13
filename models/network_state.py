"""Network State model: end-to-end FANET status.

In simulation (ns-3, OMNeT++, etc.) these values are typically produced by:
    - MAC layer statistics
    - Routing protocol statistics
    - FlowMonitor
    - Custom monitors (mobility, energy)

In a real deployment they come from periodic control-plane telemetry and
in-band measurements.  Values are usually averaged over a sliding window.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import StateBase


@dataclass
class NetworkState(StateBase):
    """End-to-end network performance / health state.

    Units and ranges
    ----------------
    PDR (Packet Delivery Ratio):           0-1
    average_delay:                         milliseconds
    throughput:                            Mbps
    routing_overhead:                      ratio of control bytes / data bytes
    network_density:                       0-1, fraction of theoretical max neighbors
    connectivity:                          0-1, fraction of nodes in the largest component
    collision_rate:                        0-1
    channel_busy_ratio:                    0-1
    active_flows:                          integer count
    hop_count_mean:                        average hops per delivered packet
    packet_loss_rate:                      0-1
    """

    PDR: float = 0.0
    average_delay: float = 0.0
    throughput: float = 0.0
    routing_overhead: float = 0.0
    network_density: float = 0.0
    connectivity: float = 0.0
    collision_rate: float = 0.0
    channel_busy_ratio: float = 0.0
    active_flows: int = 0
    hop_count_mean: float = 0.0
    packet_loss_rate: float = 0.0

    # ------------------------------------------------------------------ helpers

    def is_congested(self, cbr_threshold: float = 0.7,
                     loss_threshold: float = 0.1) -> bool:
        """Heuristic congestion check used by routing / MAC agents."""
        return (self.channel_busy_ratio > cbr_threshold
                or self.packet_loss_rate > loss_threshold)

    def is_partitioned(self, connectivity_threshold: float = 0.8) -> bool:
        return self.connectivity < connectivity_threshold
