"""Neighbor State model: per-neighbor table + aggregated summary.

A node typically maintains a full Neighbor Table (one entry per peer) that is
updated from periodic HELLO / beacon messages.  However, an LLM agent rarely
needs to inspect every entry, so we also pre-compute aggregate statistics
that are cheap to feed into prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .base import StateBase


@dataclass
class NeighborInfo(StateBase):
    """Per-neighbor entry in the neighbor table."""
    node_id: int = 0
    distance: float = 0.0          # meters
    relative_speed: float = 0.0    # m/s
    link_quality: float = 0.0      # 0-1
    RSSI: float = 0.0              # dBm, typically negative
    SINR: float = 0.0              # dB
    packet_loss: float = 0.0       # 0-1
    queue_length: int = 0
    battery: float = 100.0         # percentage
    LET: float = 0.0               # Link Expiration Time, seconds


@dataclass
class NeighborSummary:
    """Aggregated statistics over the neighbor table.

    These are the fields that most LLM agents should consume, because
    they convey the overall neighborhood shape without overwhelming
    the prompt with raw rows.
    """
    neighbor_num: int = 0
    avg_distance: float = 0.0
    min_distance: float = 0.0
    max_distance: float = 0.0
    avg_relative_speed: float = 0.0
    best_link_quality: float = 0.0
    avg_link_quality: float = 0.0
    avg_queue_length: float = 0.0
    max_queue_length: int = 0
    avg_energy: float = 0.0
    min_energy: float = 0.0
    avg_LET: float = 0.0
    best_forward_progress: float = 0.0

    @classmethod
    def from_neighbors(cls, neighbors: List[NeighborInfo]) -> "NeighborSummary":
        """Build a summary from a list of NeighborInfo entries.

        Handles empty input safely (returns all-zero summary).
        """
        if not neighbors:
            return cls()

        n = len(neighbors)
        distances = [nb.distance for nb in neighbors]
        speeds = [nb.relative_speed for nb in neighbors]
        qualities = [nb.link_quality for nb in neighbors]
        queues = [nb.queue_length for nb in neighbors]
        energies = [nb.battery for nb in neighbors]
        lets = [nb.LET for nb in neighbors]

        return cls(
            neighbor_num=n,
            avg_distance=sum(distances) / n,
            min_distance=min(distances),
            max_distance=max(distances),
            avg_relative_speed=sum(speeds) / n,
            best_link_quality=max(qualities),
            avg_link_quality=sum(qualities) / n,
            avg_queue_length=sum(queues) / n,
            max_queue_length=max(queues),
            avg_energy=sum(energies) / n,
            min_energy=min(energies),
            avg_LET=sum(lets) / n,
            best_forward_progress=0.0,  # filled by routing layer if available
        )


@dataclass
class NeighborState(StateBase):
    """Full neighbor state: raw table + pre-computed summary."""
    neighbors: List[NeighborInfo] = field(default_factory=list)
    neighbor_summary: Optional[NeighborSummary] = None

    def refresh_summary(self) -> None:
        """Recompute neighbor_summary from the current neighbors list."""
        self.neighbor_summary = NeighborSummary.from_neighbors(self.neighbors)

    def get_neighbor(self, node_id: int) -> Optional[NeighborInfo]:
        for nb in self.neighbors:
            if nb.node_id == node_id:
                return nb
        return None
