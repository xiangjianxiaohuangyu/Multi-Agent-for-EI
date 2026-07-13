"""Wireless perception: neighbor table and routing protocol state.

Holds the latest ``NeighborState`` and ``RoutingState`` produced by the
MAC / routing layer of the local UAV.  The neighbor summary embedded in
``NeighborState`` is preserved as supplied by the producer; this class
does not recompute it on read or update, so routing-layer augmentations
(e.g. ``best_forward_progress``) are kept intact.
"""

from __future__ import annotations

import copy
from typing import Optional

from models import NeighborState, RoutingState

__all__ = ["WirelessPerception"]


class WirelessPerception:
    """Container for wireless / routing perceived state."""

    def __init__(
        self,
        neighbor_state: Optional[NeighborState] = None,
        routing_state: Optional[RoutingState] = None,
    ) -> None:
        self._neighbor_state: NeighborState = copy.deepcopy(NeighborState())
        self._routing_state: RoutingState = copy.deepcopy(RoutingState())

        if neighbor_state is not None:
            self.update_neighbor_state(neighbor_state)
        if routing_state is not None:
            self.update_routing_state(routing_state)

    # ------------------------------------------------------------------ getters

    def get_neighbor_state(self) -> NeighborState:
        """Return a deep copy of the current neighbor state.

        The embedded ``neighbor_summary`` is returned as supplied by the
        producer; no implicit refresh is performed.
        """
        return copy.deepcopy(self._neighbor_state)

    def get_routing_state(self) -> RoutingState:
        """Return a deep copy of the current routing state."""
        return copy.deepcopy(self._routing_state)

    # ------------------------------------------------------------------ updaters

    def update_neighbor_state(self, state: NeighborState) -> None:
        """Replace the current neighbor state with a deep copy of ``state``."""
        if not isinstance(state, NeighborState):
            raise TypeError(
                f"update_neighbor_state: expected NeighborState, "
                f"got {type(state).__name__}"
            )
        self._neighbor_state = copy.deepcopy(state)

    def update_routing_state(self, state: RoutingState) -> None:
        """Replace the current routing state with a deep copy of ``state``."""
        if not isinstance(state, RoutingState):
            raise TypeError(
                f"update_routing_state: expected RoutingState, "
                f"got {type(state).__name__}"
            )
        self._routing_state = copy.deepcopy(state)
