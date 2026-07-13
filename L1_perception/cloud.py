"""Cloud-side perception: mission task and network status.

Holds the latest ``MissionState`` and ``NetworkState`` provided by the
ground control / cloud side.  Callers can read the current snapshot or
replace the whole state when a new report arrives.
"""

from __future__ import annotations

import copy
from typing import Optional

from models import MissionState, NetworkState

__all__ = ["CloudPerception"]


class CloudPerception:
    """Container for cloud / ground-control perceived state."""

    def __init__(
        self,
        mission_state: Optional[MissionState] = None,
        network_state: Optional[NetworkState] = None,
    ) -> None:
        self._mission_state: MissionState = copy.deepcopy(MissionState())
        self._network_state: NetworkState = copy.deepcopy(NetworkState())

        if mission_state is not None:
            self.update_mission_state(mission_state)
        if network_state is not None:
            self.update_network_state(network_state)

    # ------------------------------------------------------------------ getters

    def get_mission_state(self) -> MissionState:
        """Return a deep copy of the current mission state."""
        return copy.deepcopy(self._mission_state)

    def get_network_state(self) -> NetworkState:
        """Return a deep copy of the current end-to-end network state."""
        return copy.deepcopy(self._network_state)

    # ------------------------------------------------------------------ updaters

    def update_mission_state(self, state: MissionState) -> None:
        """Replace the current mission state with a deep copy of ``state``."""
        if not isinstance(state, MissionState):
            raise TypeError(
                f"update_mission_state: expected MissionState, "
                f"got {type(state).__name__}"
            )
        self._mission_state = copy.deepcopy(state)

    def update_network_state(self, state: NetworkState) -> None:
        """Replace the current network state with a deep copy of ``state``."""
        if not isinstance(state, NetworkState):
            raise TypeError(
                f"update_network_state: expected NetworkState, "
                f"got {type(state).__name__}"
            )
        self._network_state = copy.deepcopy(state)
