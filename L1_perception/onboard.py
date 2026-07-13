"""Onboard perception: the node's own state and surrounding environment.

Holds the latest ``SelfState`` and ``EnvironmentState`` produced by the
onboard sensors of the local UAV.  Callers can read the current snapshot
or replace the whole state when new sensor data arrives.
"""

from __future__ import annotations

import copy
from typing import Optional

from models import EnvironmentState, SelfState

__all__ = ["OnboardPerception"]


class OnboardPerception:
    """Container for onboard sensor perceived state."""

    def __init__(
        self,
        self_state: Optional[SelfState] = None,
        environment_state: Optional[EnvironmentState] = None,
    ) -> None:
        self._self_state: SelfState = copy.deepcopy(SelfState())
        self._environment_state: EnvironmentState = copy.deepcopy(EnvironmentState())

        if self_state is not None:
            self.update_self_state(self_state)
        if environment_state is not None:
            self.update_environment_state(environment_state)

    # ------------------------------------------------------------------ getters

    def get_self_state(self) -> SelfState:
        """Return a deep copy of the current self state."""
        return copy.deepcopy(self._self_state)

    def get_environment_state(self) -> EnvironmentState:
        """Return a deep copy of the current environment state."""
        return copy.deepcopy(self._environment_state)

    # ------------------------------------------------------------------ updaters

    def update_self_state(self, state: SelfState) -> None:
        """Replace the current self state with a deep copy of ``state``."""
        if not isinstance(state, SelfState):
            raise TypeError(
                f"update_self_state: expected SelfState, "
                f"got {type(state).__name__}"
            )
        self._self_state = copy.deepcopy(state)

    def update_environment_state(self, state: EnvironmentState) -> None:
        """Replace the current environment state with a deep copy of ``state``."""
        if not isinstance(state, EnvironmentState):
            raise TypeError(
                f"update_environment_state: expected EnvironmentState, "
                f"got {type(state).__name__}"
            )
        self._environment_state = copy.deepcopy(state)
