"""FANET state models.

This package defines the typed state objects used by every agent in the
multi-agent system.  All classes inherit from `StateBase` and can be
freely converted to / from plain dicts and JSON.

Layout
------
    base              : StateBase (to_dict / from_dict / to_json helpers)
    self_state        : SelfState + Position / Velocity / Battery
    neighbor_state    : NeighborState, NeighborInfo, NeighborSummary
    network_state     : NetworkState
    environment_state : EnvironmentState
    mission_state     : MissionState
    routing_state     : RoutingState
"""

from .base import StateBase
from .self_state import (
    Battery,
    Position,
    SelfState,
    Velocity,
)
from .neighbor_state import (
    NeighborInfo,
    NeighborState,
    NeighborSummary,
)
from .network_state import NetworkState
from .environment_state import EnvironmentState
from .mission_state import MissionState
from .routing_state import RoutingState

__all__ = [
    # base
    "StateBase",
    # self
    "Position",
    "Velocity",
    "Battery",
    "SelfState",
    # neighbor
    "NeighborInfo",
    "NeighborSummary",
    "NeighborState",
    # network
    "NetworkState",
    # environment
    "EnvironmentState",
    # mission
    "MissionState",
    # routing
    "RoutingState",
]
