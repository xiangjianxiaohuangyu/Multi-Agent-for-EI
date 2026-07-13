"""Perception: unified snapshot combining cloud, onboard and wireless views.

This module is the single entry point for downstream agents (L2 cognition,
L3 agent, ...).  It owns references to the three sub-perception containers
(`CloudPerception`, `OnboardPerception`, `WirelessPerception`) and exposes
a `get()` method that pulls the latest snapshot from every source and
returns it as a single JSON-serializable structure (dict) or JSON string.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

from models import (
    EnvironmentState,
    MissionState,
    NeighborState,
    NetworkState,
    RoutingState,
    SelfState,
)

from .cloud import CloudPerception
from .onboard import OnboardPerception
from .wireless import WirelessPerception

__all__ = ["Perception"]


class Perception:
    """Aggregator that fuses cloud / onboard / wireless perception snapshots."""

    def __init__(
        self,
        cloud: Optional[CloudPerception] = None,
        onboard: Optional[OnboardPerception] = None,
        wireless: Optional[WirelessPerception] = None,
    ) -> None:
        self._cloud: CloudPerception = cloud if cloud is not None else CloudPerception()
        self._onboard: OnboardPerception = onboard if onboard is not None else OnboardPerception()
        self._wireless: WirelessPerception = wireless if wireless is not None else WirelessPerception()

    # ------------------------------------------------------------------ accessors

    def get_cloud(self) -> CloudPerception:
        return self._cloud

    def get_onboard(self) -> OnboardPerception:
        return self._onboard

    def get_wireless(self) -> WirelessPerception:
        return self._wireless

    # ------------------------------------------------------------------ update

    def update(
        self,
        *,
        mission_state: Optional[MissionState] = None,
        network_state: Optional[NetworkState] = None,
        self_state: Optional[SelfState] = None,
        environment_state: Optional[EnvironmentState] = None,
        neighbor_state: Optional[NeighborState] = None,
        routing_state: Optional[RoutingState] = None,
    ) -> None:
        """更新各子感知容器；只写入非 None 的字段。

        每个参数对应一个子容器的 ``update_*_state`` 调用，
        传 None 的字段保持不变，便于增量更新。
        """
        if mission_state is not None:
            self._cloud.update_mission_state(mission_state)
        if network_state is not None:
            self._cloud.update_network_state(network_state)
        if self_state is not None:
            self._onboard.update_self_state(self_state)
        if environment_state is not None:
            self._onboard.update_environment_state(environment_state)
        if neighbor_state is not None:
            self._wireless.update_neighbor_state(neighbor_state)
        if routing_state is not None:
            self._wireless.update_routing_state(routing_state)

    # ------------------------------------------------------------------ snapshot

    def get_snapshot_dict(self) -> Dict[str, Any]:
        """Collect a fresh deep-copy snapshot from every sub-perception.

        Returns a plain dict (JSON-serializable) ready to be dumped or fed
        into an LLM prompt.
        """
        # 通过公共访问器获取子感知容器，便于将来做 lazy init / mock / 代理
        cloud = self.get_cloud()
        onboard = self.get_onboard()
        wireless = self.get_wireless()
        return {
            "cloud": {
                "mission_state": cloud.get_mission_state().to_dict(),
                "network_state": cloud.get_network_state().to_dict(),
            },
            "onboard": {
                "self_state": onboard.get_self_state().to_dict(),
                "environment_state": onboard.get_environment_state().to_dict(),
            },
            "wireless": {
                "neighbor_state": wireless.get_neighbor_state().to_dict(),
                "routing_state": wireless.get_routing_state().to_dict(),
            },
        }

    def get(self, as_json: bool = False, indent: Optional[int] = None) -> Union[str, Dict[str, Any]]:
        """Return the complete perception snapshot.

        Parameters
        ----------
        as_json : bool, default False
            If True, return a JSON string.  Otherwise return a plain dict.
        indent : int, optional
            Indentation level passed to ``json.dumps`` when ``as_json`` is True.

        Returns
        -------
        str or dict
            The complete perception snapshot across cloud / onboard / wireless.
        """
        snapshot = self.get_snapshot_dict()
        if as_json:
            return json.dumps(snapshot, indent=indent, ensure_ascii=False)
        return snapshot