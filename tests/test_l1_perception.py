"""Unit tests for the L1 perception state containers."""

from __future__ import annotations

import os
import sys
import unittest

# Make the project importable when running this file directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from L1_perception.cloud import CloudPerception
from L1_perception.onboard import OnboardPerception
from L1_perception.wireless import WirelessPerception
from models import (
    EnvironmentState,
    MissionState,
    NeighborInfo,
    NeighborState,
    NeighborSummary,
    NetworkState,
    RoutingState,
    SelfState,
)


class CloudPerceptionTests(unittest.TestCase):
    def test_default_construction(self) -> None:
        cloud = CloudPerception()
        self.assertIsInstance(cloud.get_mission_state(), MissionState)
        self.assertIsInstance(cloud.get_network_state(), NetworkState)
        self.assertEqual(cloud.get_mission_state().mission_type, "Idle")
        self.assertEqual(cloud.get_network_state().active_flows, 0)

    def test_injected_state_is_readable(self) -> None:
        mission = MissionState(mission_type="Search", priority="Critical")
        network = NetworkState(PDR=0.9, average_delay=20.0, active_flows=3)
        cloud = CloudPerception(mission_state=mission, network_state=network)

        self.assertEqual(cloud.get_mission_state().mission_type, "Search")
        self.assertEqual(cloud.get_mission_state().priority, "Critical")
        self.assertEqual(cloud.get_network_state().PDR, 0.9)
        self.assertEqual(cloud.get_network_state().active_flows, 3)

    def test_update_replaces_state(self) -> None:
        cloud = CloudPerception()
        cloud.update_mission_state(MissionState(mission_type="Relay"))
        cloud.update_network_state(NetworkState(active_flows=7))
        self.assertEqual(cloud.get_mission_state().mission_type, "Relay")
        self.assertEqual(cloud.get_network_state().active_flows, 7)

    def test_type_validation(self) -> None:
        cloud = CloudPerception()
        with self.assertRaises(TypeError):
            cloud.update_mission_state("not a mission state")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cloud.update_network_state(MissionState())  # type: ignore[arg-type]

    def test_input_isolation(self) -> None:
        mission = MissionState(destination=[1.0, 2.0, 3.0])
        cloud = CloudPerception(mission_state=mission)
        mission.destination[0] = 99.0
        self.assertEqual(cloud.get_mission_state().destination, [1.0, 2.0, 3.0])

        new_state = MissionState(destination=[5.0, 6.0, 7.0])
        cloud.update_mission_state(new_state)
        new_state.destination[1] = 88.0
        self.assertEqual(cloud.get_mission_state().destination, [5.0, 6.0, 7.0])

    def test_output_isolation(self) -> None:
        cloud = CloudPerception()
        snapshot = cloud.get_mission_state()
        snapshot.destination[0] = 42.0
        self.assertEqual(cloud.get_mission_state().destination, [0.0, 0.0, 0.0])

    def test_getter_returns_new_instance(self) -> None:
        cloud = CloudPerception()
        self.assertIsNot(cloud.get_mission_state(), cloud.get_mission_state())
        self.assertIsNot(cloud.get_network_state(), cloud.get_network_state())


class OnboardPerceptionTests(unittest.TestCase):
    def test_default_construction(self) -> None:
        onboard = OnboardPerception()
        self.assertIsInstance(onboard.get_self_state(), SelfState)
        self.assertIsInstance(onboard.get_environment_state(), EnvironmentState)
        self.assertEqual(onboard.get_self_state().node_id, 0)
        self.assertEqual(onboard.get_environment_state().terrain_type, "open")

    def test_injected_state_is_readable(self) -> None:
        self_state = SelfState(node_id=4, speed=12.5, queue_length=10)
        env_state = EnvironmentState(wind_speed=8.0, rain=True, gps_quality=0.7)
        onboard = OnboardPerception(
            self_state=self_state, environment_state=env_state
        )

        self.assertEqual(onboard.get_self_state().node_id, 4)
        self.assertEqual(onboard.get_self_state().speed, 12.5)
        self.assertEqual(onboard.get_environment_state().wind_speed, 8.0)
        self.assertTrue(onboard.get_environment_state().rain)

    def test_update_replaces_state(self) -> None:
        onboard = OnboardPerception()
        onboard.update_self_state(SelfState(node_id=9, queue_length=12))
        onboard.update_environment_state(EnvironmentState(temperature=-5.0))
        self.assertEqual(onboard.get_self_state().node_id, 9)
        self.assertEqual(onboard.get_self_state().queue_length, 12)
        self.assertEqual(onboard.get_environment_state().temperature, -5.0)

    def test_type_validation(self) -> None:
        onboard = OnboardPerception()
        with self.assertRaises(TypeError):
            onboard.update_self_state(EnvironmentState())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            onboard.update_environment_state(SelfState())  # type: ignore[arg-type]

    def test_input_isolation(self) -> None:
        from models import Position

        self_state = SelfState(position=Position(x=1.0, y=2.0, z=3.0))
        onboard = OnboardPerception(self_state=self_state)
        self_state.position.x = 99.0
        self.assertEqual(onboard.get_self_state().position.x, 1.0)

    def test_output_isolation_nested(self) -> None:
        onboard = OnboardPerception(
            self_state=SelfState(queue_length=5),
            environment_state=EnvironmentState(visibility=500.0),
        )
        snapshot = onboard.get_self_state()
        snapshot.queue_length = 99
        snapshot.battery.remaining = 10.0
        self.assertEqual(onboard.get_self_state().queue_length, 5)
        self.assertEqual(onboard.get_self_state().battery.remaining, 100.0)

        env_snapshot = onboard.get_environment_state()
        env_snapshot.visibility = 100.0
        self.assertEqual(onboard.get_environment_state().visibility, 500.0)

    def test_getter_returns_new_instance(self) -> None:
        onboard = OnboardPerception()
        self.assertIsNot(onboard.get_self_state(), onboard.get_self_state())
        self.assertIsNot(
            onboard.get_environment_state(), onboard.get_environment_state()
        )


class WirelessPerceptionTests(unittest.TestCase):
    def test_default_construction(self) -> None:
        wireless = WirelessPerception()
        self.assertIsInstance(wireless.get_neighbor_state(), NeighborState)
        self.assertIsInstance(wireless.get_routing_state(), RoutingState)
        self.assertEqual(wireless.get_routing_state().protocol, "GPSR")

    def test_injected_state_is_readable(self) -> None:
        neighbors = NeighborState(
            neighbors=[
                NeighborInfo(node_id=1, distance=12.0, link_quality=0.8),
                NeighborInfo(node_id=2, distance=20.0, link_quality=0.6),
            ]
        )
        neighbors.refresh_summary()
        routing = RoutingState(protocol="AODV", hello_interval=0.5, route_break_times=2)
        wireless = WirelessPerception(
            neighbor_state=neighbors, routing_state=routing
        )

        snapshot = wireless.get_neighbor_state()
        self.assertEqual(len(snapshot.neighbors), 2)
        self.assertIsNotNone(snapshot.neighbor_summary)
        self.assertEqual(snapshot.neighbor_summary.neighbor_num, 2)
        self.assertAlmostEqual(snapshot.neighbor_summary.avg_distance, 16.0)
        self.assertEqual(wireless.get_routing_state().protocol, "AODV")
        self.assertEqual(wireless.get_routing_state().route_break_times, 2)

    def test_summary_is_not_recomputed_on_read(self) -> None:
        neighbors = NeighborState(
            neighbors=[NeighborInfo(node_id=1, distance=5.0, link_quality=0.9)]
        )
        neighbors.refresh_summary()
        neighbors.neighbor_summary.best_forward_progress = 0.42
        wireless = WirelessPerception(neighbor_state=neighbors)
        snapshot = wireless.get_neighbor_state()
        self.assertEqual(snapshot.neighbor_summary.best_forward_progress, 0.42)

        wireless.update_neighbor_state(NeighborState())
        wireless.update_neighbor_state(
            NeighborState(
                neighbors=[NeighborInfo(node_id=1, distance=5.0, link_quality=0.9)]
            )
        )
        snapshot2 = wireless.get_neighbor_state()
        # No implicit refresh: summary stays None until the producer calls it.
        self.assertIsNone(snapshot2.neighbor_summary)

    def test_update_replaces_state(self) -> None:
        wireless = WirelessPerception()
        wireless.update_neighbor_state(
            NeighborState(neighbors=[NeighborInfo(node_id=7, distance=15.0)])
        )
        wireless.update_routing_state(
            RoutingState(protocol="OLSR", routing_table_size=4)
        )
        self.assertEqual(len(wireless.get_neighbor_state().neighbors), 1)
        self.assertEqual(wireless.get_neighbor_state().neighbors[0].node_id, 7)
        self.assertEqual(wireless.get_routing_state().protocol, "OLSR")
        self.assertEqual(wireless.get_routing_state().routing_table_size, 4)

    def test_type_validation(self) -> None:
        wireless = WirelessPerception()
        with self.assertRaises(TypeError):
            wireless.update_neighbor_state(RoutingState())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            wireless.update_routing_state(NeighborState())  # type: ignore[arg-type]

    def test_input_isolation(self) -> None:
        neighbor = NeighborState(
            neighbors=[NeighborInfo(node_id=1, distance=10.0)],
        )
        neighbor.refresh_summary()
        wireless = WirelessPerception(neighbor_state=neighbor)
        neighbor.neighbors[0].node_id = 99
        neighbor.neighbor_summary.best_forward_progress = 1.0
        snapshot = wireless.get_neighbor_state()
        self.assertEqual(snapshot.neighbors[0].node_id, 1)
        self.assertEqual(snapshot.neighbor_summary.best_forward_progress, 0.0)

    def test_output_isolation(self) -> None:
        neighbors = NeighborState(
            neighbors=[NeighborInfo(node_id=1, distance=10.0)],
        )
        neighbors.refresh_summary()
        wireless = WirelessPerception(neighbor_state=neighbors)
        snapshot = wireless.get_neighbor_state()
        snapshot.neighbors.append(NeighborInfo(node_id=2, distance=5.0))
        snapshot.neighbor_summary.avg_distance = 99.0
        self.assertEqual(len(wireless.get_neighbor_state().neighbors), 1)
        self.assertEqual(
            wireless.get_neighbor_state().neighbor_summary.avg_distance, 10.0
        )

    def test_getter_returns_new_instance(self) -> None:
        wireless = WirelessPerception()
        self.assertIsNot(wireless.get_neighbor_state(), wireless.get_neighbor_state())
        self.assertIsNot(wireless.get_routing_state(), wireless.get_routing_state())


if __name__ == "__main__":
    unittest.main()
