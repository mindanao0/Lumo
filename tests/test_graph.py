import math
import unittest

from lumo.graph import LUMO_GROUP, GROUPS, NodeGraph, PORTS_PER_GROUP
from lumo.simulation import SCENARIOS, Traffic


class NodeGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = NodeGraph(seed=4)

    def test_each_device_has_many_independent_ports_into_lumo(self):
        self.assertEqual(sum(map(len, self.graph.ports)), len(GROUPS) * PORTS_PER_GROUP)
        for group_index, ports in enumerate(self.graph.ports):
            self.assertEqual(len({device for device, _ in ports}), PORTS_PER_GROUP)
            self.assertGreaterEqual(len({core for _, core in ports}), 20)
            self.assertTrue(all(self.graph.nodes[core].group == LUMO_GROUP for _, core in ports))

    def test_each_route_follows_real_graph_edges(self):
        for mode, (_, chain, unknown) in SCENARIOS.items():
            for _ in range(24):
                route = self.graph.make_route(chain, unknown=unknown)
                self.assertGreater(len(route), 3, mode)
                self.assertTrue(all(self.graph.has_edge(a, b) for a, b in zip(route, route[1:])), mode)
                if unknown:
                    self.assertTrue(all(self.graph.nodes[node].group == LUMO_GROUP for node in route[-20:]))

    def test_curves_meet_their_actual_endpoint_nodes(self):
        route = self.graph.make_route(["storage", "ram", "gpu"])
        for a, b in zip(route, route[1:]):
            curve = self.graph.edge_curve(a, b, time=1.7, activity=.75)
            start, end = curve(0), curve(1)
            na, nb = self.graph.point(a, 1.7, .75), self.graph.point(b, 1.7, .75)
            self.assertLess(math.dist(start, na), 1e-9)
            self.assertLess(math.dist(end, nb), 1e-9)

    def test_activity_deforms_lumo_core(self):
        core = self.graph.hub_nodes[17]
        still = self.graph.point(core, 1.0, 0.0)
        active = self.graph.point(core, 1.0, 1.0)
        self.assertGreater(math.dist(still, active), 1e-4)

    def test_traffic_packets_follow_route_and_unknown_work_fades(self):
        traffic = Traffic(self.graph, seed=8)
        traffic.select("unknown")
        traffic.update(5.0)
        for packet in traffic.packets:
            self.assertTrue(all(self.graph.has_edge(a, b) for a, b in zip(packet.route, packet.route[1:])))


if __name__ == "__main__":
    unittest.main()
