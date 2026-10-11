"""Deterministic node graph and GPU-ready geometry for LUMO."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random

TAU = math.tau


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    center: tuple[float, float]
    spread: tuple[float, float]
    count: int
    color: tuple[float, float, float]


@dataclass(frozen=True)
class Node:
    x: float
    y: float
    z: float
    group: int
    phase: float
    u: float = 0.0
    v: float = 0.0


@dataclass(frozen=True)
class Edge:
    a: int
    b: int
    group: int
    kind: str


GROUPS = (
    Group("cpu", "CPU", (.20, .17), (.105, .105), 104, (.29, .77, 1.0)),
    Group("ram", "RAM", (.79, .17), (.105, .105), 104, (.36, .96, .77)),
    Group("gpu", "GPU", (.88, .48), (.095, .12), 128, (1.0, .70, .31)),
    Group("storage", "STORAGE", (.77, .82), (.12, .10), 88, (.96, .82, .48)),
    Group("process", "PROCESSES", (.23, .82), (.13, .105), 104, (.74, .57, 1.0)),
    Group("network", "NETWORK", (.12, .48), (.095, .12), 96, (.35, .87, .94)),
)
LUMO_GROUP = len(GROUPS)
PORTS_PER_GROUP = 24


def lumo_point(u: float, v: float, time: float = 0.0, activity: float = 0.0) -> tuple[float, float, float]:
    """Map a torus surface coordinate to the screen; all core nodes use this map."""
    twist = v + 3.0 * u + time * .07
    tube = .285 + .04 * math.sin(3.0 * u - time * .65) + activity * .035 * math.sin(5.0 * u - time * 1.3)
    radius = 1.0 + .065 * math.sin(3.0 * u + time * .15) + tube * math.cos(twist)
    x3 = radius * math.cos(u)
    y3 = radius * math.sin(u)
    z3 = tube * math.sin(twist)
    return .51 + x3 * .137, .47 + (y3 * .85 - z3 * .52) * .193, z3


class NodeGraph:
    """Subsystem meshes connected by many curved links to LUMO's own node mesh."""

    def __init__(self, seed: int = 20261009) -> None:
        self.rng = random.Random(seed)
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.adjacency: list[list[int]] = []
        self.edge_set: set[tuple[int, int]] = set()
        self.group_nodes: list[list[int]] = []
        self.ports: list[list[tuple[int, int]]] = []
        self.hub_nodes: list[int] = []
        self._make_groups()
        self._make_lumo_mesh()
        self._join_device_ports()

    def _node(self, x: float, y: float, z: float, group: int, phase: float, u: float = 0.0, v: float = 0.0) -> int:
        index = len(self.nodes)
        self.nodes.append(Node(x, y, z, group, phase, u, v))
        self.adjacency.append([])
        return index

    def _edge(self, a: int, b: int, kind: str) -> None:
        if a == b:
            return
        pair = (min(a, b), max(a, b))
        if pair in self.edge_set:
            return
        self.edge_set.add(pair)
        self.edges.append(Edge(a, b, self.nodes[a].group, kind))
        self.adjacency[a].append(b)
        self.adjacency[b].append(a)

    def _make_groups(self) -> None:
        for group_index, group in enumerate(GROUPS):
            ids: list[int] = []
            cx, cy = group.center
            sx, sy = group.spread
            for _ in range(group.count):
                # Compact irregular clouds read as node graphs, not round particle orbits.
                radius = math.sqrt(self.rng.random())
                angle = self.rng.random() * TAU
                x = cx + math.cos(angle) * sx * radius * (0.64 + self.rng.random() * .36)
                y = cy + math.sin(angle) * sy * radius
                ids.append(self._node(x, y, self.rng.uniform(-.5, .5), group_index, self.rng.random() * TAU))
            self.group_nodes.append(ids)
            # Build a connected local web: nearest neighbors plus an explicit spanning edge.
            for offset, node_id in enumerate(ids):
                nearest = sorted((other for other in ids if other != node_id), key=lambda other: self._distance(node_id, other))
                for other in nearest[:5]:
                    self._edge(node_id, other, "local")
                if offset:
                    previous = ids[:offset]
                    self._edge(node_id, min(previous, key=lambda other: self._distance(node_id, other)), "local")

    def _make_lumo_mesh(self) -> None:
        # 6 x 64 vertices form the torus itself; links and packets run on this surface.
        self.hub_nodes = []
        for layer in range(6):
            v = layer / 6 * TAU
            for i in range(64):
                u = i / 64 * TAU
                x, y, z = lumo_point(u, v)
                self.hub_nodes.append(self._node(x, y, z, LUMO_GROUP, 0.0, u, v))
        for layer in range(6):
            for i in range(64):
                here = self.hub_nodes[layer * 64 + i]
                self._edge(here, self.hub_nodes[layer * 64 + (i + 1) % 64], "lumo")
                self._edge(here, self.hub_nodes[((layer + 1) % 6) * 64 + i], "lumo")
                self._edge(here, self.hub_nodes[((layer + 1) % 6) * 64 + (i + 1) % 64], "lumo")

    def _join_device_ports(self) -> None:
        for group_index, group in enumerate(GROUPS):
            ids = self.group_nodes[group_index]
            candidates = sorted(ids, key=lambda i: math.hypot(self.nodes[i].x - .51, self.nodes[i].y - .47))
            candidates = candidates[: math.ceil(len(ids) * .65)]
            angle = math.atan2(group.center[1] - .47, group.center[0] - .51)
            ports: list[tuple[int, int]] = []
            for i in range(PORTS_PER_GROUP):
                device_node = candidates[int(i * len(candidates) / PORTS_PER_GROUP)]
                theta = angle + (i / (PORTS_PER_GROUP - 1) - .5) * 1.8
                ring = int(round(theta / TAU * 64)) % 64
                layer = i % 6
                lumo_node = self.hub_nodes[layer * 64 + ring]
                self._edge(device_node, lumo_node, "bridge")
                ports.append((device_node, lumo_node))
            self.ports.append(ports)

    def _distance(self, a: int, b: int) -> float:
        p, q = self.nodes[a], self.nodes[b]
        return (p.x - q.x) ** 2 + (p.y - q.y) ** 2

    def path(self, start: int, finish: int, group: int) -> list[int]:
        if start == finish:
            return [start]
        queue = deque([start])
        parent = {start: -1}
        while queue:
            node = queue.popleft()
            for neighbor in self.adjacency[node]:
                if self.nodes[neighbor].group != group or neighbor in parent:
                    continue
                parent[neighbor] = node
                if neighbor == finish:
                    out = [finish]
                    while out[-1] != start:
                        out.append(parent[out[-1]])
                    return list(reversed(out))
                queue.append(neighbor)
        raise ValueError(f"No path inside node group {group}: {start} -> {finish}")

    def has_edge(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in self.edge_set

    def make_route(self, keys: list[str], unknown: bool = False) -> list[int]:
        indexes = {group.key: i for i, group in enumerate(GROUPS)}
        if not keys:
            raise ValueError("A route needs at least one subsystem")
        group_index = indexes[keys[0]]
        start = self.rng.choice(self.group_nodes[group_index])
        route = [start]
        current_group, current_node = group_index, start
        for key in keys[1:]:
            target_group = indexes[key]
            exit_node, exit_hub = self.rng.choice(self.ports[current_group])
            entry_node, entry_hub = self.rng.choice(self.ports[target_group])
            route.extend(self.path(current_node, exit_node, current_group)[1:])
            route.append(exit_hub)
            route.extend(self.path(exit_hub, entry_hub, LUMO_GROUP)[1:])
            route.append(entry_node)
            current_node = self.rng.choice(self.group_nodes[target_group])
            route.extend(self.path(entry_node, current_node, target_group)[1:])
            current_group = target_group
        if unknown:
            exit_node, exit_hub = self.rng.choice(self.ports[current_group])
            route.extend(self.path(current_node, exit_node, current_group)[1:])
            route.append(exit_hub)
            previous = -1
            at = exit_hub
            for _ in range(20):
                choices = [n for n in self.adjacency[at] if self.nodes[n].group == LUMO_GROUP and n != previous]
                next_node = self.rng.choice(choices)
                route.append(next_node)
                previous, at = at, next_node
        if any(not self.has_edge(a, b) for a, b in zip(route, route[1:])):
            raise AssertionError("Generated route contains a jump that is not a graph edge")
        return route

    def point(self, node_id: int, time: float, activity: float) -> tuple[float, float, float]:
        node = self.nodes[node_id]
        if node.group == LUMO_GROUP:
            return lumo_point(node.u, node.v, time, activity)
        # Small coherent cloud motion; subsystem nodes do not orbit the core.
        return node.x + .0015 * math.sin(time * .55 + node.phase), node.y + .0015 * math.cos(time * .60 + node.phase), node.z

    def edge_curve(self, a: int, b: int, time: float, activity: float):
        """Return a curve that meets its graph nodes exactly at t=0 and t=1."""
        na, nb = self.nodes[a], self.nodes[b]
        if na.group == LUMO_GROUP and nb.group == LUMO_GROUP:
            du = math.atan2(math.sin(nb.u - na.u), math.cos(nb.u - na.u))
            dv = math.atan2(math.sin(nb.v - na.v), math.cos(nb.v - na.v))
            return lambda f: lumo_point(na.u + du * f, na.v + dv * f, time, activity)
        p, q = self.point(a, time, activity), self.point(b, time, activity)
        if na.group == nb.group:
            return lambda f: tuple(p[i] + (q[i] - p[i]) * f for i in range(3))
        reversed_link = na.group == LUMO_GROUP
        if reversed_link:
            a, b, na, nb, p, q = b, a, nb, na, q, p
        surface = self.nodes[b]
        tangent = lumo_point(surface.u + .065, surface.v, time, activity)
        origin = lumo_point(surface.u, surface.v, time, activity)
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = math.hypot(dx, dy)
        tlen = math.hypot(tangent[0] - origin[0], tangent[1] - origin[1]) or 1.0
        bend = math.sin((a + b) * .47) * .35
        c1 = (p[0] + dx * .37 - dy * bend, p[1] + dy * .37 + dx * bend, 0.0)
        direction = 1.0 if dx * (tangent[0] - origin[0]) + dy * (tangent[1] - origin[1]) >= 0 else -1.0
        c2 = (q[0] - (tangent[0] - origin[0]) / tlen * length * .24 * direction,
              q[1] - (tangent[1] - origin[1]) / tlen * length * .24 * direction, 0.0)

        def bezier(f: float):
            s = 1.0 - f
            return tuple(s**3 * p[i] + 3*s*s*f*c1[i] + 3*s*f*f*c2[i] + f**3*q[i] for i in range(3))
        return (lambda f: bezier(1.0 - f)) if reversed_link else bezier
