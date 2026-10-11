"""Sample task traffic mapped to real graph paths. This is clearly simulation."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .graph import GROUPS, NodeGraph


SCENARIOS = {
    "read": ("Read file", ["storage", "ram", "process", "cpu"], False),
    "gpu": ("GPU job", ["cpu", "ram", "gpu"], False),
    "download": ("Download", ["network", "ram", "storage"], False),
    "unknown": ("Unknown route", ["process"], True),
}


@dataclass
class Packet:
    route: list[int]
    source_group: int
    progress: float
    speed: float
    color: tuple[float, float, float]
    fade: float = 1.0
    unknown: bool = False


class Traffic:
    def __init__(self, graph: NodeGraph, seed: int = 1948) -> None:
        self.graph = graph
        self.rng = random.Random(seed)
        self.mode = "read"
        self.packets: list[Packet] = []
        self.clock = 0.0
        self.spawn_clock = 0.0
        self.paused = False
        self.activity = 0.0
        self.select(self.mode)

    def select(self, mode: str) -> None:
        if mode not in SCENARIOS:
            raise ValueError(f"Unknown sample scenario: {mode}")
        self.mode = mode
        self.packets.clear()
        self.spawn_clock = 0.0
        for i in range(18):
            packet = self._make_packet()
            packet.progress = i / 18 * .82
            self.packets.append(packet)

    def _make_packet(self) -> Packet:
        _, chain, unknown = SCENARIOS[self.mode]
        route = self.graph.make_route(chain, unknown=unknown)
        source_index = next(i for i, group in enumerate(GROUPS) if group.key == chain[0])
        return Packet(route, source_index, 0.0, self.rng.uniform(.075, .115), GROUPS[source_index].color, unknown=unknown)

    def update(self, dt: float) -> None:
        if self.paused:
            return
        self.clock += dt
        self.spawn_clock += dt
        while self.spawn_clock >= .36:
            self.spawn_clock -= .36
            if len(self.packets) < 56:
                self.packets.append(self._make_packet())
        alive: list[Packet] = []
        for packet in self.packets:
            packet.progress += dt * packet.speed
            if packet.progress >= 1.0:
                packet.fade -= dt * (1.0 if packet.unknown else 1.6)
                if packet.fade <= 0:
                    continue
                packet.progress = 1.0
            alive.append(packet)
        self.packets = alive
        in_core = sum(self._current_group(packet) == len(GROUPS) for packet in self.packets)
        wanted = min(1.0, in_core / 12)
        self.activity += (wanted - self.activity) * min(1.0, dt * 3.0)

    def _current_group(self, packet: Packet) -> int:
        scaled = min(packet.progress * (len(packet.route) - 1), len(packet.route) - 1)
        edge_index = min(int(scaled), len(packet.route) - 2)
        return self.graph.nodes[packet.route[edge_index]].group

    def packet_positions(self) -> list[tuple[float, float, tuple[float, float, float], float]]:
        result = []
        for packet in self.packets:
            steps = len(packet.route) - 1
            if steps <= 0:
                continue
            scaled = min(packet.progress * steps, steps - 1e-6)
            index = min(int(scaled), steps - 1)
            f = scaled - index
            a, b = packet.route[index], packet.route[index + 1]
            curve = self.graph.edge_curve(a, b, self.clock, self.activity)
            point = curve(f)
            result.append((point[0], point[1], packet.color, packet.fade))
        return result

    def active_edges(self) -> dict[tuple[int, int], tuple[tuple[float, float, float], float]]:
        active: dict[tuple[int, int], tuple[tuple[float, float, float], float]] = {}
        for packet in self.packets:
            scaled = min(packet.progress * (len(packet.route) - 1), len(packet.route) - 1)
            index = min(int(scaled), len(packet.route) - 2)
            a, b = packet.route[index], packet.route[index + 1]
            active[(min(a, b), max(a, b))] = (packet.color, packet.fade)
        return active

    @property
    def status(self) -> str:
        name, _, unknown = SCENARIOS[self.mode]
        if unknown:
            return f"{name.upper()} · stays in LUMO, then fades"
        return f"{name.upper()} · {len(self.packets)} sample packets"

