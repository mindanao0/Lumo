"""CPU-ranked live ownership, with separate outgoing render identities.

Default: five ranked groups in slots 0..4 and remainder in slot 5.
Reduced capacities use the first N slots. Explicit names preserve caller order
and bypass ranking; legacy capacities up to 13 skip reserved Other slot 5.
CPU margins use whole-machine percentage points. A gap over two seconds breaks
observed evidence rather than crediting an unobserved interval.
"""
from dataclasses import dataclass, field
import math
import time

from .telemetry import ProcessReading

SOURCE_COLORS = ((.16,.65,1.), (.14,.91,.86), (.72,.42,1.),
                 (.30,.94,.72), (1.,.62,.20), (1.,.80,.47),
                 (.30,.76,1.), (.98,.38,.58), (.63,.57,1.), (.28,1.,.62),
                 (.90,.72,.35), (.52,.90,1.), (.88,.42,.83), (1.,.48,.24))
APP_SLOTS = tuple(index for index in range(len(SOURCE_COLORS)) if index != 5)
CLOUD_FADE_SECONDS = 3.
PARTICLE_DRAIN_SECONDS = 34.
RANK_INTERVAL_SECONDS = 1.
RANK_HOLD_SECONDS = 3.
RANK_MARGIN_POINTS = 3.
MAX_SAMPLE_GAP_SECONDS = 2.
MAX_RETIRING_PER_SLOT = 12


def _positive(value: float) -> float:
    return float(value) if math.isfinite(value) and value > 0. else 0.


def _ease(value: float) -> float:
    value = min(1., max(0., value))
    return value*value*(3.-2.*value)


@dataclass
class ProcessSource:
    name: str
    slot: int = 0
    color: tuple[float, float, float] = SOURCE_COLORS[0]
    pids: tuple[int, ...] = ()
    cpu: float = 0.
    memory: float = 0.
    emission: float = 0.
    render_emission: float = 0.
    born_at: float = 0.
    born_flow: float = -40.
    closed_at: float | None = None
    stopped_flow: float | None = None
    opacity: float = 1.
    cloud_enabled: bool = True
    flow_clock: float = 0.
    flow_speed: float = .15
    cpu_ready: bool = True

    def measure(self, members: list[ProcessReading], logical_cpus: int = 1) -> None:
        self.pids = tuple(sorted(p.pid for p in members))
        self.cpu = sum(_positive(p.cpu) for p in members)
        self.cpu_ready = bool(members) and all(p.cpu_ready for p in members)
        self.memory = sum(_positive(p.memory) for p in members)
        self.emission = min(1.,self.cpu/(100.*max(1,logical_cpus))) if members and self.cpu_ready else 0.


@dataclass
class ProcessSources:
    names: list[str] = field(default_factory=list)
    max_apps: int = 5
    sources: list[ProcessSource] = field(default_factory=list)
    _live: dict[str, ProcessSource] = field(default_factory=dict)
    _retiring: list[ProcessSource] = field(default_factory=list)
    _background: ProcessSource | None = None
    _pending: dict[tuple[str, str], float] = field(default_factory=dict)
    _last_rank: float | None = None
    _evidence_sample: float | None = None
    _initialized: bool = False
    _last_sample: float | None = None
    _last_frame: float | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_apps <= len(APP_SLOTS):
            raise ValueError(f"max_apps must be between 1 and {len(APP_SLOTS)}")

    @property
    def live_sources(self) -> tuple[ProcessSource, ...]:
        """Current PID owners, including Other; excludes outgoing identities."""
        return tuple(sorted([*self._live.values(),
                             *([self._background] if self._background else [])],
                            key=lambda source: source.slot))

    @property
    def retiring_sources(self) -> tuple[ProcessSource, ...]:
        """Outgoing object identities for renderers; never allow new births."""
        return tuple(self._retiring)

    def _create(self, name: str, slot: int, now: float, flow: float, initial: bool) -> ProcessSource:
        color = SOURCE_COLORS[slot]
        source = ProcessSource(name, slot=slot, color=color, born_at=now-3. if initial else now,
                               born_flow=flow-40. if initial else flow, opacity=1. if initial else 0., flow_clock=flow)
        self._live[name] = source
        return source

    def update(self, readings: list[ProcessReading], *, now: float | None = None,
               flow_time: float | None = None, sampled_at: float | None = None,
               instant_emission: bool = False, logical_cpus: int = 1, source_speeds: dict[str, float] | None = None) -> list[ProcessSource]:
        now = time.monotonic() if now is None else now
        flow = now if flow_time is None else flow_time
        dt = 0. if self._last_frame is None else max(0.,now-self._last_frame)
        speed_blend = 1.-math.exp(-dt/1.2)
        for source in self.sources:
            if source_speeds is None:
                source.flow_clock = flow
            else:
                target = source_speeds.get(source.name,.15)
                source.flow_speed += (target-source.flow_speed)*speed_blend
                source.flow_clock += dt*source.flow_speed
        fresh = sampled_at is None or self._last_sample is None or sampled_at > self._last_sample
        if fresh:
            self._last_sample = sampled_at
            # Count each PID once even if a caller supplies a duplicated reading.
            groups: dict[str, list[ProcessReading]] = {}
            for process in {p.pid: p for p in readings}.values():
                groups.setdefault(process.name, []).append(process)
            stamp = now if sampled_at is None else sampled_at
            cpus = {name: sum(_positive(p.cpu) for p in members if p.cpu_ready)
                    / max(1, logical_cpus) for name, members in groups.items()}
            eligible = {name for name, members in groups.items()
                        if any(_positive(p.memory) > 0. for p in members)}
            ranked = sorted(eligible, key=lambda name: (-cpus[name], name))
            ordered = [s.name for s in sorted(self._live.values(), key=lambda s: s.slot)
                       if s.name in eligible]
            if self.names:
                chosen = [name for name in dict.fromkeys(self.names) if name in eligible][:self.max_apps]
            elif not self._initialized:
                chosen = ranked[:self.max_apps]
            else:
                chosen = ordered[:]
                # Missing groups release ownership immediately. Vacancies have no
                # incumbent to protect; fill them in current CPU order.
                chosen += [n for n in ranked if n not in chosen][:self.max_apps-len(chosen)]
                ready = {n for n in eligible if all(p.cpu_ready for p in groups[n])}
                advantages = {(a, b) for a in ready for b in ready
                              if cpus[a] >= cpus[b] + RANK_MARGIN_POINTS}
                continuous = (self._evidence_sample is not None and
                              0. < stamp-self._evidence_sample <= MAX_SAMPLE_GAP_SECONDS)
                self._pending = {pair: since for pair, since in self._pending.items()
                                 if pair in advantages and continuous}
                for pair in advantages:
                    self._pending.setdefault(pair, stamp)
                if self._last_rank is None or stamp-self._last_rank >= RANK_INTERVAL_SECONDS:
                    self._last_rank = stamp
                    # Insertion through the committed order applies the same
                    # pairwise evidence to internal reversals and admissions.
                    candidates = chosen + [n for n in ranked if n not in chosen]
                    for i in range(1, len(candidates)):
                        j = i
                        while j and stamp-self._pending.get((candidates[j], candidates[j-1]), stamp) >= RANK_HOLD_SECONDS:
                            candidates[j-1], candidates[j] = candidates[j], candidates[j-1]
                            j -= 1
                    chosen = candidates[:self.max_apps]
            self._evidence_sample = stamp
            initial = not self._initialized
            desired = dict(zip(chosen, APP_SLOTS[:self.max_apps]))
            for name, source in list(self._live.items()):
                if desired.get(name) != source.slot:
                    source.closed_at, source.stopped_flow = now, source.flow_clock
                    source.pids, source.cpu, source.memory, source.emission = (), 0., 0., 0.
                    self._retiring.append(source)
                    del self._live[name]
            for name, slot in desired.items():
                if name not in self._live:
                    # At most one visible outgoing cloud per position.
                    for old in self._retiring:
                        if old.slot == slot and old.closed_at != now:
                            old.cloud_enabled = False
                    self._create(name, slot, now, flow, initial)
            self._initialized = True
            for name, source in self._live.items():
                source.measure(groups[name],logical_cpus)
                if source.opacity == 0. or instant_emission:
                    source.render_emission = source.emission
            if self._background is None:
                self._background = ProcessSource("other processes",slot=5,color=SOURCE_COLORS[5],born_flow=flow-40.)
            self._background.measure([p for name,members in groups.items() if name not in self._live for p in members],logical_cpus)
            if self._last_frame is None or instant_emission:
                self._background.render_emission = self._background.emission

        self._last_frame = now
        blend = 1.-math.exp(-dt*1.8)
        for source in self._live.values():
            source.opacity = _ease((now-source.born_at)/CLOUD_FADE_SECONDS)
            source.render_emission += (source.emission-source.render_emission)*blend
        for source in self._retiring:
            source.opacity = 1.-_ease((now-source.closed_at)/CLOUD_FADE_SECONDS) if source.cloud_enabled else 0.
        self._retiring = [source for source in self._retiring
                          if source.flow_clock-source.stopped_flow < PARTICLE_DRAIN_SECONDS
                          and now-source.closed_at < 180.]
        self._retiring = [source for slot in APP_SLOTS
                          for source in [s for s in self._retiring if s.slot == slot][-MAX_RETIRING_PER_SLOT:]]
        if self._background is not None:
            self._background.render_emission += (self._background.emission-self._background.render_emission)*blend
        self.sources = sorted([*self._live.values(),*self._retiring,*([self._background] if self._background else [])],key=lambda source: source.slot)
        return self.sources
