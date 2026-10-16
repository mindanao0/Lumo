"""Pure six-slot contract. Coordinates are normalized, with top-left origin.

Measured from the 2048 x 1143 reference; core bounds include its diffuse halo.
No GUI objects or telemetry polling belong in this module.
"""
from dataclasses import dataclass
import math

from .process_sources import SOURCE_COLORS

SLOT_ANCHORS = ((.16, .19), (.12, .50), (.16, .79),
                (.84, .19), (.88, .50), (.84, .79))
LABEL_ANCHORS = ((.16, .303), (.12, .606), (.16, .892),
                 (.84, .303), (.88, .606), (.84, .892))
LABEL_OFFSETS = tuple((lx-x, ly-y) for (x,y),(lx,ly) in zip(SLOT_ANCHORS, LABEL_ANCHORS))
CORE_CENTER = (.50, .50)
CORE_BOUNDS = (.30, .20, .69, .84)
SLOT_ROLES = ('cpu', 'cpu', 'cpu', 'gpu', 'ram', 'other')
SLOT_COLORS = SOURCE_COLORS[:6]


def percent(value, total=100.):
    """Unknown denominators/counters stay unknown; measured zero stays zero."""
    if value is None or not math.isfinite(value) or value < 0 or not math.isfinite(total) or total <= 0:
        return None
    return min(100., value / total * 100.)


def short_name(name, limit=22):
    return name if len(name) <= limit else name[:limit-1] + '…'


@dataclass(frozen=True)
class SlotInfo:
    slot: int
    role: str
    identity: str | None
    name: str
    occupied: bool
    color: tuple[float, float, float]
    anchor: tuple[float, float]
    label_anchor: tuple[float, float]
    cpu_percent: float | None = None
    ram_percent: float | None = None
    gpu_percent: float | None = None
    total_ram_percent: float | None = None
    worker_ram_percent: float | None = None
    worker_ram_share: float | None = None
    activity: float = 0.
    stream_enabled: bool = False
    cloud_opacity: float = .04
    opacity: float = 1.


def build_slots(sources, snapshot, logical_cpus=None):
    """Build six current occupants; retired render identities never own metrics.

    GPU is whole-machine utilization, RAM activity is whole-machine used RAM.
    LUMO text is all worker RSS / machine capacity; share is RSS / used RAM.
    Pass ProcessSources or its source list and the matching telemetry Snapshot.
    """
    sources = getattr(sources, 'sources', sources)
    live = {s.slot: s for s in sources if s.closed_at is None}
    cores = max(1, len(snapshot.cores) if logical_cpus is None else logical_cpus)
    readings = {p.pid: p for p in snapshot.process_details}.values()
    worker_rss = sum(p.memory for p in readings if p.name.startswith('LUMO_')
                     and math.isfinite(p.memory) and p.memory > 0.)
    result = []
    for slot, role in enumerate(SLOT_ROLES):
        source = live.get(slot)
        occupied = source is not None and bool(source.pids)
        metrics = {}
        if occupied:
            cpu = percent(source.cpu, 100.*cores) if source.cpu_ready else None
            ram = percent(source.memory, snapshot.ram_total)
            activity = (cpu or 0.) / 100.
            metrics.update(cpu_percent=cpu, ram_percent=ram)
            if role == 'gpu':
                gpu = percent(snapshot.gpu)
                metrics['gpu_percent'] = gpu
                if gpu is not None:
                    activity = gpu / 100.
            elif role == 'ram':
                total = percent(snapshot.ram_used, snapshot.ram_total)
                metrics.update(total_ram_percent=total,
                               worker_ram_percent=percent(worker_rss, snapshot.ram_total),
                               worker_ram_share=percent(worker_rss, snapshot.ram_used))
                activity = (total or 0.) / 100.
            metrics.update(activity=activity, stream_enabled=True,
                           cloud_opacity=source.opacity, opacity=source.opacity)
        result.append(SlotInfo(slot, role, source.name if occupied else None,
                               ('Other programs' if slot == 5 else short_name(source.name)) if occupied else '—',
                               occupied, SLOT_COLORS[slot], SLOT_ANCHORS[slot], LABEL_ANCHORS[slot], **metrics))
    return tuple(result)
