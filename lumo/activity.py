"""Map live device readings to bounded light intensities without fabricating percentages."""
from __future__ import annotations

import math
from .telemetry import Snapshot


def machine_cpu_percent(raw_cpu: float, logical_cpus: int) -> float:
    """Express process CPU on the same whole-machine scale as the HUD."""
    return min(100., max(0., raw_cpu / max(1, logical_cpus))) if math.isfinite(raw_cpu) else 0.


def motion_activity(loads: tuple[float, ...]) -> float:
    """Processing drives motion; memory occupancy never accelerates it."""
    compute = max(loads[0], loads[4])
    return min(1., compute + (1.-compute)*.25*max(loads[1], loads[5]))


def flow_speed(activity: float) -> float:
    return float(.15 + .95 * max(0., min(1., activity)) ** .65)


def device_activity(snapshot: Snapshot) -> tuple[float, ...]:
    """Return CPU, network, process CPU, RAM, GPU, disk activity in station order.

    CPU/GPU/RAM are measured fractions. Process CPU is normalized across logical
    cores. Network and disk have no universal utilization denominator, so their
    byte rates drive a monotonic visual intensity, never a displayed percentage.
    """
    def bounded(value: float) -> float:
        return min(1.,max(0.,value)) if math.isfinite(value) else 0.

    def flow(value: float, reference: float) -> float:
        rate=max(0.,value) if math.isfinite(value) else 0.
        return rate/(rate+reference)

    process_cpu=sum(process.cpu for process in snapshot.process_details)
    cores=max(1,len(snapshot.cores))
    return (bounded(snapshot.cpu/100.),
            flow(snapshot.net_down+snapshot.net_up,2**20*4.),
            bounded(process_cpu/(cores*100.)),
            bounded(snapshot.ram_used/max(1.,snapshot.ram_total)),
            bounded((snapshot.gpu or 0.)/100.),
            flow(snapshot.disk_read+snapshot.disk_write,2**20*12.))


def program_flow_speed(activity: float) -> float:
    """A deliberate wide range: quiet streams drift, busy streams visibly accelerate."""
    return float(.16+2.6*max(0.,min(1.,activity))**.5)
