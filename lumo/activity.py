"""Map live device readings to bounded light intensities without fabricating percentages."""
from __future__ import annotations

import math
from .telemetry import Snapshot


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
