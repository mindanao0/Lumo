"""Presentation timing; particle motion is evaluated by the GPU."""
from dataclasses import dataclass

SCENES = {
    "read": ("AURORA", (0.10, 0.65, 1.0), (1.0, 0.64, 0.23)),
    "gpu": ("SOLAR", (0.42, 0.36, 1.0), (1.0, 0.47, 0.20)),
    "download": ("TIDAL", (0.08, 0.83, 0.72), (0.60, 0.79, 1.0)),
    "unknown": ("ECLIPSE", (0.48, 0.30, 1.0), (1.0, 0.38, 0.67)),
}

@dataclass
class ShowcaseState:
    clock: float = 0.0
    paused: bool = False
    mode: str = "read"
    reactive: bool = True

    def select(self, mode: str) -> None:
        if mode not in SCENES:
            raise ValueError(f"Unknown scene: {mode}")
        self.mode = mode
        self.reactive = False

    def update(self, dt: float) -> None:
        if not self.paused:
            self.clock += dt
