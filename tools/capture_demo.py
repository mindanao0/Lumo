"""Capture LUMO's actual OpenGL framebuffer, without recording the desktop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
from OpenGL.GL import GL_BACK, GL_PACK_ALIGNMENT, GL_RENDERER, GL_RGB, GL_UNSIGNED_BYTE, GL_VERSION
from OpenGL.GL import glGetString, glPixelStorei, glReadBuffer, glReadPixels

from lumo.gpu_view import GpuView
from lumo.showcase import SCENES, ShowcaseState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seconds-per-mode", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mode", choices=tuple(SCENES), help="Override live-load colors with a manual palette")
    parser.add_argument("--palettes", action="store_true", help="Capture all four manual palettes")
    parser.add_argument("--warmup-seconds", type=float, default=0, help="Allow live sensors to settle before capture")
    args = parser.parse_args()
    if args.fps <= 0 or args.seconds_per_mode <= 0 or args.warmup_seconds < 0:
        parser.error("fps and seconds-per-mode must be positive; warmup-seconds must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)

    state = ShowcaseState()
    view = None
    encoder = None
    completed = False
    started = time.monotonic()
    try:
        view = GpuView(state)
        width, height = view.screen.get_size()
        renderer = glGetString(GL_RENDERER).decode()
        version = glGetString(GL_VERSION).decode()
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        encoder = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(args.fps), "-i", "pipe:0", "-an",
            "-vf", "vflip,scale=1280:720", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(args.output / "lumo-demo.mp4"),
        ], stdin=subprocess.PIPE)
        clock = pygame.time.Clock()
        frames_per_mode = max(1, round(args.seconds_per_mode * args.fps))
        snapshots = []
        modes = tuple(SCENES) if args.palettes else (args.mode or "read",)
        if args.mode:
            state.select(args.mode)
        for _ in range(round(args.warmup_seconds * args.fps)):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            state.update(1 / args.fps)
            view.draw()
            clock.tick(args.fps)
        for scene_index, mode in enumerate(modes):
            if args.mode or args.palettes:
                state.select(mode)
            for frame in range(frames_per_mode):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                    ):
                        raise KeyboardInterrupt
                state.update(1 / args.fps)
                view.draw(present=False)
                glReadBuffer(GL_BACK)
                pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
                encoder.stdin.write(pixels)
                if frame == frames_per_mode // 2:
                    surface = pygame.image.frombytes(pixels, (width, height), "RGB", True)
                    pygame.image.save(surface, args.output / f"lumo-{mode}.png")
                    snapshots.append({"mode": mode, "scene": "LIVE LOAD" if state.reactive else SCENES[mode][0],
                                      "tracked_processes": len(view.snapshot.process_details),
                                      "color_activity": [round(float(value),3) for value in view.loads],
                                      "process_sources": [{"name":source.name,"cpu_percent":round(source.cpu,1),
                                                           "pid_count":len(source.pids),"memory_mib":round(source.memory/2**20)}
                                                          for source in view.process_sources.sources],
                                      "gpu_percent": view.snapshot.gpu})
                pygame.display.flip()
                clock.tick(args.fps)
            print(f"Captured {mode}", flush=True)
        encoder.stdin.close()
        if encoder.wait(timeout=30) != 0:
            raise RuntimeError("FFmpeg could not finish the video")
        completed = True
        report = {"renderer": renderer, "opengl": version, "fps": args.fps,
                  "particles": view.particle_count,
                  "motion_backend": "GPU vertex shader / per-source particle lifetime VBOs",
                  "light_pipeline": "4x MSAA RGBA16F HDR / two-scale GPU Gaussian bloom / filmic composite",
                  "connections": "no external LUMO cables or static filament lines",
                  "traffic": "same source particles enter LUMO, circulate with their source hue, and decay; illustrative flow",
                  "color_mode": "live process CPU emission / persistent source hues" if state.reactive else "manual palettes",
                  "frames": frames_per_mode * len(modes),
                  "video_seconds": frames_per_mode * len(modes) / args.fps,
                  "capture_wall_seconds": round(time.monotonic() - started, 2),
                  "snapshots": snapshots}
        (args.output / "capture-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        if encoder is not None and encoder.poll() is None:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            try:
                encoder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                encoder.terminate()
                encoder.wait(timeout=10)
        if view is not None:
            view.close()
        if not completed:
            print("Capture did not complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
