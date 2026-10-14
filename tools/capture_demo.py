"""Capture LUMO's actual OpenGL framebuffer, without recording the desktop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import os
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
from OpenGL.GL import GL_BACK, GL_PACK_ALIGNMENT, GL_RENDERER, GL_RGB, GL_UNSIGNED_BYTE, GL_VERSION
from OpenGL.GL import glGetString, glPixelStorei, glReadBuffer, glReadPixels

from lumo.gpu_view import GpuView
from lumo.showcase import SCENES, ShowcaseState
from lumo.activity import machine_cpu_percent, motion_activity, flow_speed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seconds-per-mode", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mode", choices=tuple(SCENES), help="Override live-load colors with a manual palette")
    parser.add_argument("--palettes", action="store_true", help="Capture all four manual palettes")
    parser.add_argument("--warmup-seconds", type=float, default=0, help="Allow live sensors to settle before capture")
    parser.add_argument("--load-demo", action="store_true", help="Record baseline, owned CPU workers, then recovery using real telemetry")
    parser.add_argument("--activity-demo", action="store_true", help="Record real CPU, RAM and disk I/O stages, then recovery")
    parser.add_argument("--view",choices=("connected","process"),default="process")
    args = parser.parse_args()
    live_demo=args.load_demo or args.activity_demo
    if live_demo and (args.mode or args.palettes):
        parser.error("Live activity demos require live colors")
    if args.fps <= 0 or args.seconds_per_mode <= 0 or args.warmup_seconds < 0:
        parser.error("fps and seconds-per-mode must be positive; warmup-seconds must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)

    state = ShowcaseState()
    view = None
    encoder = None
    completed = False
    workers = []
    temp_folder=tempfile.TemporaryDirectory(prefix="lumo-preview-",dir=args.output)
    started = time.monotonic()
    try:
        view = GpuView(state,connected=args.view=="connected",size=(1920,1080))
        width, height = view.screen.get_size()
        renderer = glGetString(GL_RENDERER).decode()
        version = glGetString(GL_VERSION).decode()
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        encoder = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(args.fps), "-i", "pipe:0", "-an",
            "-vf", "vflip", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(args.output / "lumo-demo.mp4"),
        ], stdin=subprocess.PIPE)
        clock = pygame.time.Clock()
        frames_per_mode = max(1, round(args.seconds_per_mode * args.fps))
        snapshots = []
        modes = ("baseline", "cpu-load", "ram-load", "disk-io", "recovery") if args.activity_demo else ("baseline", "cpu-load", "recovery") if args.load_demo else tuple(SCENES) if args.palettes else (args.mode or "read",)
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
            if live_demo:
                for worker in workers:
                    if worker.poll() is None:
                        worker.terminate()
                    worker.wait(timeout=5)
                workers=[]
                if mode == "cpu-load":
                    worker_code = "import ctypes,time; ctypes.CDLL(None).prctl(15,b'LUMO_LOAD',0,0,0); deadline=time.monotonic()+60; value=1\nwhile time.monotonic()<deadline:\n for i in range(10000): value=(value*1664525+1013904223)&0xffffffff"
                    workers = [subprocess.Popen([sys.executable,"-c",worker_code])
                               for _ in range(max(1,min(6,(os.cpu_count() or 2)-2)))]
                elif mode == "ram-load":
                    worker_code="import ctypes,time; ctypes.CDLL(None).prctl(15,b'LUMO_RAM',0,0,0); memory=bytearray(512*1024*1024)\nfor i in range(0,len(memory),4096): memory[i]=1\ntime.sleep(60)"
                    workers=[subprocess.Popen([sys.executable,"-c",worker_code])]
                elif mode == "disk-io":
                    filename=str(Path(temp_folder.name)/"owned-disk-demo.bin")
                    worker_code=f"import ctypes,time,os; ctypes.CDLL(None).prctl(15,b'LUMO_DISK',0,0,0); payload=os.urandom(4*1024*1024); deadline=time.monotonic()+60; f=open({filename!r},'wb')\nwhile time.monotonic()<deadline:\n f.seek(0); f.write(payload); f.flush(); os.fsync(f.fileno()); time.sleep(.1)"
                    workers=[subprocess.Popen([sys.executable,"-c",worker_code])]
            elif args.mode or args.palettes:
                state.select(mode)
            view.capture_label = f"LIVE DEMO / {mode.upper()} / OWNED WORKERS: {len(workers)}" if live_demo else ""
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
                if frame == frames_per_mode * 3 // 4:
                    surface = pygame.image.frombytes(pixels, (width, height), "RGB", True)
                    pygame.image.save(surface, args.output / f"lumo-{mode}.png")
                    snapshots.append({"mode": mode, "scene": "LIVE LOAD" if state.reactive else SCENES[mode][0],
                                      "tracked_processes": len(view.snapshot.process_details),
                                      "color_activity": [round(float(value),3) for value in view.loads],
                                      "machine_cpu_percent":view.snapshot.cpu,
                                      "ram_used_gib":round(view.snapshot.ram_used/2**30,3),
                                      "ram_total_gib":round(view.snapshot.ram_total/2**30,3),
                                      "disk_read_mib_s":round(view.snapshot.disk_read/2**20,3),
                                      "disk_write_mib_s":round(view.snapshot.disk_write/2**20,3),
                                      "motion_activity":round(float(motion_activity(view.loads)),3),
                                      "shared_flow_speed":round(float(flow_speed(motion_activity(view.loads))),3),
                                      "process_sources": [{"name":source.name,"cpu_percent":round(machine_cpu_percent(source.cpu,len(view.snapshot.cores)),1),
                                                           "raw_core_cpu_percent":round(source.cpu,1),
                                                           "flow_speed":round(float(source.flow_speed),3),
                                                           "pid_count":len(source.pids),"memory_mib":round(source.memory/2**20),
                                                           "ram_percent_of_machine":round(source.memory/max(1.,view.snapshot.ram_total)*100,1)}
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
                  "view":args.view,"load_demo":args.load_demo,"activity_demo":args.activity_demo,
                  "particles": view.particle_count,
                  "motion_backend": "GPU vertex shader / stable spiral cloud / measured system streams" if args.view=="connected" else "GPU vertex shader / per-source particle lifetime VBOs",
                  "light_pipeline": "4x MSAA RGBA16F HDR / two-scale GPU Gaussian bloom / filmic composite",
                  "connections": "thin input filaments and station node graphs" if args.view=="connected" else "no external LUMO cables or static filament lines",
                  "traffic": "same source particles enter LUMO, cross the middle with their source hue, and fade; illustrative flow",
                  "color_mode": ("live system activity / device hues" if args.view=="connected" else "live process CPU/RAM emission / persistent source hues") if state.reactive else "manual palettes",
                  "frames": frames_per_mode * len(modes),
                  "video_seconds": frames_per_mode * len(modes) / args.fps,
                  "capture_wall_seconds": round(time.monotonic() - started, 2),
                  "snapshots": snapshots}
        (args.output / "capture-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
            worker.wait(timeout=5)
        temp_folder.cleanup()
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
