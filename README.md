# LUMO — live process light

LUMO visualizes points of light from live program groups and the remaining
background processes entering one large circular vortex. Each packet
is born at its source, approaches on a curved path, smoothly joins circulation,
and makes 2.2–2.8 complete revolutions before a gradual 5–7 phase-second fade.
It keeps circulating throughout that fade. Live activity scales the presentation
clock, so elapsed wall time varies with load. The incoming particle population forms the
visible ring; no independent opaque ring mesh or permanent connecting cables
are drawn. The central opening contains a crystal without text.

## Live sources and readings

Local psutil readings group PIDs by program name. Five prominent groups are
shown at startup. A newly observed program or a new PID in an unshown program
forms an additional group after two successive sensor polls. Groups form and
fade over three seconds; transient processes and zero-resident-memory kernel
workers stay in `other processes`. Existing
programs retain their positions and colors when other apps open, close, or change
CPU ranking. By default up to 13 named program groups can be visible; overflow
remains in `other processes` and can be promoted when space becomes available.

Closing a program stops new particle births. Its cloud fades, while particles
already in flight or circulating keep their original source color and finish
their lifetime. The renderer retains these draining particles after their visible
slot is reused. Reopening the same program preserves its cached color and starts
a new birth interval. This is process-name grouping, not a count of desktop windows.
Main sources show measured aggregate CPU %, resident memory in MiB, and PID
count; additional compact sources show CPU %. CPU is the sum of process percentages and may exceed 100% for work on
multiple cores. Memory is summed resident memory across the group's PIDs.

Higher program CPU activity increases the visible contribution of its source hue.
A small resident-process glow keeps idle but present programs visible. Device
CPU, GPU, RAM, network, and storage readings remain as plain numbers at the top.
The system GPU/network counters are not attributed to individual programs.
All particles illustrate activity; they are not traces of actual data packets.
There are no metric cards, gauges, or bottom status strip.

## GPU rendering

Static buffers hold particle phase, source lane, wake age, and deterministic seeds.
GLSL evaluates the complete source-to-ring lifetime and grazing highlights on the
GPU. A cubic arrival path matches the circulation tangent at capture, so the same
particle continues through the junction. Fading wakes share their head's lifetime
and source hue. The canonical circular envelope stays fixed while internal
currents travel and fold through depth.

A 4× multisampled RGBA16F HDR framebuffer, two scales of Gaussian bloom,
and filmic compositing preserve sparks and surrounding light. No reference image
is used as the background. The particle lifecycle and light review used the
installed [shader-dev skill](https://github.com/MiniMax-AI/skills/tree/60aaae52bb2af8162732751a4332f62a5fef518b/skills/shader-dev),
adapted to the existing desktop OpenGL implementation.

## Run and controls

Run from any directory in a graphical Fedora desktop terminal:

```bash
lumo
lumo --fullscreen
lumo --max-apps 13 --fps 60
lumo --help
```

The `lumo` command is installed in `~/.local/bin`. To install it again,
run these commands from the project directory (with `~/.local/bin` on `PATH`):

```bash
mkdir -p ~/.local/bin
ln -s "$PWD/bin/lumo" ~/.local/bin/lumo
```

You can also launch directly with `./run.sh` or
`/home/da00/code/Lumo/run.sh`.

Press `Ctrl+C` in the terminal or `Q` / `Esc` in the window to exit.
The launcher prepares the virtual environment and installs dependencies when
missing. Existing working dependencies do not require a network check on every
launch. Desktop OpenGL 3.3
compatibility is required. Rendering has been verified on the local NVIDIA
GeForce GTX 1660 Ti. No hosted model or API key is needed.

| Key | Action |
| --- | --- |
| `A` | Toggle source colors / manual palette |
| `1`–`4` | Override with Aurora, Solar, Tidal, or Eclipse manual colors |
| `F` | Toggle fullscreen |
| `Space` | Pause or resume particle animation |
| `Esc` / `Q` | Quit |

Manual palette selection turns off live color response; press `A` to restore it.
Live readings continue updating during an animation pause. Manual colors still
retain the measured process emission response.

## Capture

Capture the current live-load mode from the actual OpenGL framebuffer:

```bash
.venv/bin/python tools/capture_demo.py --output artifacts/live-load --warmup-seconds 2 --seconds-per-mode 12
```

FFmpeg is required. The video exports at 1280×720 and 30 fps; the screenshot
retains the window's native resolution. Optional flags include `--fps`,
`--seconds-per-mode`, `--warmup-seconds`, `--mode read` (manual colors), and
`--palettes` (all four manual themes).

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/verify_render.py
.venv/bin/python tools/verify_live_apps.py
```

The second command needs a graphical OpenGL session and writes
`artifacts/cinematic-verification.json`. It checks actual GPU vertices for
single winding, circular outlines, packet origins, smooth capture, attached wakes, and frozen
positions while paused. It also checks HDR values and resizing, and measures
actual source hue and brightness changes under controlled process CPU fixtures,
separate from real capture telemetry. The third command launches and terminates
an owned `LUMO_PROBE` process, verifies its cloud forms and fades from live psutil
readings, and saves screenshots and a report under `artifacts/live-apps`.

Geometry and the color field are in `lumo/shaders.py`, program aggregation in `lumo/process_sources.py`, activity normalization
in `lumo/activity.py`, rendering and labels in `lumo/gpu_view.py`, and local
sensor readings in `lumo/telemetry.py`.
