# LUMO — live process light

LUMO reads local device and process telemetry and renders GPU light currents.
The default program view restores the design used before the image-reference
experiments: loose light clouds for running programs and a diffuse central flow.
The central cloud fills an ellipsoid with circular cross sections, curved light
paths and a density that fades toward its boundary, avoiding straight slab edges
as it rotates. It rotates freely in 3D using a quaternion orientation. Drag the
cloud to set its angle; `O` toggles automatic rotation and `R` resets the angle.
The light scene fills the window. Each of its six permanent cloud positions has
three plain text lines: program name, CPU percentage of the whole machine, and
RAM percentage of total machine memory (summed resident memory for that group).
Each surrounding cloud rotates slowly and breathes gently about its fixed
anchor, with an individual phase and smoothly integrated activity response.
Labels stay anchored and all motion freezes when paused.
There are no floating cards, sidebar, top instrumentation band or bottom dock.
Unknown readings show a dash; values use one decimal place. Closing groups fade
with their labels while packets already in flight finish their journeys.

Use Space to pause, O to toggle automatic rotation, A to toggle live colors,
R to reset the angle and F for fullscreen. The cloud can still be rotated by
dragging. Inter names pair with Source Code Pro percentages, with font fallbacks.

Run `lumo --view process` explicitly for this view. It is also the default when
launching without options, and `--max-apps` controls its visible program groups.

## Experimental connected system view

The reference-based device constellations and spiral core remain available with
`lumo --view connected`. This experiment has no bottom status band, charts or
process table. It is not the default view.

## Live sources and readings

Local psutil readings group PIDs by program name. Five prominent groups are
shown at startup. A newly observed program or a new PID in an unshown program
forms an additional group after two successive sensor polls. Groups form and
fade over three seconds; transient processes and zero-resident-memory kernel
workers stay in `other processes`. Existing
programs retain their positions and colors when other apps open, close, or change
CPU ranking. By default up to five named program groups can be visible; overflow
remains in `other processes` and can be promoted when space becomes available.

Closing a program stops new particle births. Its cloud fades, while particles
already in flight or crossing keep their original source color and finish
their lifetime. The renderer retains these draining particles after their visible
slot is reused. Reopening the same program preserves its cached color and starts
a new birth interval. This is process-name grouping, not a count of desktop windows.
All sources show compact name / CPU% / RAM% labels without card backgrounds.
CPU sums the member PIDs and divides by logical CPU count; RAM sums resident
memory and divides by machine RAM. Unprimed readings show a dash.

Program CPU sets a target live packet population (20 + 1000 × its machine CPU
fraction). Births per second compensate for travel time: target population ×
current speed / 14, with a quiet floor of 0.25. Dust is distributed across and along the packet so its added points remain
visible rather than only increasing bloom. Each packet contains 36 dust
points and 12 wake points. Expired slots are excluded from drawing. Flow speed
uses a separate range (0.16 + 2.6 × sqrt(fraction)).
RAM changes the source cloud size, never its traffic rate. Births are scheduled
on the CPU and their coherent positions and wakes are evaluated on the GPU.
An ambient central volume responds to measured whole-machine CPU/GPU activity,
without assigning total GPU usage to individual programs.
A small resident-process glow keeps idle programs visible. The system GPU and
network counters are not attributed to individual programs. Particles illustrate
activity rather than tracing literal data packets. Packet lifetime covers the complete arrival, crossing and fade, divided by the
source speed at birth. Quiet particles persist roughly 74–101 seconds; busy
work crosses sooner in proportion to its measured CPU fraction. Existing programs start with a populated
particle volume, and the central ambient density has a stable floor so the
rounded shape stays visible at both low and high CPU/GPU load.

## GPU rendering

Dynamic packet pools hold birth phase, source lane, expiry, wake age and seeds.
GLSL evaluates the complete source-to-volume lifetime and soft light on the
GPU. A cubic arrival path matches the crossing velocity at capture, so the same
particle continues through the junction. Fading wakes share their head's lifetime
and source hue. Small smooth noise variations move the interior without large changes to the outline.
Particles span parallel lanes and depths, with no forced central aperture. Incoming
paths match the tangent of the evolving current. Source speeds ease over roughly
1.2 seconds; even idle programs retain a very slow resident drift.

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
lumo --max-apps 5 --fps 60
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
forward-only crossings, coherent neighbouring velocities, bounded paths through the centre, packet origins, smooth capture, attached wakes, and frozen
positions while paused. It also checks HDR values and resizing, and measures
actual source hue and brightness changes under controlled process CPU fixtures,
separate from real capture telemetry. The third command launches and terminates
an owned `LUMO_PROBE` process, verifies its cloud forms and fades from live psutil
readings, and saves screenshots and a report under `artifacts/live-apps`.

Geometry and the color field are in `lumo/shaders.py`, program aggregation in `lumo/process_sources.py`, activity normalization
in `lumo/activity.py`, rendering and labels in `lumo/gpu_view.py`, and local
sensor readings in `lumo/telemetry.py`.

## Capture review

Record a real baseline, CPU workers, a resident 512 MiB allocation, disk writes, and recovery:

```bash
.venv/bin/python tools/capture_demo.py --output artifacts/connected-system --warmup-seconds 4 --seconds-per-mode 10 --activity-demo
```

The recording uses live sensors throughout and labels each stage. The script
terminates its own workers between stages and also cleans them up on exit.
The disk stage repeatedly writes and syncs one owned temporary file, which is
removed after the recording. No global device counters are assigned to an app.
The capture report includes machine-normalized CPU, raw core CPU, and measured
source/shared visual clock speeds. Recording and rendering themselves contribute
to the observed CPU/GPU load.
