# LUMO: skill selection and particle direction

Research date: 2026-10-10. This records the initial review for discussion.
The user subsequently authorized a trial implementation and email delivery.
The current renderer now implements continuous per-source particle lifetimes;
see README.md and `artifacts/process-flow` for the resulting trial.

## Selected skill

Installed `shader-dev` from [MiniMax-AI/skills](https://github.com/MiniMax-AI/skills/tree/60aaae52bb2af8162732751a4332f62a5fef518b/skills/shader-dev)
at `/home/da00/.codex/skills/shader-dev`. Installation is pinned to commit
`60aaae52bb2af8162732751a4332f62a5fef518b`. All 37 technique links in SKILL.md
resolve to installed files. It can be discovered as a skill on the next turn.

The relevant material is particle-system, procedural-noise, domain-warping,
and post-processing. Use its particle lifecycle, flow-field, visibility, and
HDR composition principles. Its examples target ShaderToy/WebGL; adapt these
principles to the existing Python/OpenGL renderer rather than changing stacks.
Example code is a reference, not a substitute for compilation and measured
rendering checks. In particular, brightness and gamma handling need verification
against our actual HDR pipeline rather than a universal particle-count formula.

The [Three.js particle skill](https://github.com/cesartevisual/threejs-skills/blob/main/skills/threejs-particles/SKILL.md)
was also reviewed. Its buffer/lifecycle advice is useful, but its browser API
focus is less direct for this project; it was not installed.

## Plugin status

Blender MCP tools are exposed in this session. A read-only scene-info call
failed with `Could not connect to Blender. Make sure the Blender addon is running.`
Do not claim the plugin is connected. No existing Blender scene was modified.
A Blender search in the plugin directory returned no entries. A broader search
returned the Render hosting service, which is unrelated to graphics rendering
and was not suggested or installed.

## What the review found in LUMO

- In `lumo/gpu_view.py`, core particle positions and packet phases are initialized
  directly around the existing ring. No process identity is attached to them.
- In `lumo/shaders.py`, those packets advect around `ribbon(path)`; there is no
  incoming stage from an individual process source.
- `lumo/telemetry.py` already supplies PID, name, CPU, and memory readings for
  real processes. These can inform visible sources and emission strength.
- The current concept image depicts incoming process particles, but the running
  renderer does not implement that relationship yet.

## Direction for the next preview

Keep a stable circular overall envelope, an open dark center without text, and
plain device readings. Make the visible body of LUMO come from the incoming
particle population. The lifecycle should be continuous:

`process source -> incoming packets -> capture into circulation -> decay`

Every particle retains its source identity and color through entry and circulation.
The incoming packets should smoothly acquire the local tangential flow instead
of disappearing at an endpoint and being replaced by unrelated ring particles.
Use short fading wakes with individually readable heads, not permanent cables.

Start with a bounded set of prominent real process sources and an aggregated
background-work source. Keep source positions stable when process rankings change.
Use sampled process CPU activity for emission strength; memory can be a separate
visual parameter. Do not present total GPU/network counters as per-process values.

Add bounded flow-field variation and depth to create irregular gathering currents
within the circular envelope. Preserve dark gaps between denser folds. Retain
the source hues through convergence; reduce excess white bloom so those hues and
individual packets remain legible. The desired visual improvement is clearer
formation and circulation, supported by depth and light.

Evaluate a preview before editing the main application. Check incoming travel,
continuous capture, source hue preservation, low-activity decay, pause, circular
envelope, and GPU frame time. Technical implementation choices such as persistent
simulation versus analytic lifetime trajectories should follow that preview's
needs and the existing OpenGL support.
