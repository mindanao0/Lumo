# LUMO interface direction

Use the particle scene as the visual focus. Interface elements explain the
actual programs and device readings, with ample room around the central cloud.

## Tokens

- Canvas: #03090e; panel: #0b1720; raised control: #162935.
- Main text: #e8f3f8; secondary text: #9eb5c2; focus: #68cddf.
- Keep each program's existing particle color as its identity accent.
- Inter for labels, program names and controls; Source Code Pro for measurements.
- Font sizes at 900px height: 11, 13, 16, 22 and 25px.
- A 16px internal rhythm; larger panel curves, smaller control curves.

## Layout

Restore the full-window light scene. Five named program clouds and one aggregate
keep fixed positions. English labels sit below each cloud: a centered program
name followed by a compact measurement trace and paired CPU/RAM percentages.
CPU history rises above the baseline, RAM history falls below it, both using
the absolute share of system capacity. Keep the last 48 sensor samples;
unknown measurements break the trace. No synthetic oscillation or per-source
autoscaling. Inter names contrast with Source Code Pro readings. Explain
system capacity once below the small Lumo wordmark; omit routine process
counts and repeated footnotes. At 900px height use 14px names, 12px readings,
a 180px trace width, and a 19px amplitude on each side of the baseline.
No cards, sidebar, top band or bottom dock; all controls use existing shortcuts.

## Review against the brief

Both floating cards and a fixed sidebar were rejected. The scene should read as
light and space, with only the small percentage labels needed to understand it.
Program counts must not add UI blocks or source positions. Keep the central cloud
rounded, freely rotating, without visible coordinate axes.

CPU uses machine-normalized process CPU. RAM uses summed resident memory divided
by physical machine RAM, shown to one decimal. Unknown or closing readings show
a dash. Keep labels below each cloud, with no decorative progress meters.

Design guidance: https://github.com/anthropics/skills/tree/main/skills/frontend-design
