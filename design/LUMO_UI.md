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
keep fixed positions. Only three centered text lines sit below each cloud:
program name, CPU percentage of the whole machine, RAM percentage of total RAM.
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
