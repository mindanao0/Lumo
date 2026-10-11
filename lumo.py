"""LUMO's GPU particle-art showcase."""
from __future__ import annotations

import sys
import argparse

from lumo.gpu_view import GpuView
from lumo.showcase import SCENES, ShowcaseState


def main() -> int:
    parser=argparse.ArgumentParser(prog="lumo", description="LUMO: live GPU particle flow from running programs")
    parser.add_argument("--max-apps",type=int,default=13,choices=range(1,14),metavar="1-13",help="Maximum visible program groups (default: 13); remaining work is aggregated")
    parser.add_argument("--fullscreen",action="store_true",help="Start fullscreen")
    parser.add_argument("--fps",type=int,default=60,help="Frame rate limit (default: 60)")
    args=parser.parse_args()
    if not 1<=args.fps<=240:
        parser.error("--fps must be between 1 and 240")
    state = ShowcaseState()
    try:
        view = GpuView(state,max_apps=args.max_apps,fullscreen=args.fullscreen)
    except Exception as exc:
        print("LUMO could not create an OpenGL window. Install a Fedora Mesa/NVIDIA/AMD OpenGL driver and run from a graphical desktop.", file=sys.stderr)
        print(f"OpenGL error: {exc}", file=sys.stderr)
        return 2

    import pygame

    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            dt = min(clock.tick(args.fps) / 1000.0, max(.25,1./args.fps))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        state.paused = not state.paused
                    elif event.key == pygame.K_a:
                        state.reactive = not state.reactive
                    elif event.key == pygame.K_f:
                        pygame.display.toggle_fullscreen()
                        view._set_projection()
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                        mode = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4).index(event.key)
                        state.select(tuple(SCENES)[mode])
                elif event.type == pygame.VIDEORESIZE:
                    view._set_projection()
            state.update(dt)
            view.draw()
    except KeyboardInterrupt:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        view.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
