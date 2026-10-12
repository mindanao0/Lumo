"""Exercise live process discovery and cloud fade using a real owned Linux process."""
from pathlib import Path
import json
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import pygame
from OpenGL.GL import GL_BACK,GL_RGB,GL_UNSIGNED_BYTE,glReadBuffer,glReadPixels

from lumo.gpu_view import GpuView
from lumo.showcase import ShowcaseState


def main() -> None:
    folder=Path("artifacts/live-apps")
    folder.mkdir(parents=True,exist_ok=True)
    state=ShowcaseState()
    view=GpuView(state,connected=False)
    child=None
    clock=pygame.time.Clock()
    try:
        def tick():
            pygame.event.pump()
            state.update(clock.tick(30)/1000.)
            view.draw()

        def capture(filename):
            view.draw(present=False)
            glReadBuffer(GL_BACK)
            pixels=glReadPixels(0,0,view.width,view.height,GL_RGB,GL_UNSIGNED_BYTE)
            pygame.image.save(pygame.image.frombytes(pixels,(view.width,view.height),"RGB",True),folder/filename)

        end=time.monotonic()+2.
        while time.monotonic()<end:
            tick()
        baseline={s.name:(s.slot,s.color) for s in view.process_sources.sources if s.closed_at is None}
        capture("before.png")
        child=subprocess.Popen([sys.executable,"-c",
            "import ctypes,time; ctypes.CDLL(None).prctl(15,b'LUMO_PROBE',0,0,0); memory=bytearray(16*1024*1024); time.sleep(60)"])
        deadline=time.monotonic()+10.
        source=None
        while time.monotonic()<deadline:
            tick()
            source=next((s for s in view.process_sources.sources if s.name=="LUMO_PROBE" and s.closed_at is None),None)
            if source and source.opacity>.99:
                break
        assert child.poll() is None,"Probe process exited unexpectedly"
        assert source and source.opacity>.99,"Real new process did not form a visible cloud"
        assert child.pid in source.pids,"The cloud is not attached to the real probe PID"
        slot,color=source.slot,source.color
        capture("opened.png")
        child.terminate()
        child.wait(timeout=3)
        deadline=time.monotonic()+8.
        while time.monotonic()<deadline:
            tick()
            if source.closed_at is not None and source.opacity<.01:
                break
        assert source.closed_at is not None and source.opacity<.01,"Closed process cloud did not fade"
        assert source in view.process_sources.sources and source.render_emission>0.,"In-flight light was dropped on close"
        for live in view.process_sources.sources:
            if live.name in baseline and live.closed_at is None:
                assert (live.slot,live.color)==baseline[live.name],"An existing program changed identity"
        capture("closed.png")
        report={"real_process_pid":child.pid,"name":"LUMO_PROBE","new_group_slot":slot,
                "source_color":color,"live_open_and_close_verified":True,
                "closed_cloud_opacity":source.opacity,"in_flight_particles_retained":True,
                "existing_positions_and_colors_preserved":True}
        (folder/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=3)
        view.close()


if __name__=="__main__":
    main()
