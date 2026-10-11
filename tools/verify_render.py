"""Verify the actual OpenGL pipeline, internal packet flow, pause, palettes, and resizing."""
from __future__ import annotations

import ctypes
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pygame
from pygame._sdl2.video import Window
from OpenGL.GL import (GL_BACK,GL_FRAMEBUFFER,GL_FRAMEBUFFER_COMPLETE,GL_RENDERER,GL_RGB,
                       GL_UNSIGNED_BYTE,GL_VERSION,glBindFramebuffer,glCheckFramebufferStatus,
                       glFinish,glGetError,glGetString,glReadBuffer,glReadPixels)

from lumo.gpu_view import GpuView, STATIONS, NODE_COLORS, SOURCE_BANDS
from lumo.showcase import SCENES,ShowcaseState
from lumo.shaders import PARTICLE_VERTEX,LINK_FRAGMENT
from lumo.telemetry import Snapshot, ProcessReading
from lumo.process_sources import ProcessSources


def verify_geometry_and_flow(view: GpuView) -> dict:
    """Read actual GPU vertices to check the circular shell and moving packet wakes."""
    import OpenGL.GL as gl
    from OpenGL.GL.shaders import compileShader
    program=gl.glCreateProgram()
    vs=compileShader(PARTICLE_VERTEX,gl.GL_VERTEX_SHADER)
    fs=compileShader(LINK_FRAGMENT,gl.GL_FRAGMENT_SHADER)
    gl.glAttachShader(program,vs)
    gl.glAttachShader(program,fs)
    captures=[ctypes.create_string_buffer(name) for name in (b"gl_Position",b"vAlpha")]
    names=(ctypes.POINTER(ctypes.c_char)*2)(*[ctypes.cast(name,ctypes.POINTER(ctypes.c_char)) for name in captures])
    gl.glTransformFeedbackVaryings(program,2,names,gl.GL_INTERLEAVED_ATTRIBS)
    gl.glLinkProgram(program)
    assert gl.glGetProgramiv(program,gl.GL_LINK_STATUS),gl.glGetProgramInfoLog(program)
    uniforms={name:gl.glGetUniformLocation(program,name) for name in view.uniforms}
    source=int(gl.glGenBuffers(1))
    output=int(gl.glGenBuffers(1))
    previous_clock,previous_flow=view.state.clock,view.flow_time
    winding_checks=[]
    roundness_checks=[]
    try:
        def evaluate(kind: int,positions,seed=.5,slot=0,birth=-40.,stop=1.e10,cloud_alpha=1.) -> np.ndarray:
            gl.glUseProgram(program)
            view._configure(uniforms,view.state.clock)
            gl.glUniform1i(uniforms["uKind"],kind)
            x,y=STATIONS[slot][:2]
            gl.glUniform2f(uniforms["uCenter"],x,y)
            gl.glUniform3f(uniforms["uNodeColor"],*NODE_COLORS[slot])
            gl.glUniform1f(uniforms["uEntry"],np.arctan2(y,x+.025))
            gl.glUniform1f(uniforms["uBand"],SOURCE_BANDS[slot])
            gl.glUniform1f(uniforms["uBirthTime"],birth)
            gl.glUniform1f(uniforms["uStopTime"],stop)
            gl.glUniform1f(uniforms["uCloudAlpha"],cloud_alpha)
            colors=np.tile((.3,.5,seed),(len(positions),1))
            data=np.ascontiguousarray(np.column_stack((positions,colors)),dtype=np.float32)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER,source)
            gl.glBufferData(gl.GL_ARRAY_BUFFER,data.nbytes,data,gl.GL_STREAM_DRAW)
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glVertexPointer(3,gl.GL_FLOAT,24,ctypes.c_void_p(0))
            gl.glColorPointer(3,gl.GL_FLOAT,24,ctypes.c_void_p(12))
            gl.glBindBuffer(gl.GL_TRANSFORM_FEEDBACK_BUFFER,output)
            gl.glBufferData(gl.GL_TRANSFORM_FEEDBACK_BUFFER,len(positions)*20,None,gl.GL_STREAM_READ)
            gl.glBindBufferBase(gl.GL_TRANSFORM_FEEDBACK_BUFFER,0,output)
            gl.glEnable(gl.GL_RASTERIZER_DISCARD)
            gl.glBeginTransformFeedback(gl.GL_POINTS)
            gl.glDrawArrays(gl.GL_POINTS,0,len(positions))
            gl.glEndTransformFeedback()
            gl.glDisable(gl.GL_RASTERIZER_DISCARD)
            raw=gl.glGetBufferSubData(gl.GL_TRANSFORM_FEEDBACK_BUFFER,0,len(positions)*20)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            return np.frombuffer(raw,dtype=np.float32).reshape(-1,5).copy()

        for clock in (0.,7.,25.):
            view.state.clock=clock
            circle=evaluate(0,[(u,0.,0.) for u in np.linspace(0,2*np.pi,1025)])
            xy=circle[:,:2].astype(float)*np.array((view.aspect,1.))-np.array((-.025,0.))
            angle=np.unwrap(np.arctan2(xy[:,1],xy[:,0]))
            turns=float((angle[-1]-angle[0])/(2*np.pi))
            assert abs(turns-1.)<.001,f"Core winds {turns} times instead of once"
            assert np.linalg.norm(xy[0]-xy[-1])<1e-5,"Core seam does not close"
            winding_checks.append(turns)
            radial=np.linalg.norm(xy,axis=1)
            roundness=float(radial.max()-radial.min())
            assert roundness<2e-6,"Core centreline is not circular"
            angles=np.linspace(0,2*np.pi,73)
            profile=np.linspace(-1.,1.,721)
            surface=evaluate(0,[(u,v,1.) for u in angles for v in profile])
            surface_xy=surface[:,:2].astype(float)*np.array((view.aspect,1.))-np.array((-.025,0.))
            radii=np.linalg.norm(surface_xy,axis=1).reshape(len(angles),len(profile))
            outer_error=float(np.max(np.abs(radii.max(axis=1)-.830)))
            inner_error=float(np.max(np.abs(radii.min(axis=1)-.240)))
            assert outer_error<1e-5 and inner_error<1e-5,"Core rim or aperture is not circular"
            roundness_checks.append({"centre_radius_spread":roundness,"outer_radius_error":outer_error,"inner_radius_error":inner_error})

        # Freeze the shell to isolate real packet travel from slow surface motion.
        view.state.clock=0.
        view.flow_time=0.
        packets=[(phase,lane,0.) for phase in np.linspace(0.,1.,37)[:-1]
                 for lane in (-.10,0.,.10)]
        head=evaluate(19,packets)
        tail_start=evaluate(20,packets)
        wake_error=float(np.max(np.abs(head[:,:4]-tail_start[:,:4])))
        assert wake_error<1e-6,"A particle wake is detached from its advancing head"
        view.flow_time=.35
        advanced=evaluate(19,packets)
        displacement=np.linalg.norm((advanced[:,:2]-head[:,:2])*np.array((view.aspect,1.)),axis=1)
        assert np.isfinite(advanced).all() and displacement.mean()>.06,"Internal data heads did not travel visibly"
        repeated=evaluate(19,packets)
        assert np.array_equal(advanced,repeated),"Frozen packet phase does not pause exactly"
        arrival=3.25
        fade_start=arrival+2.5*2*np.pi/.71
        lifetime=fade_start+6.
        junctions=[]
        rotation_checks=[]
        fade_checks=[]
        for slot,(x,y,*_) in enumerate(STATIONS):
            view.flow_time=0.
            origin=evaluate(19,[(0.,0.,0.)],slot=slot)[0,:2]*np.array((view.aspect,1.))
            assert np.linalg.norm(origin-np.array((x,y)))<.05,"A packet is not born at its process cloud"
            samples=evaluate(19,[(age/lifetime,0.,0.) for age in (arrival-.002,arrival,arrival+.002)],slot=slot)[:,:2]
            incoming=(samples[1]-samples[0])/.002
            outgoing=(samples[2]-samples[1])/.002
            mismatch=float(np.linalg.norm(incoming-outgoing)/max(np.linalg.norm(incoming),.01))
            assert mismatch<.04,f"A source jumps or changes direction abruptly on capture: {mismatch}"
            junctions.append(mismatch)
            ring_points=evaluate(19,[(age/lifetime,lane,0.) for age in np.linspace(arrival+.1,lifetime-.1,120) for lane in (-.14,0.,.14)],slot=slot)
            world=ring_points[:,:2]*np.array((view.aspect,1.))-np.array((-.025,0.))
            radii=np.linalg.norm(world,axis=1)
            assert radii.min()>=.23999 and radii.max()<=.83001,"Captured packets left the LUMO interior"
            # Follow one actual GPU particle through its bright residence and fade.
            held=evaluate(19,[(age/lifetime,0.,0.) for age in np.linspace(arrival+.05,fade_start,181)],slot=slot)
            xy=held[:,:2]*np.array((view.aspect,1.))-np.array((-.025,0.))
            angles=np.unwrap(np.arctan2(xy[:,1],xy[:,0]))
            turns=float((angles[-1]-angles[0])/(2*np.pi))
            assert turns>2.2,"Particle faded before completing several revolutions"
            assert held[:,4].min()>.15,"Captured particle started fading during its bright residence"
            faded=evaluate(19,[(age/lifetime,0.,0.) for age in np.linspace(fade_start+.01,lifetime-.01,25)],slot=slot)
            assert faded[:5,4].mean()>faded[-5:,4].mean()*3.,"Particle does not fade gradually after rotating"
            assert faded[-1,4]<.0001,"Particle remains visible at the end of its lifetime"
            rotation_checks.append(turns)
            fade_checks.append({"start_alpha":float(faded[0,4]),"end_alpha":float(faded[-1,4])})
            wake=evaluate(20,[(phase,lane,age) for phase,lane,_ in packets for age in (0.,.5,1.)],slot=slot)
            assert np.isfinite(wake).all(),"Invalid process particle wake"
        view.flow_time=5.
        def age_point(age):
            return [(((age-view.flow_time)%lifetime)/lifetime,0.,0.)]
        old=evaluate(19,age_point(6.),stop=4.)
        new=evaluate(19,age_point(.2),stop=4.)
        assert old[0,4]>.1,"Closing an app removed particles already circulating"
        assert new[0,4]==0.,"A closed app is still spawning new particles"
        unopened=evaluate(19,age_point(6.),birth=3.)
        opened=evaluate(19,age_point(1.),birth=3.)
        assert unopened[0,4]==0. and opened[0,4]>.1,"A new app inherited pre-existing ring particles"
        hidden_cloud=evaluate(11,[(0.,0.,0.)],cloud_alpha=0.)
        assert hidden_cloud[0,4]==0.,"A faded group cloud remains visible"
        view.flow_time=.35
        assert gl.glGetError()==0,"GL error during internal flow verification"
        return {"core_revolutions_from_gpu":winding_checks,"circular_silhouette_checks":roundness_checks,
                "internal_packet_checks":len(packets),"wake_head_max_error":wake_error,
                "mean_packet_travel_in_350ms":float(displacement.mean()),
                "frozen_flow_positions_identical":True,"packets_stay_inside_ring":True,
                "process_cloud_origins_checked":len(STATIONS),"max_capture_tangent_mismatch":max(junctions),
                "visible_revolutions_before_fade":rotation_checks,"gradual_fade_checks":fade_checks,
                "app_open_birth_gate_verified":True,"closed_app_stops_new_births":True,
                "closed_app_particles_finish_circulating":True,"cloud_fade_uniform_verified":True}
    finally:
        view.state.clock,view.flow_time=previous_clock,previous_flow
        gl.glDisable(gl.GL_RASTERIZER_DISCARD)
        gl.glBindBufferBase(gl.GL_TRANSFORM_FEEDBACK_BUFFER,0,0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER,0)
        gl.glUseProgram(0)
        gl.glDeleteBuffers(2,[source,output])
        gl.glDeleteProgram(program)
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)


def verify_process_emission(view: GpuView) -> dict:
    """Measure real rendered source hues and emission response with controlled fixtures."""
    saved=(view.telemetry.update,view.snapshot,view.state.paused,view.state.reactive,
           view.state.mode,view.loads.copy(),view.cool.copy(),view.warm.copy(),
           view.last_mode,view.last_time,view.loads_initialized,view.process_sources)
    yy,xx=np.mgrid[0:view.height,0:view.width]
    x=(2.*(xx+.5)/view.width-1.)*view.aspect+.025
    y=2.*(yy+.5)/view.height-1.
    radius=np.sqrt(x*x+y*y)
    ring=(radius>.25)&(radius<.83)
    timestamp=1000.
    try:
        view.state.paused=True
        view.state.reactive=True
        view.process_sources=ProcessSources(names=[f"fixture-{i}" for i in range(5)])
        def capture(blue_cpu=0.,gold_cpu=0.,global_cpu=0.,global_gpu=0.):
            nonlocal timestamp
            timestamp+=1.
            processes=[ProcessReading(i+1,f"fixture-{i}",blue_cpu if i==0 else gold_cpu if i==4 else 0.,2**24) for i in range(6)]
            sample=Snapshot(cpu=global_cpu,gpu=global_gpu,cores=[0.]*8,ram_total=16.*2**30,updated=timestamp,process_details=processes)
            view.telemetry.update=lambda:sample
            view.loads_initialized=False
            view.draw(present=False)
            glReadBuffer(GL_BACK)
            return np.frombuffer(glReadPixels(0,0,view.width,view.height,GL_RGB,GL_UNSIGNED_BYTE),dtype=np.uint8).reshape(view.height,view.width,3).copy()
        idle=capture()
        blue_low=capture(blue_cpu=20.)
        blue_high=capture(blue_cpu=100.)
        gold_high=capture(gold_cpu=100.)
        def contrast(frame,first,last):
            pixels=frame[ring].astype(float)
            return float((pixels[:,first]-pixels[:,last]).mean())
        blue_gain=contrast(blue_high,2,0)-contrast(idle,2,0)
        gold_gain=contrast(gold_high,0,2)-contrast(idle,0,2)
        scale=float(blue_high[ring,2].mean()-blue_low[ring,2].mean())
        assert blue_gain>1.,f"Active blue source did not strengthen its hue in LUMO: {blue_gain}"
        assert gold_gain>1.,f"Active gold source did not strengthen its hue in LUMO: {gold_gain}"
        assert scale>1.,f"Higher process activity did not increase visible emission: {scale}"
        view.state.reactive=False
        view.state.mode="read"
        manual_idle=capture(blue_cpu=20.,gold_cpu=20.)
        manual_loaded=capture(blue_cpu=20.,gold_cpu=20.,global_cpu=100.,global_gpu=100.)
        difference=float(np.abs(manual_idle[ring].astype(float)-manual_loaded[ring]).mean())
        assert difference<.05,"Global device counters were incorrectly attributed to individual processes"
        return {"measured_on_actual_framebuffer":True,"blue_source_hue_gain":blue_gain,
                "gold_source_hue_gain":gold_gain,"higher_process_load_additional_light":scale,
                "device_counters_not_attributed_to_processes":True}
    finally:
        (view.telemetry.update,view.snapshot,view.state.paused,view.state.reactive,
         view.state.mode,loads,cool,warm,view.last_mode,view.last_time,view.loads_initialized,
         view.process_sources)=saved
        view.loads[:]=loads
        view.cool[:]=cool
        view.warm[:]=warm
        view.hud_key=None


def main() -> None:
    state=ShowcaseState()
    view=GpuView(state)
    try:
        view.draw(present=False)
        snapshot=view.snapshot
        # Freeze telemetry to isolate animation and compositing from real counter changes.
        view.telemetry.update=lambda: snapshot
        def frame() -> np.ndarray:
            view.draw(present=False)
            glReadBuffer(GL_BACK)
            return np.frombuffer(glReadPixels(0,0,view.width,view.height,GL_RGB,GL_UNSIGNED_BYTE),dtype=np.uint8).copy()
        before=frame()
        state.update(.6)
        moved=frame()
        assert np.mean(np.abs(before.astype(float)-moved))>.1,"Animation did not move"
        state.paused=True
        frozen=frame()
        state.update(1.)
        paused=frame()
        delta=np.abs(frozen.astype(float)-paused)
        # Additive float blending and multisample resolve allow a few quantization levels.
        assert delta.max()<=3 and delta.mean()<.01,"Paused geometry moved"
        assert len({p.pid for p in snapshot.process_details})==len(snapshot.process_details),"Process count includes duplicate PIDs"
        def check_hdr() -> None:
            import OpenGL.GL as gl
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER,view.targets[0][0])
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            raw=gl.glReadPixels(0,0,view.width,view.height,gl.GL_RGB,gl.GL_FLOAT)
            values=np.frombuffer(raw,dtype=np.float32)
            assert np.isfinite(values).all(),"Invalid HDR light values could create black tiles"
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER,0)
            gl.glReadBuffer(gl.GL_BACK)
        for mode in SCENES:
            state.select(mode)
            frame()
            assert glGetError()==0,f"GL error in {mode}"
            check_hdr()
        previous_clock=state.clock
        for light_time in (7.,25.,100.):
            state.clock=light_time
            frame()
            check_hdr()
        state.clock=previous_clock
        view.last_time=previous_clock
        flow=verify_geometry_and_flow(view)
        load_colors=verify_process_emission(view)
        # Exercise target reallocation with the same live GL context.
        window=Window.from_display_module()
        resized=[]
        for size in ((1280,720),(1600,900)):
            window.size=size
            pygame.event.pump()
            view._set_projection()
            for fbo,_,_,_ in view.targets:
                glBindFramebuffer(GL_FRAMEBUFFER,fbo)
                assert glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE
            glBindFramebuffer(GL_FRAMEBUFFER,0)
            frame()
            assert glGetError()==0,"GL error after resize"
            resized.append([view.width,view.height])
        state.paused=False
        state.select("read")
        state.reactive=True
        for _ in range(30):
            state.update(1/60)
            view.draw(present=False)
        glFinish()
        started=time.perf_counter()
        for _ in range(120):
            state.update(1/60)
            view.draw(present=False)
            glFinish()
        elapsed=time.perf_counter()-started
        assert glGetError()==0,"GL error after render benchmark"
        report={"renderer":glGetString(GL_RENDERER).decode(),
                "opengl":glGetString(GL_VERSION).decode(),"particles":view.particle_count,
                "light_pipeline":"4x MSAA RGBA16F HDR / two-scale GPU bloom / filmic composite",
                "animation_changes_pixels":True,"pause_freezes_art":True,
                "pause_max_pixel_difference":float(delta.max()),
                "pause_mean_pixel_difference":float(delta.mean()),
                "all_palettes_gl_error_free":True,"hdr_light_values_finite":True,"process_count_unique":True,
                "processes":len(snapshot.process_details),"internal_flow":flow,"load_colors":load_colors,"framebuffer_resize_checks":resized,
                "gpu_completed_render_fps":round(120/elapsed,1)}
        Path("artifacts/cinematic-verification.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))
    finally:
        view.close()


if __name__=="__main__":
    main()
