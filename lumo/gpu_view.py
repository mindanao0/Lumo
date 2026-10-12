"""OpenGL internal data currents, six device constellations, and live metrics."""
from __future__ import annotations

import ctypes
import math
from pathlib import Path
from collections import deque

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from .shaders import (BACKGROUND_FRAGMENT, BACKGROUND_VERTEX, BLOOM_FRAGMENT, COMPOSITE_FRAGMENT,
                      LINK_FRAGMENT, PARTICLE_FRAGMENT, PARTICLE_VERTEX, POST_VERTEX)
from .showcase import SCENES, ShowcaseState
from .telemetry import LocalTelemetry, Snapshot
from .activity import device_activity, machine_cpu_percent, motion_activity, flow_speed, program_flow_speed
from .process_sources import ProcessSources, SOURCE_COLORS
from .packets import PacketPool
from .core_rotation import CoreRotation
from .process_hud import paint_process_hud
from .connected import build_connected, draw_connected, paint_connected_hud

WIDTH, HEIGHT = 1600, 900
STATIONS = (
    (-1.20,.60,"01","SOURCE","PROGRAM"),
    (-1.36,.00,"02","SOURCE","PROGRAM"),
    (-1.20,-.57,"03","SOURCE","PROGRAM"),
    (1.22,.60,"04","SOURCE","PROGRAM"),
    (1.36,.00,"05","SOURCE","PROGRAM"),
    (1.20,-.57,"06","SOURCE","OTHER PROCESSES"),
) + tuple((1.36*math.cos(math.pi-i*math.tau/14),.73*math.sin(math.pi-i*math.tau/14),
           str(i+1),"SOURCE","PROGRAM") for i in range(6,14))

NODE_COLORS = SOURCE_COLORS
SOURCE_BANDS = (-.78,-.47,-.16,.16,.47,.78,-.62,-.31,0.,.31,.62,-.70,.05,.70)


class GpuView:
    def __init__(self, state: ShowcaseState, title: str = "LUMO · SYSTEM", *, max_apps: int = 5, fullscreen: bool = False, connected: bool = False) -> None:
        pygame.display.init()
        pygame.font.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_COMPATIBILITY)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        flags=pygame.OPENGL | pygame.DOUBLEBUF | (pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), flags, vsync=1)
        pygame.display.set_caption(title)
        self.state = state
        self.system_view = connected
        self.core_rotation = CoreRotation()
        self.device_clocks=np.zeros(6)
        self.device_speeds=np.full(6,.15)
        self.telemetry = LocalTelemetry()
        self.snapshot = Snapshot()
        self.program = compileProgram(compileShader(PARTICLE_VERTEX, GL_VERTEX_SHADER), compileShader(PARTICLE_FRAGMENT, GL_FRAGMENT_SHADER))
        self.background = compileProgram(compileShader(BACKGROUND_VERTEX, GL_VERTEX_SHADER), compileShader(BACKGROUND_FRAGMENT, GL_FRAGMENT_SHADER))
        self.links = compileProgram(compileShader(PARTICLE_VERTEX, GL_VERTEX_SHADER), compileShader(LINK_FRAGMENT, GL_FRAGMENT_SHADER))
        self.bloom = compileProgram(compileShader(POST_VERTEX, GL_VERTEX_SHADER), compileShader(BLOOM_FRAGMENT, GL_FRAGMENT_SHADER))
        self.composite = compileProgram(compileShader(POST_VERTEX, GL_VERTEX_SHADER), compileShader(COMPOSITE_FRAGMENT, GL_FRAGMENT_SHADER))
        self.bloom_uniforms = {name: glGetUniformLocation(self.bloom,name) for name in ("uSource","uStep","uExtract")}
        self.composite_uniforms = {name: glGetUniformLocation(self.composite,name) for name in ("uScene","uBloom","uWide")}
        self.targets = []
        self.scene_fbo = self.scene_rbo = 0
        names = ("uCanvasFraction", "uWorldScale", "uCoreRotation", "uTime", "uFlowTime", "uParticleTime", "uShapeActivity", "uShapeDrift", "uAspect", "uScale", "uGlow", "uKind", "uCenter", "uCool", "uWarm", "uNodeColor", "uReactive", "uConnected", "uLoads", "uEntry", "uBand", "uEmission", "uBirthTime", "uStopTime", "uCloudAlpha", "uCloudScale")
        self.uniforms = {name: glGetUniformLocation(self.program, name) for name in names}
        self.link_uniforms = {name: glGetUniformLocation(self.links, name) for name in names}
        self.back_uniforms = {name: glGetUniformLocation(self.background, name) for name in ("uAspect", "uCool", "uWarm")}
        self.buffers: dict[str, tuple[int, int]] = {}
        self.hud_texture = int(glGenTextures(1))
        self.hud_key = None
        self.metric_history={key:deque(maxlen=24) for key in ("cpu","gpu","ram","network","storage","disk")}
        self.fonts: dict[tuple, pygame.font.Font] = {}
        self.cool = np.array(SCENES[state.mode][1], dtype=np.float32)
        self.warm = np.array(SCENES[state.mode][2], dtype=np.float32)
        self.last_time = state.clock
        self.flow_time = 0.
        self.shape_activity = 0.
        self.shape_drift = 0.
        self.loads=np.zeros(6,dtype=np.float32)
        self.loads_initialized=False
        self.last_mode=state.mode
        self.process_sources=ProcessSources(max_apps=min(5,max_apps))
        self.packet_pools = {}
        self._make_particles()
        build_connected(self)
        glEnable(GL_BLEND)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)
        glDisable(GL_DEPTH_TEST)
        self._set_projection()

    def _set_projection(self) -> None:
        self.width, self.height = pygame.display.get_window_size()
        glViewport(0, 0, self.width, self.height)
        self.canvas_fraction = 1.
        self.world_scale = 1.
        self.aspect = self.width*self.canvas_fraction / self.height
        self.scale = self.height / 900
        self._make_targets()


    def _delete_targets(self) -> None:
        if self.targets:
            glDeleteTextures([t[1] for t in self.targets])
            glDeleteFramebuffers(len(self.targets),[t[0] for t in self.targets])
            self.targets=[]
        if self.scene_fbo:
            glDeleteFramebuffers(1,[self.scene_fbo])
            glDeleteRenderbuffers(1,[self.scene_rbo])
            self.scene_fbo=self.scene_rbo=0

    def _make_targets(self) -> None:
        self._delete_targets()
        # Floating point multisample target keeps bright sparks above 1.0.
        self.scene_fbo=int(glGenFramebuffers(1))
        self.scene_rbo=int(glGenRenderbuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER,self.scene_fbo)
        glBindRenderbuffer(GL_RENDERBUFFER,self.scene_rbo)
        glRenderbufferStorageMultisample(GL_RENDERBUFFER,4,GL_RGBA16F,self.width,self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_RENDERBUFFER,self.scene_rbo)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("HDR multisample framebuffer is incomplete")
        for width,height in ((self.width,self.height),
                             (max(1,self.width//4),max(1,self.height//4)),
                             (max(1,self.width//4),max(1,self.height//4)),
                             (max(1,self.width//12),max(1,self.height//12)),
                             (max(1,self.width//12),max(1,self.height//12))):
            texture=int(glGenTextures(1))
            glBindTexture(GL_TEXTURE_2D,texture)
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,width,height,0,GL_RGBA,GL_FLOAT,None)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE)
            fbo=int(glGenFramebuffers(1))
            glBindFramebuffer(GL_FRAMEBUFFER,fbo)
            glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,texture,0)
            if glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("HDR texture framebuffer is incomplete")
            self.targets.append((fbo,texture,width,height))
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        glBindRenderbuffer(GL_RENDERBUFFER,0)

    @staticmethod
    def _quad() -> None:
        glBegin(GL_QUADS)
        for x,y,u,v in ((-1,-1,0,0),(1,-1,1,0),(1,1,1,1),(-1,1,0,1)):
            glTexCoord2f(u,v)
            glVertex2f(x,y)
        glEnd()

    def _compose_light(self) -> None:
        glBindFramebuffer(GL_READ_FRAMEBUFFER,self.scene_fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER,self.targets[0][0])
        glBlitFramebuffer(0,0,self.width,self.height,0,0,self.width,self.height,GL_COLOR_BUFFER_BIT,GL_NEAREST)
        glDisable(GL_BLEND)
        glActiveTexture(GL_TEXTURE0)
        glUseProgram(self.bloom)
        glUniform1i(self.bloom_uniforms["uSource"],0)
        results=[]
        for pair in (self.targets[1:3],self.targets[3:5]):
            source=self.targets[0][1]
            for step in range(4):
                fbo,texture,width,height=pair[step%2]
                glBindFramebuffer(GL_FRAMEBUFFER,fbo)
                glViewport(0,0,width,height)
                glBindTexture(GL_TEXTURE_2D,source)
                glUniform1i(self.bloom_uniforms["uExtract"],int(step==0))
                glUniform2f(self.bloom_uniforms["uStep"],1.0/width if step%2==0 else 0.,0. if step%2==0 else 1.0/height)
                self._quad()
                source=texture
            results.append(source)
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        glViewport(0,0,self.width,self.height)
        glUseProgram(self.composite)
        for unit,(name,texture) in enumerate(zip(("uScene","uBloom","uWide"),(self.targets[0][1],*results))):
            glActiveTexture(GL_TEXTURE0+unit)
            glBindTexture(GL_TEXTURE_2D,texture)
            glUniform1i(self.composite_uniforms[name],unit)
        self._quad()
        glActiveTexture(GL_TEXTURE0)
        glUseProgram(0)
        glEnable(GL_BLEND)

    def _upload(self, name: str, positions: np.ndarray, colors: np.ndarray) -> None:
        data = np.ascontiguousarray(np.column_stack((positions, colors)), dtype=np.float32)
        buffer = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, buffer)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        self.buffers[name] = (buffer, len(data))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _make_particles(self) -> None:
        rng = np.random.default_rng(1948)
        # A packet's phase describes its entire source -> entry -> crossing lifetime.
        # The core is composed of these particles; there is no independent ring mesh.
        for index in range(6):
            emitter=np.random.default_rng(1948+index*127)
            count=15000
            data=np.column_stack((emitter.random(count),emitter.uniform(-1.,1.,count),np.zeros(count)))
            colors=np.column_stack((emitter.random(count),emitter.uniform(.12,.42,count),emitter.random(count)))
            self._upload(f"process_dust_{index}",data,colors)
            count=90
            heads=np.column_stack((emitter.random(count),emitter.uniform(-1.,1.,count),np.zeros(count)))
            shades=np.column_stack((emitter.random(count),emitter.uniform(.45,.90,count),emitter.random(count)))
            self._upload(f"process_head_{index}",heads,shades)
            wakes=np.repeat(heads,20,axis=0)
            wakes[:,2]=np.tile(np.linspace(.03,1.,20),count)
            self._upload(f"process_wake_{index}",wakes,np.repeat(shades,20,axis=0))

        # Each subsystem has its own branched structure and pockets of luminous dust.
        for index in range(6):
            local_rng=np.random.default_rng(730+index*121)
            anchors=local_rng.normal(0,.23,(5,3))
            anchors-=anchors.mean(axis=0)
            anchors[:,0]*=(1.1,.9,1.3,1.0,1.25,1.2)[index]
            anchors[:,1]*=(1.3,.95,.85,1.15,1.,.8)[index]
            choices=local_rng.integers(0,5,1100)
            cloud=anchors[choices]+local_rng.normal(0,.33,(1100,3))
            # Filament bridges keep the five dust pockets in one connected group.
            bridges=[]
            for pocket in range(1,5):
                parent=int(np.argmin(np.linalg.norm(anchors[:pocket]-anchors[pocket],axis=1)))
                for t in np.linspace(0,1,22):
                    bridges.append((1-t)*anchors[pocket]+t*anchors[parent]+local_rng.normal(0,.032,3))
            cloud=np.vstack((cloud,bridges))
            cloud*=np.minimum(1.,1.45/np.maximum(np.linalg.norm(cloud,axis=1),.001))[:,None]
            seeds=local_rng.random(len(cloud))
            self._upload(f"station_{index}",cloud,np.column_stack((seeds,local_rng.uniform(.25,.85,len(cloud)),seeds)))
            distance=np.linalg.norm(cloud[:,None,:]-cloud[None,:,:],axis=-1)
            np.fill_diagonal(distance,np.inf)
            pairs=set()
            for i in range(len(cloud)):
                for j in np.argsort(distance[i])[:4]:
                    if distance[i,j]<.55:
                        pairs.add((min(i,int(j)),max(i,int(j))))
            edges=np.array([cloud[k] for pair in sorted(pairs) for k in pair])
            edge_seeds=np.array([seeds[k] for pair in sorted(pairs) for k in pair])
            self._upload(f"station_edges_{index}",edges,np.column_stack((edge_seeds,np.full(len(edges),.16),edge_seeds)))
            dust=cloud[local_rng.integers(0,len(cloud),5600)]+local_rng.normal(0,.09,(5600,3))
            seeds=local_rng.random(len(dust))
            self._upload(f"station_dust_{index}",dust,np.column_stack((seeds,local_rng.uniform(.04,.18,len(dust)),seeds)))
            hubs=anchors+local_rng.normal(0,.13,(5,3))
            hubs=np.vstack((hubs,cloud[local_rng.choice(len(cloud),22,replace=False)]))
            hub_seeds=local_rng.random(len(hubs))
            self._upload(f"station_sparks_{index}",hubs,np.column_stack((np.ones(len(hubs)),local_rng.uniform(.6,1.,len(hubs)),hub_seeds)))

        ambient=rng.normal(0,1,(5000,3))
        ambient/=np.maximum(np.linalg.norm(ambient,axis=1),.001)[:,None]
        ambient*=rng.random((5000,1))**.5
        ambient*=np.array((.53,.53,.46))
        seeds=rng.random(len(ambient))
        self._upload("core_ambient",ambient,np.column_stack((seeds,np.full(len(ambient),.10),seeds)))
        count=3400
        stars=np.column_stack((rng.uniform(-1.9,1.9,count),rng.uniform(-.79,.96,count),rng.random(count)))
        self._upload("stars",stars,np.column_stack((rng.random(count),rng.uniform(.05,.38,count),rng.random(count))))

    @property
    def particle_count(self) -> int:
        if self.system_view:
            return (sum(self.buffers[f"{name}_{slot}"][1] for slot in range(6)
                        for name in ("process_dust","process_head","process_wake","station","station_dust","station_sparks"))
                    +sum(self.buffers[name][1] for name in ("stars","connected_dust","connected_heads")))
        return (sum(self.buffers[f"{name}_{source.slot%6}"][1] for source in self.process_sources.sources
                    for name in ("station","station_dust","station_sparks"))
                +sum(count for key,(_,count) in self.buffers.items() if key.startswith("packets_"))
                +self.buffers["stars"][1]+self.buffers["core_ambient"][1])

    def _configure(self, uniforms: dict, now: float) -> None:
        glUniformMatrix3fv(uniforms["uCoreRotation"],1,GL_FALSE,self.core_rotation.matrix.T.copy())
        glUniform1f(uniforms["uTime"],now)
        glUniform1f(uniforms["uFlowTime"],self.flow_time)
        glUniform1f(uniforms["uParticleTime"],self.flow_time)
        glUniform1f(uniforms["uShapeActivity"],self.shape_activity)
        glUniform1f(uniforms["uShapeDrift"],self.shape_drift)
        glUniform1f(uniforms["uCanvasFraction"],self.canvas_fraction)
        glUniform1f(uniforms["uWorldScale"],self.world_scale)
        glUniform1f(uniforms["uAspect"],self.aspect)
        glUniform1f(uniforms["uScale"],self.scale)
        glUniform1f(uniforms["uGlow"],0.)
        glUniform3f(uniforms["uCool"],*self.cool)
        glUniform3f(uniforms["uWarm"],*self.warm)
        glUniform3f(uniforms["uNodeColor"],*self.cool)
        glUniform2f(uniforms["uCenter"],0.,0.)
        glUniform1i(uniforms["uReactive"],int(self.state.reactive))
        glUniform1i(uniforms["uConnected"],int(self.system_view))
        glUniform1fv(uniforms["uLoads"],6,self.loads)
        glUniform1f(uniforms["uEntry"],2.6)
        glUniform1f(uniforms["uBand"],-.78)
        glUniform1f(uniforms["uEmission"],1.)
        glUniform1f(uniforms["uBirthTime"],-40.)
        glUniform1f(uniforms["uStopTime"],1.e10)
        glUniform1f(uniforms["uCloudAlpha"],1.)
        glUniform1f(uniforms["uCloudScale"],.235)

    def _points(self, name: str, kind: int, glow: bool, center=(0.,0.)) -> None:
        glUniform1i(self.uniforms["uKind"],kind)
        glUniform1f(self.uniforms["uGlow"],float(glow))
        glUniform2f(self.uniforms["uCenter"],*center)
        self._draw_buffer(name,GL_POINTS)

    def _draw_buffer(self, name: str, mode: int) -> None:
        buffer,count=self.buffers[name]
        glBindBuffer(GL_ARRAY_BUFFER,buffer)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3,GL_FLOAT,24,ctypes.c_void_p(0))
        glColorPointer(3,GL_FLOAT,24,ctypes.c_void_p(12))
        glDrawArrays(mode,0,count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER,0)

    def draw(self, *, present: bool = True) -> None:
        old_reading=self.snapshot.updated
        self.snapshot=self.telemetry.update()
        if self.snapshot.updated!=old_reading:
            self.metric_history["cpu"].append(self.snapshot.cpu/100.)
            self.metric_history["gpu"].append((self.snapshot.gpu or 0.)/100.)
            self.metric_history["ram"].append(self.snapshot.ram_used/max(1.,self.snapshot.ram_total))
            self.metric_history["storage"].append(self.snapshot.storage_percent/100.)
            self.metric_history["network"].append(self.snapshot.net_down+self.snapshot.net_up)
            self.metric_history["disk"].append(self.snapshot.disk_read+self.snapshot.disk_write)
        now=self.state.clock
        dt=max(0.,now-self.last_time)
        self.last_time=now
        self.core_rotation.step(dt,now)
        blend=1.-math.exp(-dt*1.8)
        if self.state.paused and self.state.mode!=self.last_mode:
            blend=1.
        self.last_mode=self.state.mode
        initializing=not self.loads_initialized
        target=np.array(device_activity(self.snapshot),dtype=np.float32)
        if not self.loads_initialized or (self.state.paused and self.snapshot.updated!=old_reading):
            self.loads[:]=target
            self.loads_initialized=True
        else:
            self.loads+=(target-self.loads)*blend
        # Integrate speed so live load changes never teleport the moving packets.
        activity=motion_activity(self.loads)
        self.flow_time+=dt*flow_speed(activity)
        targets=np.array([flow_speed(float(v)) for v in self.loads])
        targets[3]=.20  # Resident RAM changes density, not packet speed.
        self.device_speeds+=(targets-self.device_speeds)*(1.-math.exp(-dt/1.2))
        self.device_clocks+=dt*self.device_speeds
        if initializing or dt>0.:
            self.shape_activity=float(activity)
            self.shape_drift=math.atan2(float(self.loads[0]-self.loads[4])+.01,float(self.loads[1]-self.loads[5])+.01)
        cores=max(1,len(self.snapshot.cores))
        grouped={}
        for process in self.snapshot.process_details:
            grouped[process.name]=grouped.get(process.name,0.)+max(0.,process.cpu)
        speeds={name:program_flow_speed(machine_cpu_percent(cpu,cores)/100.)
                for name,cpu in grouped.items()}
        # Existing named sources respond to their own measured CPU. Unattributed I/O
        # contributes only to the background source and the shared central current.
        named=set(self.process_sources._live) if self.process_sources._initialized else set(
            sorted(grouped,key=lambda name:grouped[name],reverse=True)[:self.process_sources.max_apps])
        background_cpu=sum(cpu for name,cpu in grouped.items() if name not in named)
        background=max(machine_cpu_percent(background_cpu,cores)/100.,.25*max(self.loads[1],self.loads[5]))
        speeds["other processes"]=program_flow_speed(background)
        if not self.system_view:
            self.process_sources.update(self.snapshot.process_details,now=now,flow_time=self.flow_time,
                                        sampled_at=self.snapshot.updated,instant_emission=initializing,logical_cpus=cores,
                                        source_speeds=speeds)
        self.cool+=(np.array(SCENES[self.state.mode][1])-self.cool)*blend
        self.warm+=(np.array(SCENES[self.state.mode][2])-self.warm)*blend
        glBindFramebuffer(GL_FRAMEBUFFER,self.scene_fbo)
        glViewport(0,0,self.width,self.height)
        glDisable(GL_BLEND)
        glUseProgram(self.background)
        glUniform1f(self.back_uniforms["uAspect"],self.aspect)
        glUniform3f(self.back_uniforms["uCool"],*self.cool)
        glUniform3f(self.back_uniforms["uWarm"],*self.warm)
        glBegin(GL_QUADS)
        for x,y in ((-1,-1),(1,-1),(1,1),(-1,1)):
            glVertex2f(x,y)
        glEnd()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE)
        glEnable(GL_POINT_SPRITE)
        glUseProgram(self.program)
        self._configure(self.uniforms,now)
        self._points("stars",3,False)
        if self.system_view:
            draw_connected(self,now)
        else:
            active={id(source) for source in self.process_sources.sources}
            for identity in list(self.packet_pools):
                if identity not in active:
                    for kind in ("head","wake","dust"):
                        key=f"packets_{identity}_{kind}"
                        if key in self.buffers:
                            glDeleteBuffers(1,[self.buffers.pop(key)[0]])
                    del self.packet_pools[identity]
            for source in self.process_sources.sources:
                identity=id(source)
                if identity not in self.packet_pools:
                    self.packet_pools[identity]=PacketPool(source)
                pool=self.packet_pools[identity]
                if pool.update(source,dt,now):
                    for kind,(positions,colors) in pool.arrays().items():
                        key=f"packets_{identity}_{kind}"
                        if key in self.buffers:
                            glDeleteBuffers(1,[self.buffers.pop(key)[0]])
                        self._upload(key,positions,colors)
            for glow in (True,False):
                self._points("core_ambient",32,glow)
                for source in self.process_sources.sources:
                    slot=source.slot
                    index=slot%6
                    x,y=STATIONS[slot][:2]
                    glUniform3f(self.uniforms["uNodeColor"],*source.color)
                    glUniform1f(self.uniforms["uEntry"],math.atan2(y,x+.025))
                    glUniform1f(self.uniforms["uBand"],SOURCE_BANDS[slot])
                    glUniform1f(self.uniforms["uEmission"],source.render_emission)
                    glUniform1f(self.uniforms["uParticleTime"],source.flow_clock)
                    glUniform1f(self.uniforms["uBirthTime"],source.born_flow)
                    glUniform1f(self.uniforms["uStopTime"],source.stopped_flow if source.stopped_flow is not None else 1.e10)
                    glUniform1f(self.uniforms["uCloudAlpha"],source.opacity)
                    count=max(1,len(self.process_sources._live)+1)
                    cloud_scale=.19 if count<=6 else .125
                    resident=min(1.,source.memory/max(1.,self.snapshot.ram_total))
                    glUniform1f(self.uniforms["uCloudScale"],cloud_scale*(.80+1.1*math.sqrt(resident)))
                    for kind,shader in (("dust",29),("wake",31),("head",30)):
                        key=f"packets_{id(source)}_{kind}"
                        if key in self.buffers:self._points(key,shader,glow,(x,y))
                    self._points(f"station_dust_{index}",11,glow,(x,y))
                    self._points(f"station_{index}",11,glow,(x,y))
                    self._points(f"station_sparks_{index}",14,glow,(x,y))
        glUseProgram(0)
        glDisable(GL_POINT_SPRITE)
        self._compose_light()
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        self._draw_hud()
        if present:
            pygame.display.flip()

    def _font(self, size: int, bold: bool = False, family: str = "mono") -> pygame.font.Font:
        scaled=max(8,round(size*self.scale))
        key=(scaled,bold,family)
        if key not in self.fonts:
            preferred={
                "ui":Path("/usr/share/fonts/rsms-inter-fonts")/("Inter-Bold.ttf" if bold else "Inter-Regular.ttf"),
                "data":Path("/usr/share/fonts/adobe-source-code-pro-fonts")/("SourceCodePro-Semibold.otf" if bold else "SourceCodePro-Regular.otf"),
            }.get(family)
            if preferred is not None and preferred.is_file():
                self.fonts[key]=pygame.font.Font(str(preferred),scaled)
            else:
                self.fonts[key]=pygame.font.SysFont({"ui":"Inter,DejaVu Sans","data":"Source Code Pro,DejaVu Sans Mono","mono":"DejaVu Sans Mono"}[family],scaled,bold=bold)
        return self.fonts[key]

    def _text(self, text: str, x: float, y: float, size: int = 12,
              color=(163,189,207), bold=False, align="left", spacing: int = 0, opacity: float = 1., family: str = "mono") -> None:
        font=self._font(size,bold,family)
        if spacing:
            glyphs=[font.render(char,True,color) for char in text]
            gap=max(1,round(spacing*self.scale))
            width=sum(g.get_width() for g in glyphs)+gap*(len(glyphs)-1)
            surface=pygame.Surface((width,font.get_height()),pygame.SRCALPHA)
            offset=0
            for glyph in glyphs:
                surface.blit(glyph,(offset,0))
                offset+=glyph.get_width()+gap
        else:
            surface=font.render(text,True,color)
        tw,th=surface.get_size()
        if opacity<1.:
            surface.set_alpha(round(255*opacity))
        x=x*self.width-(tw/2 if align=="center" else tw if align=="right" else 0)
        self.hud_surface.blit(surface,(round(x),round(y*self.height)))

    def _line(self, start, end, color=(123,167,184,150), width=1) -> None:
        pygame.draw.line(self.hud_surface,color,(round(start[0]*self.width),round(start[1]*self.height)),
                         (round(end[0]*self.width),round(end[1]*self.height)),max(1,round(width*self.scale)))

    def _draw_hud(self) -> None:
        transitioning=any(0.<source.opacity<1. for source in self.process_sources.sources)
        key=(self.width,self.height,self.state.mode,self.state.reactive,self.state.paused,self.core_rotation.auto,self.system_view,self.snapshot.updated,getattr(self,"capture_label",""),
             int(self.state.clock*20) if transitioning else None)
        if key!=self.hud_key:
            self.hud_surface=pygame.Surface((self.width,self.height),pygame.SRCALPHA)
            self._paint_hud()
            glBindTexture(GL_TEXTURE_2D,self.hud_texture)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,self.width,self.height,0,GL_RGBA,GL_UNSIGNED_BYTE,pygame.image.tobytes(self.hud_surface,"RGBA",True))
            self.hud_key=key
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0,self.width,self.height,0,-1,1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D,self.hud_texture)
        glColor4f(1,1,1,1)
        glBegin(GL_QUADS)
        for tx,ty,px,py in ((0,1,0,0),(1,1,self.width,0),(1,0,self.width,self.height),(0,0,0,self.height)):
            glTexCoord2f(tx,ty)
            glVertex2f(px,py)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def _sparkline(self, key: str, x: float, y: float, width: float = .112) -> None:
        values=list(self.metric_history[key])
        self._line((x,y),(x+width,y),(42,65,77,170))
        if len(values)<2:
            return
        peak=max(max(values),float(2**20))
        # This is recent measured throughput, scaled to its own peak, not utilisation.
        points=[(x+(24-len(values)+i)*width/23,y-.017*value/peak) for i,value in enumerate(values)]
        for first,last in zip(points,points[1:]):
            self._line(first,last,(117,171,186,190))

    def _paint_hud(self) -> None:
        if self.system_view:
            paint_connected_hud(self)
            return
        paint_process_hud(self)



    def handle_event(self, event) -> None:
        if self.system_view:
            return
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            x,y=event.pos
            core_x=self.width*.5-self.height*.0125
            if (x-core_x)**2+(y-self.height*.5)**2<(self.height*.30)**2:
                self.core_rotation.begin(event.pos,self.width,self.height)
        elif event.type==pygame.MOUSEMOTION:
            self.core_rotation.drag(event.pos,self.width,self.height)
        elif event.type==pygame.MOUSEBUTTONUP and event.button==1:
            self.core_rotation.drag_start=None
        elif event.type==pygame.WINDOWFOCUSLOST:
            self.core_rotation.drag_start=None
        elif event.type==pygame.KEYDOWN:
            if event.key==pygame.K_r:
                self.core_rotation.reset()
            elif event.key==pygame.K_o:
                self.core_rotation.auto=not self.core_rotation.auto

    def close(self) -> None:
        self._delete_targets()
        glDeleteTextures([self.hud_texture])
        glDeleteBuffers(len(self.buffers),[b for b,_ in self.buffers.values()])
        glDeleteProgram(self.program)
        glDeleteProgram(self.background)
        glDeleteProgram(self.links)
        glDeleteProgram(self.bloom)
        glDeleteProgram(self.composite)
        self.telemetry.close()
        pygame.quit()
