"""OpenGL internal data currents, six device constellations, and live metrics."""
from __future__ import annotations

import ctypes
import math

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from .shaders import (BACKGROUND_FRAGMENT, BACKGROUND_VERTEX, BLOOM_FRAGMENT, COMPOSITE_FRAGMENT,
                      LINK_FRAGMENT, PARTICLE_FRAGMENT, PARTICLE_VERTEX, POST_VERTEX)
from .showcase import SCENES, ShowcaseState
from .telemetry import LocalTelemetry, Snapshot
from .activity import device_activity
from .process_sources import ProcessSources, SOURCE_COLORS

WIDTH, HEIGHT = 1600, 900
STATIONS = (
    (-1.20, .60, "01", "SOURCE", "PROGRAM"),
    (-1.36, .00, "02", "SOURCE", "PROGRAM"),
    (-1.20, -.57, "03", "SOURCE", "PROGRAM"),
    (1.22, .60, "04", "SOURCE", "PROGRAM"),
    (1.36, .00, "05", "SOURCE", "PROGRAM"),
    (1.20, -.57, "06", "SOURCE", "OTHER PROCESSES"),
    (-1.52,.30,"07","SOURCE","PROGRAM"),
    (-1.52,-.29,"08","SOURCE","PROGRAM"),
    (1.52,.30,"09","SOURCE","PROGRAM"),
    (1.52,-.29,"10","SOURCE","PROGRAM"),
    (-.52,.90,"11","SOURCE","PROGRAM"),
    (.52,.90,"12","SOURCE","PROGRAM"),
    (-.52,-.90,"13","SOURCE","PROGRAM"),
    (.52,-.90,"14","SOURCE","PROGRAM"),
)
NODE_COLORS = SOURCE_COLORS
SOURCE_BANDS = (-.78,-.47,-.16,.16,.47,.78,-.62,-.31,0.,.31,.62,-.70,.05,.70)


class GpuView:
    def __init__(self, state: ShowcaseState, title: str = "LUMO · SYSTEM", *, max_apps: int = 13, fullscreen: bool = False) -> None:
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
        names = ("uTime", "uFlowTime", "uAspect", "uScale", "uGlow", "uKind", "uCenter", "uCool", "uWarm", "uNodeColor", "uReactive", "uLoads", "uEntry", "uBand", "uEmission", "uBirthTime", "uStopTime", "uCloudAlpha", "uCloudScale")
        self.uniforms = {name: glGetUniformLocation(self.program, name) for name in names}
        self.link_uniforms = {name: glGetUniformLocation(self.links, name) for name in names}
        self.back_uniforms = {name: glGetUniformLocation(self.background, name) for name in ("uAspect", "uCool", "uWarm")}
        self.buffers: dict[str, tuple[int, int]] = {}
        self.hud_texture = int(glGenTextures(1))
        self.hud_key = None
        self.fonts: dict[tuple[int, bool], pygame.font.Font] = {}
        self.cool = np.array(SCENES[state.mode][1], dtype=np.float32)
        self.warm = np.array(SCENES[state.mode][2], dtype=np.float32)
        self.last_time = state.clock
        self.flow_time = 0.
        self.loads=np.zeros(6,dtype=np.float32)
        self.loads_initialized=False
        self.last_mode=state.mode
        self.process_sources=ProcessSources(max_apps=max_apps)
        self._make_particles()
        glEnable(GL_BLEND)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)
        glDisable(GL_DEPTH_TEST)
        self._set_projection()

    def _set_projection(self) -> None:
        self.width, self.height = pygame.display.get_window_size()
        glViewport(0, 0, self.width, self.height)
        self.aspect = self.width / self.height
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
        # A packet's phase describes its entire source -> entry -> circulation lifetime.
        # The core is composed of these particles; there is no independent ring mesh.
        for index in range(6):
            emitter=np.random.default_rng(1948+index*127)
            count=15000
            data=np.column_stack((emitter.random(count),emitter.uniform(-.14,.14,count),np.zeros(count)))
            colors=np.column_stack((emitter.random(count),emitter.uniform(.12,.42,count),emitter.random(count)))
            self._upload(f"process_dust_{index}",data,colors)
            count=180
            heads=np.column_stack((emitter.random(count),emitter.uniform(-.14,.14,count),np.zeros(count)))
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

        count=3400
        stars=np.column_stack((rng.uniform(-1.9,1.9,count),rng.uniform(-.79,.96,count),rng.random(count)))
        self._upload("stars",stars,np.column_stack((rng.random(count),rng.uniform(.05,.38,count),rng.random(count))))

        # The central aperture contains only a faceted crystal and faint orbital light.
        top=(0,.090,0); bottom=(0,-.090,0)
        rim=[(.048,0,0),(0,0,.048),(-.048,0,0),(0,0,-.048)]
        diamond=np.array([p for i in range(4) for pair in ((top,rim[i]),(bottom,rim[i]),(rim[i],rim[(i+1)%4])) for p in pair])
        self._upload("diamond",diamond,np.tile((.6,.66,.75),(len(diamond),1)))

        facets=np.array([p for i in range(4) for tri in ((top,rim[i],rim[(i+1)%4]),(bottom,rim[(i+1)%4],rim[i])) for p in tri])
        self._upload("diamond_faces",facets,np.tile((.6,.18,.75),(len(facets),1)))
        ring=np.array([(math.cos(a)*r,math.sin(a)*r*.32,0) for r in (.13,.18) for a in np.linspace(0,math.tau,240)])
        ring=ring.reshape(2,240,3)
        ring=np.stack((ring[:,:-1],ring[:,1:]),axis=-2).reshape(-1,3)
        self._upload("seed_orbit",ring,np.tile((.6,.30,.75),(len(ring),1)))

    @property
    def particle_count(self) -> int:
        return (sum(self.buffers[f"{name}_{source.slot%6}"][1] for source in self.process_sources.sources
                    for name in ("process_dust","process_head","process_wake","station","station_dust","station_sparks"))
                +self.buffers["stars"][1])

    def _configure(self, uniforms: dict, now: float) -> None:
        glUniform1f(uniforms["uTime"],now)
        glUniform1f(uniforms["uFlowTime"],self.flow_time)
        glUniform1f(uniforms["uAspect"],self.aspect)
        glUniform1f(uniforms["uScale"],self.scale)
        glUniform1f(uniforms["uGlow"],0.)
        glUniform3f(uniforms["uCool"],*self.cool)
        glUniform3f(uniforms["uWarm"],*self.warm)
        glUniform3f(uniforms["uNodeColor"],*self.cool)
        glUniform2f(uniforms["uCenter"],0.,0.)
        glUniform1i(uniforms["uReactive"],int(self.state.reactive))
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
        now=self.state.clock
        dt=max(0.,now-self.last_time)
        self.last_time=now
        blend=1.-math.exp(-dt*1.8)
        if self.state.paused and self.state.mode!=self.last_mode:
            blend=1.
        self.last_mode=self.state.mode
        self.process_sources.update(self.snapshot.process_details,now=now,flow_time=self.flow_time,
                                    sampled_at=self.snapshot.updated,instant_emission=not self.loads_initialized)
        target=np.array(device_activity(self.snapshot),dtype=np.float32)
        if not self.loads_initialized or (self.state.paused and self.snapshot.updated!=old_reading):
            self.loads[:]=target
            self.loads_initialized=True
        else:
            self.loads+=(target-self.loads)*blend
        # Integrate speed so live load changes never teleport the moving packets.
        self.flow_time+=dt*(.85+.80*float(self.loads.mean()))
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
        glDisable(GL_POINT_SPRITE)
        glUseProgram(self.links)
        self._configure(self.link_uniforms,now)
        glLineWidth(1.)
        glUniform1i(self.link_uniforms["uKind"],9)
        self._draw_buffer("diamond_faces",GL_TRIANGLES)
        self._draw_buffer("diamond",GL_LINES)

        glEnable(GL_POINT_SPRITE)
        glUseProgram(self.program)
        self._configure(self.uniforms,now)
        self._points("stars",3,False)
        for glow in (True,False):
            for source in self.process_sources.sources:
                slot=source.slot
                index=slot%6
                x,y=STATIONS[slot][:2]
                glUniform3f(self.uniforms["uNodeColor"],*source.color)
                glUniform1f(self.uniforms["uEntry"],math.atan2(y,x+.025))
                glUniform1f(self.uniforms["uBand"],SOURCE_BANDS[slot])
                glUniform1f(self.uniforms["uEmission"],source.render_emission)
                glUniform1f(self.uniforms["uBirthTime"],source.born_flow)
                glUniform1f(self.uniforms["uStopTime"],source.stopped_flow if source.stopped_flow is not None else 1.e10)
                glUniform1f(self.uniforms["uCloudAlpha"],source.opacity)
                glUniform1f(self.uniforms["uCloudScale"],.235 if slot<6 else .12 if slot<10 else .065)
                self._points(f"process_dust_{index}",18,glow,(x,y))
                self._points(f"process_wake_{index}",20,glow,(x,y))
                self._points(f"process_head_{index}",19,glow,(x,y))
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

    def _font(self, size: int, bold: bool = False) -> pygame.font.Font:
        scaled=max(8,round(size*self.scale))
        key=(scaled,bold)
        if key not in self.fonts:
            self.fonts[key]=pygame.font.SysFont("DejaVu Sans Mono",scaled,bold=bold)
        return self.fonts[key]

    def _text(self, text: str, x: float, y: float, size: int = 12,
              color=(163,189,207), bold=False, align="left", spacing: int = 0, opacity: float = 1.) -> None:
        font=self._font(size,bold)
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
        key=(self.width,self.height,self.state.mode,self.state.reactive,self.snapshot.updated,
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

    def _paint_hud(self) -> None:
        snap=self.snapshot
        self._text("LUMO / PROCESS FLOW",.022,.023,22,(222,236,245),spacing=4)
        self._text("FEDORA · LIVE METRICS / ILLUSTRATIVE FLOW",.023,.068,9,(126,167,190),spacing=1)
        gpu=f"{snap.gpu:.0f}%" if snap.gpu is not None else "—"
        self._text(f"CPU {snap.cpu:.0f}%   GPU {gpu}   RAM {snap.ram_used/2**30:.1f}/{snap.ram_total/2**30:.1f} GiB",.975,.024,13,(178,211,227),align="right")
        self._text(f"NET ↓{snap.net_down/2**20:.1f} ↑{snap.net_up/2**20:.1f} MiB/s   DISK {snap.storage_percent:.0f}%",.975,.053,11,(112,161,183),align="right")
        labels=((.025,.137,"left"),(.023,.488,"left"),(.025,.824,"left"),
                (.975,.137,"right"),(.976,.488,"right"),(.975,.824,"right"))
        for source in self.process_sources.sources:
            if source.opacity<.01:
                continue
            slot=source.slot
            if slot<6:
                px,py,align=labels[slot]
            elif slot<10:
                px,py,align=(.02 if slot<8 else .98),(.275 if slot in (6,8) else .646),("left" if slot<8 else "right")
            else:
                x,y=STATIONS[slot][:2]
                px,py,align=(1.+x/self.aspect)/2.+.035,(1.-y)/2.-.023,"left"
            color=tuple(round(c*255) for c in source.color)
            title=source.name if len(source.name)<=20 else source.name[:19]+"…"
            self._text(title,px,py,17 if slot<6 else 12,(213,231,240),align=align,opacity=source.opacity)
            self._text(f"CPU {source.cpu:.1f}%",px,py+(.028 if slot<6 else .021),20 if slot<6 else 12,color,align=align,opacity=source.opacity)
            if slot<6:
                self._text(f"{source.memory/2**20:.0f} MiB · {len(source.pids)} PID",px,py+.057,10,(129,163,185),align=align,opacity=source.opacity)


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
