"""Device constellations and a restrained spiral point cloud."""
import math
import numpy as np
import pygame
from OpenGL.GL import *
from .process_sources import SOURCE_COLORS

POSITIONS=((-.95,.60),(-1.18,.08),(-.87,-.43),(.98,.60),(1.20,.08),(.93,-.43))
NAMES=("CPU","NETWORK","PROCESSES","RAM","GPU","STORAGE")


def build_connected(view):
    rng=np.random.default_rng(20261012)
    # Loose spiral arms share the point-and-link language of the device clouds.
    n=7800
    arms=rng.integers(0,6,n)
    phase=arms*math.tau/6+rng.normal(0,.12,n)
    phase[:n//5]=rng.uniform(0,math.tau,n//5)
    radius=rng.beta(1.5,1.5,n)
    radius[:n//5]=rng.beta(.7,2.,n//5)
    points=np.column_stack((phase,2*radius-1,rng.normal(1.,.07,n)))
    view._upload("connected_dust",points,
                 np.column_stack((rng.random(n),rng.uniform(.12,.40,n),rng.random(n))))
    # Fine local links, rather than a continuous sheet or an outlined ring.
    samples=points[::12]
    r=.035+.48*((samples[:,1]+1)*.5)**.82
    angle=samples[:,0]+r*7.
    positions=np.column_stack((r*np.cos(angle),r*np.sin(angle)*.65))
    distance=np.sum((positions[:,None,:]-positions[None,:,:])**2,axis=2)
    np.fill_diagonal(distance,np.inf)
    edges=[]
    for i in range(len(samples)):
        for j in np.argsort(distance[i])[:3]:
            if j>i and distance[i,j]<.004:
                edges.extend((samples[i],samples[j]))
    view._upload("connected_core",np.array(edges),np.tile((.5,.10,.5),(len(edges),1)))
    n=64
    view._upload("connected_heads",np.column_stack((rng.integers(0,6,n)*math.tau/6,
                 rng.uniform(-1,1,n),np.ones(n))),
                 np.column_stack((rng.random(n),rng.uniform(.18,.42,n),rng.random(n))))
    paths=[]; colors=[]
    for lane in np.linspace(-.88,.88,16):
        seed=float(rng.random())
        points=np.array([(a,lane,0.) for a in np.linspace(0.,1.,81)])
        pairs=np.stack((points[:-1],points[1:]),axis=1).reshape(-1,3)
        paths.extend(pairs)
        colors.extend([(seed,.11,seed)]*len(pairs))
    view._upload("connected_inlets",np.array(paths),np.array(colors))


def _configure_device(view,uniforms,slot):
    x,y=POSITIONS[slot]
    glUniform2f(uniforms["uCenter"],x,y)
    glUniform3f(uniforms["uNodeColor"],*SOURCE_COLORS[slot])
    glUniform1f(uniforms["uBand"],(-.78,-.47,-.16,.16,.47,.78)[slot])
    glUniform1f(uniforms["uParticleTime"],float(view.device_clocks[slot]))
    glUniform1f(uniforms["uBirthTime"],-40.)
    glUniform1f(uniforms["uStopTime"],1.e10)
    glUniform1f(uniforms["uCloudAlpha"],1.)
    glUniform1f(uniforms["uCloudScale"],.185)
    load=float(view.loads[slot])
    if slot==2:
        count=len(view.snapshot.process_details)
        load=count/(count+300.)
    glUniform1f(uniforms["uCloudAlpha"],.45+.55*load)
    glUniform1f(uniforms["uEmission"],.10+.72*load)


def draw_connected(view,now):
    glDisable(GL_POINT_SPRITE)
    glUseProgram(view.links)
    view._configure(view.link_uniforms,now)
    glLineWidth(1.)
    glUniform1i(view.link_uniforms["uKind"],21)
    view._draw_buffer("connected_core",GL_LINES)
    for slot in range(6):
        _configure_device(view,view.link_uniforms,slot)
        glUniform1i(view.link_uniforms["uKind"],24)
        view._draw_buffer("connected_inlets",GL_LINES)
        glUniform1i(view.link_uniforms["uKind"],11)
        view._draw_buffer(f"station_edges_{slot}",GL_LINES)
    glEnable(GL_POINT_SPRITE)
    glUseProgram(view.program)
    view._configure(view.uniforms,now)
    for glow in (True,False):
        view._points("connected_dust",22,glow)
        view._points("connected_heads",23,glow)
        for slot in range(6):
            _configure_device(view,view.uniforms,slot)
            centre=POSITIONS[slot]
            view._points(f"process_dust_{slot}",18,glow,centre)
            view._points(f"process_wake_{slot}",20,glow,centre)
            view._points(f"process_head_{slot}",19,glow,centre)
            view._points(f"station_dust_{slot}",11,glow,centre)
            view._points(f"station_{slot}",11,glow,centre)
            view._points(f"station_sparks_{slot}",14,glow,centre)


def paint_connected_hud(view):
    snap=view.snapshot
    quiet=(135,158,170); white=(210,227,237)
    view._text("LUMO / CONNECTED SYSTEM",.023,.026,20,white,spacing=4)
    view._text("LOCAL · LIVE",.975,.030,10,quiet,align="right",spacing=2)
    cpu=f"{snap.cpu:.1f}%" if snap.cpu_ready else "—"
    gpu=f"{snap.gpu:.1f}%" if snap.gpu is not None else "—"
    ram=f"{snap.ram_used/2**30:.1f} / {snap.ram_total/2**30:.1f} GiB"
    network=f"↓{snap.net_down/2**20:.1f} ↑{snap.net_up/2**20:.1f} MiB/s"
    storage=f"{snap.storage_percent:.1f}% used"
    readings=(cpu,network,str(len(snap.process_details)),ram,gpu,storage)
    labels=((.100,.105,"left"),(.025,.445,"left"),(.082,.749,"left"),
            (.970,.126,"right"),(.975,.455,"right"),(.812,.755,"left"))
    for slot,(x,y,align) in enumerate(labels):
        color=tuple(round(v*255) for v in SOURCE_COLORS[slot])
        view._text(NAMES[slot],x,y,12,white,align=align,spacing=1)
        view._text(readings[slot],x,y+.025,18 if slot in (0,2,4,5) else 14,color,align=align)
        end=x+(.054 if align=="left" else -.055)
        view._line((x,y+.057),(end,y+.057),(*color,195))
        targets=((.19,.19),(.11,.49),(.20,.780),(.835,.21),(.90,.49),(.80,.775))
        view._line((x if slot==5 else end,y+.057),targets[slot],(*color,145))
        pygame.draw.circle(view.hud_surface,(*color,230),(round(end*view.width),round((y+.057)*view.height)),max(2,round(2*view.scale)))
