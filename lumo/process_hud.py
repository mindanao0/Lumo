"""Minimal percentage labels anchored below the program light clouds."""
from .activity import machine_cpu_percent

INK=(222,237,245)
QUIET=(136,164,180)


def fitted(view,value,width,size=16):
    font=view._font(size,False,'ui')
    if font.size(value)[0]<=width*view.width:
        return value
    while value and font.size(value+'…')[0]>width*view.width:
        value=value[:-1]
    return value+'…'


def paint_process_hud(view):
    from .gpu_view import STATIONS
    view._text('Lumo',.025,.025,20,QUIET,family='ui')
    for source in view.process_sources.sources:
        if source.opacity<.08:
            continue
        sx,sy=STATIONS[source.slot][:2]
        name='Other programs' if source.name=='other processes' else source.name
        name=fitted(view,name,.17,16)
        x=.5+sx*view.height/(2*view.width)
        # Keep the complete text block inside the window when resized.
        x=max(.095,min(.905,x))
        y=.5-sy/2+.10
        color=tuple(round(c*255) for c in source.color)
        alive=source.closed_at is None
        cpu=f'{machine_cpu_percent(source.cpu,len(view.snapshot.cores)):.1f}%' if alive and source.cpu_ready else '—'
        total=view.snapshot.ram_total
        ram=f'{source.memory/total*100:.1f}%' if alive and total>0 else '—'
        view._text(name,x,y,16,color,align='center',opacity=source.opacity,family='ui')
        view._text('CPU '+cpu,x,y+.024,15,INK,align='center',opacity=source.opacity,family='data')
        view._text('RAM '+ram,x,y+.045,15,INK,align='center',opacity=source.opacity,family='data')
    if getattr(view,'capture_label',''):
        view._text(view.capture_label,.5,.028,11,QUIET,align='center',family='ui')
