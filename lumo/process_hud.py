"""Minimal percentage labels anchored below the program light clouds."""
from .activity import machine_cpu_percent

INK=(222,237,245)
QUIET=(136,164,180)
PROGRAM_NAMES={
    'other processes':'Other programs',
    'firefox':'Firefox', 'chrome':'Google Chrome', 'chromium':'Chromium',
    'gnome-shell':'GNOME Shell', 'code':'Visual Studio Code',
    'python':'Python', 'python3':'Python', 'codex':'Codex',
    'pipewire':'PipeWire', 'wireplumber':'WirePlumber',
    'Xwayland':'Xwayland', 'systemd':'Systemd',
    'LUMO_LEFT_A':'Left workload A', 'LUMO_LEFT_B':'Left workload B',
    'LUMO_LEFT_C':'Left workload C', 'LUMO_RIGHT_A':'Right workload A',
    'LUMO_RIGHT_B':'Right workload B',
}


def fitted(view,value,width,size=16):
    font=view._font(size,True,'ui')
    if font.size(value)[0]<=width*view.width:
        return value
    while value and font.size(value+'…')[0]>width*view.width:
        value=value[:-1]
    return value+'…'


def paint_process_hud(view):
    from .gpu_view import STATIONS
    view._text('Lumo',.025,.025,20,QUIET,family='ui')
    view._text('Program usage as % of system capacity',.025,.057,12,QUIET,family='ui')
    for source in view.process_sources.sources:
        if source.opacity<.08:
            continue
        sx,sy=STATIONS[source.slot][:2]
        name=PROGRAM_NAMES.get(source.name,source.name)
        name=fitted(view,name,.17,14)
        x=.5+sx*view.height/(2*view.width)
        # Keep the complete text block inside the window when resized.
        x=max(.095,min(.905,x))
        y=.5-sy/2+.10
        color=tuple(round(c*255*.65+quiet*.35) for c,quiet in zip(source.color,QUIET))
        alive=source.closed_at is None
        cpu=f'{machine_cpu_percent(source.cpu,len(view.snapshot.cores)):.1f}%' if alive and source.cpu_ready else '—'
        total=view.snapshot.ram_total
        ram=f'{source.memory/total*100:.1f}%' if alive and total>0 else '—'
        view._text(name,x,y,14,INK,align='center',opacity=source.opacity,family='ui')
        # A small signal trace shows recent measurements, never a synthetic oscillator.
        width=.20*view.height/view.width
        baseline=y+.040
        memory_color=tuple(round(value*.65+220*.35) for value in color)
        view._line((x-width/2,baseline),(x+width/2,baseline),(*QUIET,round(45*source.opacity)))
        history=list(view.process_history.get(id(source),()))
        for metric,direction,trace_color in [(0,-1,color),(1,1,memory_color)]:
            previous=None
            for index,sample in enumerate(history):
                value=sample[metric]
                if value is None:
                    previous=None
                    continue
                point=(x-width/2+(48-len(history)+index)*width/47,
                       baseline+direction*.021*max(0.,min(1.,value)))
                if previous is not None:
                    view._line(previous,point,(*trace_color,round(210*source.opacity)))
                previous=point
        gap=.058*view.height/view.width
        view._text('CPU '+cpu,x-gap,y+.067,12,color,align='center',opacity=source.opacity,family='data')
        view._text('RAM '+ram,x+gap,y+.067,12,memory_color,align='center',opacity=source.opacity,family='data')
        if not alive or not source.cpu_ready:
            status='Ended' if not alive else 'Reading CPU…'
            view._text(status,x,y+.090,11,QUIET,align='center',opacity=source.opacity,family='ui')
    if getattr(view,'capture_label',''):
        view._text(view.capture_label,.5,.028,11,QUIET,align='center',family='ui')
