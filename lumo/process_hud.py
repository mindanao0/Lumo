"""Reference-style labels: slot-colored names with compact numeric rows."""
import math

from .process_visuals import build_slots, short_name

INK=(236,243,247)
QUIET=(150,176,190)
DASH='—'
NAME_WIDTH=.19
EDGE=.012


def fitted(view,value,width,size=16):
    """Ellipsize to `width` (fraction of window width); never alters identities."""
    font=view._font(size,True,'ui')
    limit=width*view.width
    if font.size(value)[0]<=limit:
        return value
    while value and font.size(value+'…')[0]>limit:
        value=value[:-1]
    return value+'…'


def pct(value,digits=1):
    return DASH if value is None or not math.isfinite(value) else f'{value:.{digits}f}%'


def slot_rows(slot):
    """Numeric rows for one slot; vacant slots show only the dash name."""
    if not slot.occupied:
        return []
    cpu_ram=f'CPU {pct(slot.cpu_percent)}   RAM {pct(slot.ram_percent)}'
    if slot.role=='gpu':
        return [f'Machine GPU {pct(slot.gpu_percent,0)}',cpu_ram]
    if slot.role=='ram':
        return [f'TOTAL RAM {pct(slot.total_ram_percent)}',f'LUMO {pct(slot.worker_ram_percent)}']
    return [cpu_ram]


def _color(rgb):
    return tuple(round(max(0.,min(1.,c))*255) for c in rgb)


def _center(view,x,texts,size):
    """Centre x clamped so the widest line stays inside the window."""
    half=max(view._font(size,False,'data').size(t)[0] for t in texts)/2/view.width
    return max(EDGE+half,min(1.-EDGE-half,x))


def _block(view,slot,name,rows,opacity,name_dy=0.):
    x,y=slot.label_anchor
    name=fitted(view,name,NAME_WIDTH,15)
    x=_center(view,x,[name,*rows],13)
    view._text(name,x,y+name_dy,15,_color(slot.color),bold=True,align='center',opacity=opacity,family='ui')
    for index,row in enumerate(rows):
        view._text(row,x,y+.036+.032*index,13,INK,align='center',opacity=opacity,family='data')


def paint_process_hud(view):
    snapshot=view.snapshot
    view._text('Lumo',.025,.025,22,INK,bold=True,family='ui')
    cpu=pct(snapshot.cpu) if getattr(snapshot,'cpu_ready',True) else DASH
    status=f'Machine CPU {cpu}    GPU {pct(snapshot.gpu,0)}'
    view._text(status,_center(view,.5,[status],13),.028,13,INK,align='center',family='data')
    slots=build_slots(view.process_sources,snapshot)
    for slot in slots:
        _block(view,slot,short_name(slot.name,40) if slot.occupied else DASH,slot_rows(slot),
               slot.opacity if slot.occupied else 1.)
    # Outgoing identities keep their own name, fading with their cloud, one
    # line above the incoming name; they never carry measurements.
    by_slot={slot.slot:slot for slot in slots}
    for source in view.process_sources.sources:
        if source.closed_at is None or not source.cloud_enabled or source.opacity<.03:
            continue
        slot=by_slot.get(source.slot)
        if slot is not None:
            _block(view,slot,short_name(source.name,40),[],source.opacity,name_dy=-.032)
    if getattr(view,'capture_label',''):
        view._text(view.capture_label,.5,.058,11,QUIET,align='center',family='ui')
