"""Rate-controlled packet births; the GPU still evaluates the moving trajectories."""
import math
import numpy as np


DUST_PER_PACKET=36
WAKE_POINTS=12


def packet_rate(source):
    """Compensate shorter travel time to reach a load-dependent live population."""
    activity=max(0.,min(1.,float(source.render_emission)))
    target_population=24.+1600.*activity
    return max(.25,target_population*max(.16,float(source.flow_speed))/22.25)


class PacketPool:
    def __init__(self, source, capacity=3072):
        self.capacity=capacity
        self.rng=np.random.default_rng(1409+source.slot*113)
        self.positions=np.zeros((capacity,3),dtype=np.float32)
        self.positions[:,0]=-1.e6
        self.colors=np.zeros((capacity,3),dtype=np.float32)
        self.cursor=0;self.count=0;self.credit=0.
        self.last_flow=source.flow_clock
        self.total_births=0
        self.initialized=False
        self.now=0.
        self.active_count=0

    def update(self,source,dt,now):
        rate=packet_rate(source)
        self.now=now
        start=self.last_flow;self.last_flow=source.flow_clock
        alive=int(np.count_nonzero(self.positions[:self.count,2]>now))
        changed=alive!=self.active_count
        self.active_count=alive
        if not self.initialized:
            self.initialized=True
            # Existing programs start with a populated slow cloud, not an empty one.
            if source.pids and source.closed_at is None and source.born_flow<source.flow_clock-20:
                count=min(self.capacity//2,max(24,int(rate*22.25/max(.16,source.flow_speed))))
                for _ in range(count):
                    seed=self.rng.random();life=18.8+6.9*seed
                    age=self.rng.uniform(0.,life*.80)
                    remaining=(life-age)/max(.16,source.flow_speed)
                    self.positions[self.cursor]=(source.flow_clock-age,self.rng.uniform(-1,1),math.ceil(now+remaining+1.))
                    self.colors[self.cursor]=(seed,self.rng.uniform(.45,.90),seed)
                    self.cursor=(self.cursor+1)%self.capacity
                self.count=count;changed=True
        if dt<=0 or source.closed_at is not None or not source.pids:
            return changed
        previous=self.credit
        self.credit+=dt*rate
        births=min(self.capacity,int(self.credit))
        self.credit-=births
        for i in range(births):
            fraction=min(1.,max(0.,(i+1.-previous)/(dt*rate)))
            index=self.cursor
            seed=self.rng.random()
            # Match arrival + transit + fade in processLifetime() in the shader.
            lifetime=(18.8+6.9*seed)/max(.16,source.flow_speed)
            expiry=math.ceil(now-dt+fraction*dt+lifetime+1.)
            self.positions[index]=(start+(source.flow_clock-start)*fraction,self.rng.uniform(-1,1),expiry)
            self.colors[index]=(seed,self.rng.uniform(.45,.90),seed)
            self.cursor=(index+1)%self.capacity
            self.count=min(self.capacity,self.count+1)
        self.total_births+=births
        return changed or births>0

    def arrays(self):
        mask=self.positions[:self.count,2]>self.now
        heads=self.positions[:self.count][mask]
        colors=self.colors[:self.count][mask]
        count=len(heads)
        self.active_count=count
        wakes=np.repeat(heads,WAKE_POINTS,axis=0)
        wakes[:,2]+=np.tile(np.linspace(.06,.98,WAKE_POINTS),count)
        dust=np.repeat(heads,DUST_PER_PACKET,axis=0)
        lane_phase=np.repeat(colors[:,2],DUST_PER_PACKET)*91.+np.tile(np.arange(DUST_PER_PACKET),count)*5.
        # Separate the dust visibly along and across the packet, rather than stacking glow.
        dust[:,0]+=.35*(.5+.5*np.sin(lane_phase*.73))
        dust[:,1]=np.clip(dust[:,1]+.11*np.sin(lane_phase),-1,1)
        dust_colors=np.repeat(colors,DUST_PER_PACKET,axis=0).copy()
        dust_colors[:,1]*=.35
        dust_colors[:,2]=np.mod(dust_colors[:,2]+.025*np.sin(lane_phase*1.31),1.)
        return {'head':(heads,colors),'wake':(wakes,np.repeat(colors,WAKE_POINTS,axis=0)),
                'dust':(dust,dust_colors)}
