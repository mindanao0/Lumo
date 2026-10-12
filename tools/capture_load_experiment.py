"""Record a real closed-loop CPU/GPU 90% load experiment on this machine."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pygame
from OpenGL.GL import GL_BACK,GL_PACK_ALIGNMENT,GL_RGB,GL_UNSIGNED_BYTE,glPixelStorei,glReadBuffer,glReadPixels
from lumo.gpu_view import GpuView
from lumo.process_sources import ProcessSources
from lumo.packets import packet_rate,DUST_PER_PACKET
from lumo.showcase import ShowcaseState
from load_experiment import LoadExperiment,NAMES


def main():
    output=Path(sys.argv[1] if len(sys.argv)>1 else 'artifacts/load-90-experiment')
    output.mkdir(parents=True,exist_ok=True)
    experiment=LoadExperiment(float(sys.argv[2]) if len(sys.argv)>2 else 90.);view=None;encoder=None
    snapshots=[];complete=False
    try:
        info=experiment.start();print(json.dumps(info),flush=True)
        state=ShowcaseState();view=GpuView(state,max_apps=5)
        # Owned workloads have named clouds for this experiment only.
        view.process_sources=ProcessSources(names=NAMES,max_apps=5)
        width,height=view.screen.get_size();glPixelStorei(GL_PACK_ALIGNMENT,1)
        encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
            '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
            '-framerate','30','-i','pipe:0','-an','-vf','vflip,scale=1280:720',
            '-c:v','libx264','-preset','veryfast','-crf','25','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(output/'lumo-load-experiment.mp4')],stdin=subprocess.PIPE)
        clock=pygame.time.Clock()
        for _ in range(150):
            pygame.event.pump();state.update(1/30);view.draw();clock.tick(30)
        stages=(('baseline',15),('ramp',10),('hold',15),('recovery',20))
        for stage,seconds in stages:
            experiment.set_stage(stage)
            print(f'Starting {stage}',flush=True)
            for frame in range(seconds*30):
                for event in pygame.event.get():
                    if event.type==pygame.QUIT or (event.type==pygame.KEYDOWN and event.key in (pygame.K_ESCAPE,pygame.K_q)):
                        raise KeyboardInterrupt
                if experiment.error:raise RuntimeError(experiment.error)
                # Real observed totals, explicitly distinct from per-program CPU/RAM.
                snap=view.snapshot
                cpu=f'{snap.cpu:.1f}%' if snap.cpu_ready else '—'
                gpu=f'{snap.gpu:.0f}%' if snap.gpu is not None else '—'
                view.capture_label=f'{stage.capitalize()}    Machine CPU {cpu}    GPU {gpu}    Target {experiment.target:.0f}% during load'
                state.update(1/30);view.draw(present=False)
                glReadBuffer(GL_BACK);pixels=glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE)
                encoder.stdin.write(pixels)
                if frame==seconds*30*3//4:
                    pygame.image.save(pygame.image.frombytes(pixels,(width,height),'RGB',True),output/f'lumo-{stage}.png')
                    snapshots.append({'stage':stage,'cpu_percent':view.snapshot.cpu,'gpu_percent':view.snapshot.gpu,
                                      'ram_used_gib':round(view.snapshot.ram_used/2**30,3),
                                      'programs':[{'name':s.name,'cpu_percent':round(s.cpu/max(1,len(view.snapshot.cores)),1),
                                                   'ram_percent':round(s.memory/max(1,view.snapshot.ram_total)*100,1),
                                                   'flow_speed':round(s.flow_speed,3),'packets_per_second':round(packet_rate(s),2),'live_packets':view.packet_pools[id(s)].active_count,'dust_per_packet':DUST_PER_PACKET,'pid_count':len(s.pids)}
                                                  for s in view.process_sources.sources]})
                pygame.display.flip();clock.tick(30)
        encoder.stdin.close()
        if encoder.wait(timeout=30)!=0:raise RuntimeError('Video encoder failed')
        complete=True
    finally:
        experiment.close()
        if encoder and encoder.poll() is None:
            if encoder.stdin and not encoder.stdin.closed:encoder.stdin.close()
            try:encoder.wait(timeout=10)
            except subprocess.TimeoutExpired:encoder.kill();encoder.wait(timeout=5)
        if view:view.close()
        summary={}
        for stage in ('baseline','ramp','hold','recovery'):
            records=[s for s in experiment.samples if s['stage']==stage]
            if records:
                summary[stage]={key:{'min':min(s[key] for s in records),
                                      'mean':round(statistics.mean(s[key] for s in records),1),
                                      'max':max(s[key] for s in records)}
                                for key in ('cpu_percent','gpu_percent','gpu_temperature_c','cpu_temperature_c')}
        report={'completed':complete,'target_cpu_percent':experiment.target,'target_gpu_percent':experiment.target,
                'video_seconds':60 if complete else None,'worker_names':NAMES,
                'worker_count':len(experiment.workers),'worker_pids':[p.pid for p in experiment.workers],
                'workload':'CPU arithmetic, GPU OpenGL compute, 1 GiB RAM (or 15% available)',
                'error':experiment.error,'summary':summary,'snapshots':snapshots,'samples':experiment.samples}
        (output/'load-report.json').write_text(json.dumps(report,indent=2,default=float))
        print(json.dumps({'completed':complete,'summary':summary,'error':experiment.error}),flush=True)

if __name__=='__main__':main()
