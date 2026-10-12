"""Bounded, owned CPU/RAM/OpenGL compute workers with real utilisation feedback."""
import ctypes
import math
import multiprocessing as mp
import subprocess
import threading
import time
import psutil

NAMES=['LUMO_CPU_A','LUMO_CPU_B','LUMO_CPU_C','LUMO_GPU','LUMO_RAM']


def name_process(name):
    ctypes.CDLL(None).prctl(15,name.encode(),0,0,0)


def cpu_worker(name,duty,stop):
    name_process(name)
    resident=bytearray(8*1024*1024)
    deadline=time.monotonic()+180
    value=1
    while not stop.is_set() and time.monotonic()<deadline:
        start=time.monotonic();busy_until=start+.08*duty.value
        while time.monotonic()<busy_until:
            for _ in range(1500):
                value=(value*1664525+1013904223)&0xffffffff
        stop.wait(max(0.,.08-(time.monotonic()-start)))
    resident[0]=value&255


def ram_worker(amount,stop):
    name_process('LUMO_RAM')
    resident=bytearray(32*1024*1024)
    deadline=time.monotonic()+180
    current=len(resident)
    while not stop.is_set() and time.monotonic()<deadline:
        wanted=amount.value
        if wanted!=current:
            resident=bytearray(wanted)
            for i in range(0,len(resident),4096):
                resident[i]=1
            current=wanted
        stop.wait(.25)


def gpu_worker(duty,stop,ready):
    name_process('LUMO_GPU')
    import numpy as np
    import pygame
    from OpenGL.GL import (GL_COMPUTE_SHADER,GL_SHADER_STORAGE_BUFFER,GL_DYNAMIC_COPY,
                           GL_SHADER_STORAGE_BARRIER_BIT,GL_RENDERER,glGetString,
                           glGenBuffers,glBindBuffer,glBufferData,glBindBufferBase,
                           glUseProgram,glGetUniformLocation,glUniform1i,
                           glDispatchCompute,glMemoryBarrier,glFinish,glDeleteBuffers,glDeleteProgram)
    from OpenGL.GL.shaders import compileProgram,compileShader
    program=None;buffer=None
    try:
        pygame.display.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION,4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION,3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,pygame.GL_CONTEXT_PROFILE_COMPATIBILITY)
        pygame.display.set_mode((64,64),pygame.OPENGL|pygame.DOUBLEBUF|pygame.HIDDEN,vsync=0)
        source='''#version 430
layout(local_size_x=128) in;
layout(std430,binding=0) buffer Work {float values[];};
uniform int iterations;
void main(){uint i=gl_GlobalInvocationID.x;float v=values[i];
for(int k=0;k<iterations;k++){
v=fract(sin(v*12.9898+float(k)*.001)*43758.5453);
v=sqrt(v+.01)*.93+cos(v*7.31)*.05;
} values[i]=v;}
'''
        program=compileProgram(compileShader(source,GL_COMPUTE_SHADER))
        data=np.full(131072,.37,dtype=np.float32)
        buffer=glGenBuffers(1);glBindBuffer(GL_SHADER_STORAGE_BUFFER,buffer)
        glBufferData(GL_SHADER_STORAGE_BUFFER,data.nbytes,data,GL_DYNAMIC_COPY)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,buffer)
        glUseProgram(program);glUniform1i(glGetUniformLocation(program,'iterations'),256)
        ready.put({'ready':True,'renderer':glGetString(GL_RENDERER).decode()})
        deadline=time.monotonic()+180
        while not stop.is_set() and time.monotonic()<deadline:
            start=time.monotonic();busy_until=start+.10*duty.value
            while time.monotonic()<busy_until and not stop.is_set():
                glDispatchCompute(1024,1,1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
                glFinish()
            stop.wait(max(0.,.10-(time.monotonic()-start)))
    except Exception as error:
        ready.put({'ready':False,'error':str(error)})
    finally:
        if buffer is not None:glDeleteBuffers(1,[buffer])
        if program is not None:glDeleteProgram(program)
        pygame.quit()


class LoadExperiment:
    def __init__(self,target=90.):
        ctx=mp.get_context('spawn')
        self.stop=ctx.Event();self.cpu_duty=ctx.Value('d',0.);self.gpu_duty=ctx.Value('d',0.)
        self.ram_amount=ctx.Value('q',32*1024*1024)
        self.ready=ctx.Queue();self.workers=[];self.samples=[];self.stage='baseline'
        self.target=float(target)
        self.error=None;self.thread=None;self.started=time.monotonic()
        self.ram_high=min(1024*1024*1024,int(psutil.virtual_memory().available*.15))
        for index in range(psutil.cpu_count() or 1):
            name=NAMES[min(2,index*3//max(1,psutil.cpu_count() or 1))]
            self.workers.append(ctx.Process(target=cpu_worker,args=(name,self.cpu_duty,self.stop)))
        self.workers.append(ctx.Process(target=ram_worker,args=(self.ram_amount,self.stop)))
        self.workers.append(ctx.Process(target=gpu_worker,args=(self.gpu_duty,self.stop,self.ready)))

    def start(self):
        try:
            for worker in self.workers:worker.start()
            result=self.ready.get(timeout=20)
            if not result['ready']:raise RuntimeError(result['error'])
            self.thread=threading.Thread(target=self.control,daemon=True);self.thread.start()
            return result
        except Exception:
            self.close();raise

    def set_stage(self,stage):
        self.stage=stage
        if stage in ('ramp','hold'):
            self.ram_amount.value=self.ram_high
            if self.cpu_duty.value==0:self.cpu_duty.value=.65
            if self.gpu_duty.value==0:self.gpu_duty.value=.55
        else:
            self.cpu_duty.value=0.;self.gpu_duty.value=0.
            self.ram_amount.value=32*1024*1024

    def control(self):
        psutil.cpu_percent(interval=None)
        while not self.stop.wait(1.):
            cpu=psutil.cpu_percent(interval=None)
            try:
                result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,check=True)
                gpu,temperature=map(float,result.stdout.strip().splitlines()[0].split(','))
                cpu_temperatures=[entry.current for key,entries in psutil.sensors_temperatures().items()
                                  if key in ('coretemp','k10temp') for entry in entries]
                hottest=max(cpu_temperatures,default=0.)
                sample={'seconds':round(time.monotonic()-self.started,2),'stage':self.stage,
                        'cpu_percent':cpu,'gpu_percent':gpu,'gpu_temperature_c':temperature,
                        'cpu_temperature_c':hottest,'cpu_duty':round(self.cpu_duty.value,3),
                        'gpu_duty':round(self.gpu_duty.value,3),'ram_mib':self.ram_amount.value//2**20}
                self.samples.append(sample)
                if temperature>=83 or hottest>=90:
                    self.error=f'Temperature limit reached (GPU {temperature}, CPU {hottest})';self.stop.set();break
                if self.stage in ('ramp','hold'):
                    self.cpu_duty.value=max(.0,min(.97,self.cpu_duty.value+(self.target-cpu)*.006))
                    self.gpu_duty.value=max(.0,min(.97,self.gpu_duty.value+(self.target-gpu)*.009))
                else:
                    self.cpu_duty.value=0.;self.gpu_duty.value=0.
            except Exception as error:
                self.error=str(error);self.stop.set();break

    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=6)
        for worker in self.workers:
            if worker.pid is not None:
                worker.join(timeout=3)
                if worker.is_alive():worker.terminate();worker.join(timeout=3)
