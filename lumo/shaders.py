"""Parametric 3D silk, optical light scattering, and HDR compositing on the GPU."""

PARTICLE_VERTEX = r"""
#version 120
uniform float uTime, uFlowTime, uParticleTime, uAspect, uScale, uGlow;
uniform float uEntry,uBand,uEmission,uShapeActivity,uShapeDrift;
uniform float uBirthTime,uStopTime,uCloudAlpha,uCloudScale;
uniform int uKind,uReactive,uConnected;
uniform float uLoads[6];
uniform vec2 uCenter;
uniform mat3 uCoreRotation;
uniform float uCanvasFraction,uWorldScale;
uniform vec3 uCool, uWarm, uNodeColor;
varying vec3 vColor;
varying float vAlpha;
varying float vSpark;
vec3 rx(vec3 p,float a) {return vec3(p.x,cos(a)*p.y-sin(a)*p.z,sin(a)*p.y+cos(a)*p.z);}
vec3 ry(vec3 p,float a) {return vec3(cos(a)*p.x+sin(a)*p.z,p.y,-sin(a)*p.x+cos(a)*p.z);}
float hash3(vec3 p) {
    p=fract(p*.1031);
    p+=dot(p,p.yzx+33.33);
    return fract((p.x+p.y)*p.z);
}
float softNoise(vec3 p) {
    vec3 cell=floor(p),f=fract(p);
    f=f*f*f*(f*(f*6.-15.)+10.);
    return mix(mix(mix(hash3(cell),hash3(cell+vec3(1,0,0)),f.x),
                   mix(hash3(cell+vec3(0,1,0)),hash3(cell+vec3(1,1,0)),f.x),f.y),
               mix(mix(hash3(cell+vec3(0,0,1)),hash3(cell+vec3(1,0,1)),f.x),
                   mix(hash3(cell+vec3(0,1,1)),hash3(cell+vec3(1,1,1)),f.x),f.y),f.z)*2.-1.;
}
vec3 ribbon(vec3 q) {
    // A filled cloud volume: radius and depth vary independently of circulation.
    float t=uFlowTime*.10;
    float u=q.x+uTime*.032;
    float latitude=clamp(q.y,-.97,.97);
    float radius=.66*clamp(q.z,0.,1.);
    float crossSection=sqrt(1.-latitude*latitude);
    vec3 p=vec3(radius*cos(u)*crossSection,
                radius*sin(u)*crossSection,radius*latitude*.72);
    p=rx(p,.35);
    vec3 field=p*2.0+vec3(t*.3,t*.2,t*.15);
    float broad=softNoise(field);
    // Gentle internal currents and breathing; no cut-out centre or sharp folds.
    p.xy+=.023*vec2(broad,softNoise(field+vec3(2.7,1.4,4.2)))*q.z;
    p.x*=1.+.035*sin(t*.6);
    p.y*=.86+.035*cos(t*.5);
    p.xy+=vec2(-.025+.012*sin(t*.37),.012*cos(t*.43));
    return p;
}

vec3 connectedCore(vec3 q) {
    float t=clamp((q.y+1.)*.5,0.,1.);
    float radius=.035+.48*pow(t,.82);
    float angle=q.x+radius*7.;
    vec3 p=vec3(radius*cos(angle),radius*sin(angle)*.65,
                (q.z-1.)*.55+.045*sin(angle*2.)*t);
    p=rx(p,.24);
    p.xy+=vec2(-.025,.10);
    return p;
}
vec3 connectedPath(float age,float lane,float seed) {
    lane=.85+clamp(lane,-1.,1.)*.1;
    float arrival=2.8+seed*.9;
    float speed=.12+seed*.045;
    float entry=atan(uCenter.y-.10,uCenter.x+.025)-.51*7.;
    vec3 q=vec3(entry+(seed-.5)*.45,lane,1.);
    if(age>=arrival) {
        q.y=-1.+(lane+1.)*exp(-(age-arrival)*speed*.65);
        return connectedCore(q);
    }
    vec3 end=connectedCore(q);
    vec3 velocity=(connectedCore(q+vec3(.002,0,0))-connectedCore(q-vec3(.002,0,0)))/.004*speed;
    vec3 start=vec3(uCenter,0.);
    start.xy+=vec2(sin(seed*83.),cos(seed*71.))*.025;
    vec2 side=vec2(-(end.y-start.y),end.x-start.x);
    side*=inversesqrt(max(dot(side,side),.001));
    vec3 c1=start+(end-start)*.34;
    c1.xy+=side*(lane*.14+.12);
    vec3 c2=end-velocity*arrival/3.;
    float t=clamp(age/arrival,0.,1.),s=1.-t;
    return s*s*s*start+3.*s*s*t*c1+3.*s*t*t*c2+t*t*t*end;
}

vec3 station(vec3 q,float seed) {
    // Each cloud moves coherently about its fixed anchor, with its own phase.
    // Integrated source/device time makes activity changes continuous.
    float phase=uCenter.x*2.1+uCenter.y*3.7+uBand*.9;
    float clock=uTime*.045+uParticleTime*.08;
    vec3 p=ry(rx(q,.15+uCenter.x*.29+clock*(.52+.08*cos(phase))),
                 uCenter.y*.73+.12+clock*(.69+.09*sin(phase)));
    p*=1.+.035*sin(clock*.8+phase);
    p.y*=1.+.018*cos(clock*.65+phase);
    p.xy+=vec2(sin(seed*18.0+uTime*.13),cos(seed*21.0+uTime*.12))*.032;
    p.xy=p.xy*uCloudScale+uCenter;
    return p;
}
float processArrival(float seed) {return 2.8+seed*.9;}
float processTransitTime(float seed) {return 12.+seed*4.;}
float processFadeTime(float seed) {return 4.+seed*2.;}
float processLifetime(float seed) {
    return processArrival(seed)+processTransitTime(seed)+processFadeTime(seed);
}
vec3 processCurrent(float travel,float lane,float seed) {
    vec2 direction=-uCenter*inversesqrt(max(dot(uCenter,uCenter),.001));
    vec2 across=vec2(-direction.y,direction.x);
    float duration=processTransitTime(seed)+processFadeTime(seed);
    float t=travel/duration;
    // Circular cross sections and rounded chord ends fill an ellipsoid, not a slab.
    float radial=.52*pow(clamp((lane+1.)*.5,0.,1.),.78);
    float angle=seed*6.2831853;
    float lateral=radial*cos(angle);
    float depth=radial*sin(angle);
    float halfChord=sqrt(max(.001,.58*.58-radial*radial));
    float longitudinal=mix(-halfChord,halfChord,t);
    float envelope=sin(3.14159265*t);
    vec3 p=vec3(direction*longitudinal+across*lateral,depth*.88);
    // Gentle curves preserve a continuous source-to-core trajectory.
    p.xy+=across*.035*envelope*sin(angle+t*3.14159265);
    p.z+=.025*envelope*sin(angle*2.+t*6.2831853);
    vec3 field=p*2.3+vec3(uFlowTime*.025);
    p+=vec3(softNoise(field),softNoise(field+vec3(3.7,1.2,2.5)),
             softNoise(field+vec3(1.6,4.2,3.1)))*.018*envelope;
    // Entry stays on the source's side in world space. Rotation takes over
    // gradually inside the volume instead of dragging the inlet across it.
    vec3 rotated=uCoreRotation*p;
    p=mix(p,rotated,smoothstep(0.,.45,t));
    p.xy+=vec2(-.025,0.);
    return p;
}
vec3 processPath(float age,float lane,float seed) {
    if(uConnected==1) return connectedPath(age,lane,seed);
    float arrival=processArrival(seed);
    if(age>=arrival) return processCurrent(age-arrival,lane,seed);
    vec3 end=processCurrent(0.,lane,seed);
    vec3 velocity=(processCurrent(.002,lane,seed)-processCurrent(-.002,lane,seed))/.004;
    vec3 start=vec3(uCenter,0.);
    start.xy+=vec2(sin(seed*83.),cos(seed*71.))*.045;
    vec2 side=vec2(-(end.y-start.y),end.x-start.x);
    side*=inversesqrt(max(dot(side,side),.001));
    vec3 c1=start+(end-start)*.40;
    c1.xy+=side*lane*.06;
    vec3 c2=end-velocity*arrival/3.;
    float t=clamp(age/arrival,0.,1.),s=1.-t;
    return s*s*s*start+3.*s*s*t*c1+3.*s*t*t*c2+t*t*t*end;
}

vec3 deviceColor(int i) {
    if(i==0) return vec3(.16,.65,1.);
    if(i==1) return vec3(.14,.91,.86);
    if(i==2) return vec3(.72,.42,1.);
    if(i==3) return vec3(.30,.94,.72);
    if(i==4) return vec3(1.,.62,.20);
    return vec3(1.,.80,.47);
}
float deviceAngle(int i) {
    if(i==0) return 2.60;
    if(i==1) return 3.14;
    if(i==2) return 4.10;
    if(i==3) return .85;
    if(i==4) return -.28;
    return -1.10;
}
vec4 loadLight(vec3 p) {
    float angle=atan(p.y,p.x+.025);
    vec3 color=vec3(.17,.25,.34)*.28;
    float total=.28,activity=0.;
    for(int i=0;i<6;i++) {
        float load=clamp(uLoads[i],0.,1.);
        float delta=angle-deviceAngle(i);
        delta=atan(sin(delta),cos(delta));
        float influence=exp(-delta*delta/(.48+load*.85));
        float strength=pow(load,.75)*influence;
        color+=deviceColor(i)*strength;
        activity+=load*strength;
        total+=strength;
    }
    return vec4(color/total,activity/total);
}

void main() {
    vec3 q=gl_Vertex.xyz,p=q;
    int role=(uKind>=29 && uKind<=31)?uKind-11:uKind;
    float seed=gl_Color.b,alpha=gl_Color.g,size=1.0+seed*1.5,warmth=gl_Color.r;
    float energy=1.0, whiten=seed*.25;
    float packetTail=(uKind>=29 && uKind<=31)?fract(q.z):q.z;
    vSpark=0.0;
    if(uKind==32) {
        p=uCoreRotation*q;
        p.xy+=vec2(-.025,0.);
        float compute=max(uLoads[0],uLoads[4]);
        alpha*=(.90+compute*.65)*(1.-smoothstep(.28+.64*compute-.025,.28+.64*compute+.025,seed));
        // Small bright points retain detail instead of merging into soft fog.
        float front=smoothstep(-.46,.46,p.z);
        alpha*=.18+1.12*front;
        // Features stay in object space so rotation is visible on the rounded cloud.
        float ribbon=.5+.5*sin(atan(q.z,q.x)*2.+q.y*9.);
        float crest=pow(ribbon,6.);
        // An asymmetric luminous patch provides a stable point to follow.
        float landmark=smoothstep(.82,.97,dot(normalize(q+vec3(.0001)),normalize(vec3(.72,.35,.60))));
        alpha*=.45+1.65*crest+1.25*landmark;
        if(uGlow>.5) alpha*=.20;
        size=1.5+seed*1.3;energy=.95+compute*.65;whiten=.08+.20*max(crest,landmark);vSpark=0.;
        warmth=smoothstep(-.3,.3,q.x);
    } else if(uKind==21 || uKind==22 || uKind==23) {
        // Points drift inward on spiral arms; the cloud itself keeps a stable pose.
        if(uKind==23) q.y=2.*fract((q.y+1.)*.5-uFlowTime*.018)-1.;
        p=connectedCore(q);
        alpha*=.65+.35*smoothstep(-.12,.12,p.z);
        warmth=smoothstep(-.30,.35,p.x);
        energy=uKind==21?.8:1.45;
        whiten=.08;size=1.2+seed*2.;
        if(uKind==23) {size=2.5+seed;energy=1.45;whiten=.14;vSpark=1.;}
    } else if(uKind==24) {
        p=connectedPath(q.x*(2.8+seed*.9),uBand*.14+q.y,seed);
        alpha*=.55;
        energy=1.15;whiten=.045;
    } else     if(uKind==0 || uKind==13 || uKind==17) {
        if(uKind==13) q.x+=uTime*(.06+seed*.08);
        if(uKind==17) q.x+=uFlowTime*(.32+seed*.36);
        p=ribbon(q);
        vec3 tangent=ribbon(q+vec3(.002,0.,0.))-p;
        vec3 across=ribbon(q+vec3(0.,.002,0.))-p;
        vec3 crossed=cross(tangent,across);
        vec3 normal=crossed*inversesqrt(max(dot(crossed,crossed),1.e-20));
        float fresnel=pow(clamp(1.0-abs(normal.z),0.,1.),2.3);
        float front=smoothstep(-.25,.22,p.z);
        // Silvery grazing reflections highlight folds; rear strands stay dim.
        vec3 keyLight=normalize(vec3(-.38,.66,.72));
        float diffuse=.38+.62*max(dot(normal,keyLight),0.);
        vec3 reflected=reflect(-keyLight,normal);
        float highlight=pow(clamp(reflected.z,0.,1.),14.);
        energy=(.75+front*.85+fresnel*1.15)*diffuse+highlight*.95;
        whiten=clamp(.12+.42*fresnel+.30*highlight,0.,.82);
        warmth=smoothstep(-.14,.56,p.x+p.z*.95)*(.58+.42*(.5+.5*sin(q.y*1.5)));
        float sheet=pow(.5+.5*cos(3.14159265*q.y),2.);
        alpha*=(.32+.68*front)*(.48+.52*sheet+.20*fresnel);
        size*=.65+.4*front;
        if(uKind==13) {
            size=5.0+seed*6.0;alpha*=.9;
            energy=2.3;whiten=.62;vSpark=1.0;
        }
    } else if(uKind==2) {
        float a=q.x+uTime*(.016+seed*.016);
        p=ry(rx(vec3(cos(a)*q.y,sin(a)*q.y,q.z),.40+seed*.62),-.20);
        p.xy+=vec2(-.025,0.0);
        warmth=smoothstep(-.3,.65,p.x);
        alpha*=.65;size=1.6+seed*2.2;
    } else if(uKind==3) {
        p.xy+=vec2(sin(uTime*.025+seed*15.0),cos(uTime*.018+seed*19.0))*.012;
        alpha*=.7+.3*sin(seed*41.0+uTime*.4);
        size=1.0+seed*3.0;
        if(q.z>.91) {size=8.+seed*8.;alpha*=.12;}
        energy=.8;whiten=.12;
    } else if(uKind==15 || uKind==16) {
        // The same advancing phase drives a packet head and its curved fading wake.
        float tail=uKind==15?q.z:0.;
        float head=q.x+uFlowTime*(.42+seed*.38);
        vec3 path=vec3(head-tail*(.22+seed*.38),q.y,1.);
        p=ribbon(path);
        float front=smoothstep(-.35,.28,p.z);
        alpha*=pow(max(0.,1.-tail),1.65)*(.30+.70*front);
        energy=uKind==16?3.4:2.1;
        whiten=uKind==16?.35:.13;
        size=uKind==16?3.8+seed*3.2:1.2+2.0*(1.-tail);
        warmth=smoothstep(-.18,.55,p.x);
        if(uKind==16) vSpark=1.;
    } else if(uKind==9) {
        p=ry(rx(q,.1),uTime*.08);
        p.xy+=vec2(-.025,0.0);
        warmth=.50;size=5.;energy=3.6;whiten=.65;
    } else if(uKind==11 || uKind==14) {
        p=station(q,seed);
        float depth=smoothstep(-1.,1.,p.z);
        alpha*=.3+.7*depth;
        size=1.4+seed*3.;energy=1.25;whiten=seed*.15;
        if(uKind==14) {vSpark=1.;size=7.+seed*7.;energy=3.0;whiten=.48;}
        // Smaller groups reuse the buffers, with less light per point to preserve detail.
        float cloudRatio=clamp(uCloudScale/.235,0.,1.);
        alpha*=uCloudAlpha*max(.10,pow(cloudRatio,1.6));
        size*=max(.55,sqrt(cloudRatio));
    } else if(role==18 || role==19 || role==20) {
        float life=processLifetime(seed);
        float headAge=uKind>=29?uParticleTime-q.x:mod(uParticleTime+q.x*life,life);
        float birthTime=uParticleTime-headAge;
        float delay=role==20?packetTail*.24:0.;
        float age=clamp(headAge-delay,0.,life+.1);
        p=processPath(age,uBand*.14+q.y,seed);
        float selected=uKind>=29?1.:1.-smoothstep(uEmission-.035,uEmission+.035,gl_Color.r);
        float birth=smoothstep(0.,.28,age);
        float fadeStart=processArrival(seed)+processTransitTime(seed);
        float decay=1.-smoothstep(fadeStart,life,age);
        float front=smoothstep(-.35,.30,p.z);
        alpha*=selected*birth*decay*(.38+.62*front);
        if(uConnected==1) alpha*=.28;
        else {
            vec3 local=(p-vec3(-.025,0.,0.))*uCoreRotation;
            float distance=length(local/vec3(.60,.60,.53));
            float density=exp(-1.35*distance*distance)*(1.-smoothstep(.70,1.04,distance));
            float captured=smoothstep(processArrival(seed)-.30,processArrival(seed)+.40,age);
            alpha*=mix(1.,density*1.6,captured);
        }
        alpha*=headAge>=delay?1.:0.;
        if(uKind>=29) alpha*=1.-smoothstep(floor(q.z)-1.5,floor(q.z),uTime);
        // Stop new births when a program closes; already born particles finish their crossing.
        alpha*=birthTime>=uBirthTime && birthTime<=uStopTime?1.:0.;
        size=role==18?6.+seed*3.:2.3+seed*1.3;
        energy=role==18?.90:1.15;
        whiten=.01;
        // Shared device load lights the volume, without attributing GPU/I/O to an app.
        float coreMix=smoothstep(processArrival(seed)-.3,processArrival(seed)+.4,age);
        energy*=mix(1.,.70+1.0*max(uLoads[0],uLoads[4]),coreMix);
        if(role==18) {
            size=mix(size,1.6+seed*1.2,coreMix);
            alpha*=mix(.16,.32,coreMix)*(.85+.30*uLoads[3]);
            if(uGlow>.5) alpha*=mix(1.,.18,coreMix);
            vSpark=coreMix>.5?0.:-1.;
        }
        if(role==20) {
            alpha*=.38*pow(1.-packetTail,1.5);
            size=1.3+1.4*(1.-packetTail);energy=1.15;whiten=.01;
        }
    }

    gl_Position=vec4(p.x*uWorldScale/uAspect*uCanvasFraction+uCanvasFraction-1.,p.y*uWorldScale,clamp(-p.z*.4,-.9,.9),1.0);
    gl_PointSize=size*uScale*(uGlow>.5?(vSpark<-.5?3.0:6.0):1.0);
    vColor=mix(mix(uCool,uWarm,warmth),vec3(.90,.97,1.),whiten)*energy;
    if((uKind==0 || uKind==13 || uKind==15 || uKind==16 || uKind==17) && uReactive==1) {
        vec4 light=loadLight(p);
        vColor=mix(light.rgb,vec3(.90,.97,1.),whiten)*energy*(.75+.65*light.a);
    }
    if(uKind==21 || uKind==22 || uKind==23) {
        float sector=mod(q.x+6.2831853,6.2831853)/6.2831853*6.;
        vColor=mix(deviceColor(int(floor(sector))),vec3(.90,.97,1.),whiten)*energy;
    }
    if(uKind==11 || uKind==14)
        vColor=mix(uNodeColor,vec3(.93,.98,1.),whiten)*energy;
    if(role==18 || role==19 || role==20 || uKind==24) {
        vec3 sourceColor=uReactive==1?uNodeColor:mix(uCool,uWarm,seed);
        vColor=mix(sourceColor,vec3(.90,.97,1.),whiten)*energy;
    }
    vAlpha=alpha*(uGlow>.5?(vSpark<-.5?.08:.045):.90);
}
"""

LINK_FRAGMENT = r"""
#version 120
varying vec3 vColor;
varying float vAlpha;
void main() {gl_FragColor=vec4(vColor,vAlpha);}
"""

PARTICLE_FRAGMENT = r"""
#version 120
varying vec3 vColor;
varying float vAlpha;
varying float vSpark;
void main() {
    vec2 p=gl_PointCoord*2.0-1.0;
    float r=dot(p,p);
    if(vSpark>1.5) {
        float box=max(abs(p.x),abs(p.y));
        if(box>.82) discard;
        float edge=1.-smoothstep(.32,.78,box);
        gl_FragColor=vec4(vColor,edge*vAlpha*.75);
        return;
    }
    if(r>1.0) discard;
    float light=exp(-r*5.5)*.55+exp(-r*28.0)*.9;
    if(vSpark<-.5) light=exp(-r*3.8)*.55;
    if(vSpark>.5) {
        float rays=(exp(-abs(p.x)*55.)+exp(-abs(p.y)*55.))*exp(-r*3.5);
        light=exp(-r*40.)*2.0+rays*.26+exp(-r*6.)*.24;
    }
    gl_FragColor=vec4(vColor,light*vAlpha);
}
"""

BACKGROUND_VERTEX = r"""
#version 120
varying vec2 vPosition;
void main() {gl_Position=gl_Vertex; vPosition=gl_Vertex.xy;}
"""

BACKGROUND_FRAGMENT = r"""
#version 120
uniform float uAspect;
uniform vec3 uCool,uWarm;
varying vec2 vPosition;
void main() {
    vec2 p=vec2(vPosition.x*uAspect,vPosition.y);
    float mist=exp(-dot(p,p)*2.0);
    float side=exp(-pow(p.x-.65,2.0)*3.0-p.y*p.y*3.0);
    float opening=1.0-.88*exp(-dot(p,p)*42.0);
    float grain=fract(sin(dot(gl_FragCoord.xy,vec2(12.9898,78.233)))*43758.5453);
    vec2 cell=abs(fract(p/.10+.5)-.5);
    float grid=exp(-min(cell.x,cell.y)*180.);
    vec2 coarse=abs(fract(p/.30+.5)-.5);
    float crossMark=exp(-coarse.x*160.)*exp(-coarse.y*160.);
    float bluePrint=(grid*.22+crossMark*.70)*(1.-exp(-dot(p,p)*2.));
    vec3 color=vec3(.004,.010,.019)+uCool*mist*.017+uWarm*side*.007
        +vec3(.06,.20,.32)*bluePrint+vec3(grain*.002);
    gl_FragColor=vec4(color*opening,1.0);
}
"""

POST_VERTEX = r"""
#version 120
varying vec2 vUV;
void main() {gl_Position=gl_Vertex;vUV=gl_MultiTexCoord0.xy;}
"""

BLOOM_FRAGMENT = r"""
#version 120
uniform sampler2D uSource;
uniform vec2 uStep;
uniform int uExtract;
varying vec2 vUV;
void main() {
    vec3 c=texture2D(uSource,vUV).rgb*.227027;
    c+=(texture2D(uSource,vUV+uStep*1.384615).rgb+texture2D(uSource,vUV-uStep*1.384615).rgb)*.316216;
    c+=(texture2D(uSource,vUV+uStep*3.230769).rgb+texture2D(uSource,vUV-uStep*3.230769).rgb)*.070270;
    if(uExtract==1) {
        float peak=max(c.r,max(c.g,c.b));
        c*=smoothstep(.40,1.20,peak);
    }
    gl_FragColor=vec4(c,1.0);
}
"""

COMPOSITE_FRAGMENT = r"""
#version 120
uniform sampler2D uScene,uBloom,uWide;
varying vec2 vUV;
vec3 film(vec3 x) {return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.);}
void main() {
    vec3 scene=texture2D(uScene,vUV).rgb;
    vec3 nearGlow=texture2D(uBloom,vUV).rgb;
    vec3 farGlow=texture2D(uWide,vUV).rgb;
    vec3 color=film(scene*1.18+nearGlow*.42+farGlow*.18);
    // Preserve faint threads without lifting the black background.
    float dark=max(scene.r,max(scene.g,scene.b));
    color=mix(color,scene,.15*(1.-smoothstep(.20,.80,dark)));
    vec2 p=vUV*2.-1.;
    color*=1.-.12*dot(p,p);
    color*=.99+.01*cos(gl_FragCoord.y*3.14159265);
    gl_FragColor=vec4(color,1.);
}
"""
