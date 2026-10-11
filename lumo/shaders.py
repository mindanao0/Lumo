"""Parametric 3D silk, optical light scattering, and HDR compositing on the GPU."""

PARTICLE_VERTEX = r"""
#version 120
uniform float uTime, uFlowTime, uAspect, uScale, uGlow;
uniform float uEntry,uBand,uEmission;
uniform float uBirthTime,uStopTime,uCloudAlpha,uCloudScale;
uniform int uKind,uReactive;
uniform float uLoads[6];
uniform vec2 uCenter;
uniform vec3 uCool, uWarm, uNodeColor;
varying vec3 vColor;
varying float vAlpha;
varying float vSpark;
vec3 rx(vec3 p,float a) {return vec3(p.x,cos(a)*p.y-sin(a)*p.z,sin(a)*p.y+cos(a)*p.z);}
vec3 ry(vec3 p,float a) {return vec3(cos(a)*p.x+sin(a)*p.z,p.y,-sin(a)*p.x+cos(a)*p.z);}
vec3 ribbon(vec3 q) {
    float u=q.x+uTime*.032;
    // Circular silhouettes stay fixed; only the surface filaments wind and pleat.
    float section=3.14159265*q.y+3.*u+.10*sin(6.*u-uTime*.08)
        +.025*sin(9.*u+q.y*15.707963-uTime*.15);
    float skin=.295*clamp(q.z,0.,1.);
    float bend=section+.22*sin(3.*u+section)*sin(section);
    float radius=.535+skin*cos(bend);
    float wave=sin(section);
    float sweepAngle=u+.18*sin(3.*u+section)*(wave*wave)*clamp(q.z,0.,1.);
    float folds=.32*sin(section)+.11*sin(3.*u+2.*section-uTime*.07)
        +.025*sin(6.*u+5.*section-uTime*.12);
    vec3 p=vec3(radius*cos(sweepAngle),radius*sin(sweepAngle),folds*clamp(q.z,0.,1.));
    p.xy+=vec2(-.025,0.0);
    return p;
}

vec3 station(vec3 q,float seed) {
    vec3 p=ry(rx(q,.15+uCenter.x*.29),uCenter.y*.73+.12);
    p.xy+=vec2(sin(seed*18.0+uTime*.13),cos(seed*21.0+uTime*.12))*.032;
    p.xy=p.xy*uCloudScale+uCenter;
    return p;
}
float processArrival(float seed) {return 2.8+seed*.9;}
float processSpeed(float seed) {return .62+seed*.18;}
float processOrbitTime(float seed) {
    // Complete several revolutions before fading; all parts of a packet share this clock.
    return 6.2831853*(2.2+seed*.6)/processSpeed(seed);
}
float processFadeTime(float seed) {return 5.+seed*2.;}
float processLifetime(float seed) {
    return processArrival(seed)+processOrbitTime(seed)+processFadeTime(seed);
}
vec3 processPath(float age,float lane,float seed) {
    float arrival=processArrival(seed);
    float speed=processSpeed(seed);
    vec3 q=vec3(uEntry-uTime*.032,lane,1.);
    if(age>=arrival) {
        float travel=age-arrival;
        q.x+=travel*speed;
        q.y+=.035*sin(travel*.7+seed*31.)*smoothstep(0.,2.,travel);
        return ribbon(q);
    }
    vec3 end=ribbon(q);
    vec3 tangent=(ribbon(q+vec3(.002,0.,0.))-ribbon(q-vec3(.002,0.,0.)))/.004;
    vec3 start=vec3(uCenter,0.);
    start.xy+=vec2(sin(seed*83.),cos(seed*71.))* .045;
    vec2 side=normalize(vec2(-(end.y-start.y),end.x-start.x));
    vec3 c1=start+(end-start)*.40;
    c1.xy+=side*lane*.10;
    vec3 c2=end-tangent*speed*arrival/3.;
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
    float seed=gl_Color.b,alpha=gl_Color.g,size=1.0+seed*1.5,warmth=gl_Color.r;
    float energy=1.0, whiten=seed*.25;
    vSpark=0.0;
    if(uKind==0 || uKind==13 || uKind==17) {
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
    } else if(uKind==18 || uKind==19 || uKind==20) {
        float life=processLifetime(seed);
        float headAge=mod(uFlowTime+q.x*life,life);
        float birthTime=uFlowTime-headAge;
        float delay=uKind==20?q.z*.52:0.;
        float age=max(0.,headAge-delay);
        p=processPath(age,uBand+q.y,seed);
        float selected=1.-smoothstep(uEmission-.035,uEmission+.035,gl_Color.r);
        float birth=smoothstep(0.,.28,age);
        float fadeStart=processArrival(seed)+processOrbitTime(seed);
        float decay=1.-smoothstep(fadeStart,life,age);
        float front=smoothstep(-.35,.30,p.z);
        alpha*=selected*birth*decay*(.38+.62*front);
        alpha*=headAge>=delay?1.:0.;
        // Stop new births when a program closes; already born particles finish their orbit.
        alpha*=birthTime>=uBirthTime && birthTime<=uStopTime?1.:0.;
        size=uKind==18?1.4+seed*1.5:4.+seed*4.;
        energy=uKind==18?1.9:3.0;
        whiten=uKind==18?.10:.22;
        if(uKind==18 && age>3.7) {
            vec3 tangent=processPath(age+.005,uBand+q.y,seed)-p;
            vec3 across=processPath(age,uBand+q.y+.005,seed)-p;
            vec3 crossed=cross(tangent,across);
            vec3 normal=crossed*inversesqrt(max(dot(crossed,crossed),1.e-20));
            float rim=pow(clamp(1.-abs(normal.z),0.,1.),2.);
            energy=1.5+rim*3.2;whiten=.08+.28*rim;
        }
        if(uKind==19) vSpark=1.;
        if(uKind==20) {
            alpha*=pow(1.-q.z,1.5);
            size=1.4+2.*(1.-q.z);energy=2.1;whiten=.12;
        }
    }

    gl_Position=vec4(p.x/uAspect,p.y,clamp(-p.z*.4,-.9,.9),1.0);
    gl_PointSize=size*uScale*(uGlow>.5?6.0:1.0);
    vColor=mix(mix(uCool,uWarm,warmth),vec3(.90,.97,1.),whiten)*energy;
    if((uKind==0 || uKind==13 || uKind==15 || uKind==16 || uKind==17) && uReactive==1) {
        vec4 light=loadLight(p);
        vColor=mix(light.rgb,vec3(.90,.97,1.),whiten)*energy*(.75+.65*light.a);
    }
    if(uKind==11 || uKind==14)
        vColor=mix(uNodeColor,vec3(.93,.98,1.),whiten)*energy;
    if(uKind==18 || uKind==19 || uKind==20) {
        vec3 sourceColor=uReactive==1?uNodeColor:mix(uCool,uWarm,seed);
        vColor=mix(sourceColor,vec3(.90,.97,1.),whiten)*energy;
    }
    vAlpha=alpha*(uGlow>.5?.045:.90);
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
        c*=smoothstep(.20,.95,peak);
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
    vec3 color=film(scene*1.18+nearGlow*.80+farGlow*.55);
    // Preserve faint threads without lifting the black background.
    float dark=max(scene.r,max(scene.g,scene.b));
    color=mix(color,scene,.15*(1.-smoothstep(.20,.80,dark)));
    vec2 p=vUV*2.-1.;
    color*=1.-.12*dot(p,p);
    color*=.99+.01*cos(gl_FragCoord.y*3.14159265);
    gl_FragColor=vec4(color,1.);
}
"""
