import * as THREE from 'three';
import {organicPolicy} from './organic-motion-model.mjs';

// Ambient carriers are decorative. Only observed live source revisions raise a pulse.
export function createOrganicAtmosphere(scene,stage){
 const bed=document.createElement('div');bed.className='organic-bed';bed.setAttribute('aria-hidden','true');stage.prepend(bed);stage.classList.add('organic-motion');
 let inView=true,elapsed=0,energy=0,lastKey='',policy=organicPolicy(),events=0,lastMode='none';const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 document.addEventListener('visibilitychange',()=>{lastKey='';if(document.hidden)stage.dataset.organicMotion='paused';});
 const observer=new IntersectionObserver(entries=>{inView=entries[0]?.isIntersecting!==false;});observer.observe(stage);
 const flow=new Float32Array(24*16*4),zero=new Float32Array(24*16*3);
 for(let path=0;path<24;path++)for(let tail=0;tail<16;tail++){const i=path*16+tail;flow.set([path*2.39996323,(path*.61803398875)%1,tail/15,(path%5)/5],i*4);}
 const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(zero,3));geometry.setAttribute('flow',new THREE.BufferAttribute(flow,4));
 const uniforms={t:{value:0},aspect:{value:1},pixelScale:{value:1},energy:{value:0},tint:{value:new THREE.Color('#d8e6ff')},eventTint:{value:new THREE.Color('#d8e6ff')},strength:{value:.35}};
 const material=new THREE.ShaderMaterial({uniforms,transparent:true,depthWrite:false,depthTest:false,blending:THREE.AdditiveBlending,
 vertexShader:`attribute vec4 flow;uniform float t;uniform float aspect;uniform float pixelScale;uniform float energy;varying float alpha;
 void main(){float head=fract(flow.y+t*(.055+flow.w*.025));float p=head-flow.z*.10;float present=step(0.,p);p=clamp(p,0.,1.);float r=mix(1.,.055,p);float a=flow.x+.24*sin(p*3.14159+flow.x);vec3 q=vec3(cos(a)*r*7.1*aspect,sin(a)*r*5.,-1.8);gl_Position=projectionMatrix*modelViewMatrix*vec4(q,1.);gl_PointSize=clamp((2.6+energy*2.)*pixelScale*(1.-flow.z*.45),1.,8.);alpha=present*(1.-flow.z)*smoothstep(0.,.07,p)*(1.-smoothstep(.88,1.,p));}`,
 fragmentShader:`uniform vec3 tint;uniform vec3 eventTint;uniform float energy;uniform float strength;varying float alpha;void main(){float r=length(gl_PointCoord-.5);if(r>.5)discard;float glow=exp(-r*r*24.);vec3 color=mix(tint,eventTint,energy);gl_FragColor=vec4(color*(.65+energy*1.3),glow*alpha*(strength+energy*.55));}`});
 const points=new THREE.Points(geometry,material);points.frustumCulled=false;points.renderOrder=-1;scene.add(points);
 return {
  pulse(color,mode){if(!policy.flowing)return;energy=1;events++;lastMode=mode;uniforms.eventTint.value.set(color);},
  update(dt,{fresh,replay,paused,mobile,quality,aspect,pixelScale,mood}){
   policy=organicPolicy({fresh,replay,paused:paused||reduced.matches,visible:inView&&!document.hidden,mobile,quality});
   const key=[policy.breathing,policy.tier,mood.key,mood.color].join('|');if(key!==lastKey){lastKey=key;stage.dataset.organicMotion=policy.breathing?'running':'paused';stage.dataset.organicTier=policy.tier;stage.style.setProperty('--organic-tint',mood.color);}
   if(policy.flowing){elapsed+=dt;energy=Math.max(0,energy-dt*.3);}else energy=0;
   uniforms.t.value=elapsed;uniforms.energy.value=energy;uniforms.aspect.value=aspect;uniforms.pixelScale.value=pixelScale;uniforms.tint.value.set(mood.color);uniforms.strength.value=.26+mood.power*.16;
   geometry.setDrawRange(0,policy.paths*16);points.visible=policy.flowing;
  },
  snapshot(){return {tier:policy.tier,breathing:policy.breathing,flowing:policy.flowing,carriers:policy.flowing?policy.paths*16:0,livePulses:events,lastPulseMode:lastMode,extraDrawCalls:policy.flowing?1:0,extraRenderTargets:0};}
 };
}
