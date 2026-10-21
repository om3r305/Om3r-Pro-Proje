import * as THREE from 'three';
import {createCosmicSignals,cosmicBudget,cosmicExcitement} from './cosmic-signals.mjs';

// Procedural scenery; neither celestial objects nor their motion represent assets.
export function createCosmicScene(scene,stage){
 const tracker=createCosmicSignals(),preview=document.documentElement.dataset.cosmicPreview==='true',world=new THREE.Group();scene.add(world);
 const status=document.createElement('div');status.className='cosmic-status';status.id='cosmicStatus';status.setAttribute('role','status');stage.after(status);stage.classList.add('cosmic-flight');
 let clock=0,travel=0,signal=null,signalAge=99,demoAge=99,signalCount=0,lastLabel='',excitement=0,budget=cosmicBudget(),imported=false;
 const artLabel=stage.querySelector('.organic-art-label');if(artLabel)artLabel.textContent='KOZMİK UÇUŞ / YILDIZ · GEZEGEN · METEOR / DEKOR';
 const U={time:{value:0},travel:{value:0},aspect:{value:1},pixels:{value:1},boost:{value:0},tint:{value:new THREE.Color('#8ebdff')}};
 const seedArray=(count,kind)=>{const data=new Float32Array(count*4);for(let i=0;i<count;i++){const s=((i*16807+kind*7919)%65521)/65521;data.set([((i*.754877666)%1)*2-1,((i*.569840291+.17)%1)*2-1,s,((i*.43821+.31)%1)],i*4);}return data;};
 const starGeo=new THREE.BufferGeometry();starGeo.setAttribute('position',new THREE.BufferAttribute(new Float32Array(1400*3),3));starGeo.setAttribute('seed',new THREE.BufferAttribute(seedArray(1400,3),4));
 const starMat=new THREE.ShaderMaterial({uniforms:U,transparent:true,depthWrite:false,depthTest:false,blending:THREE.AdditiveBlending,
 vertexShader:`attribute vec4 seed;uniform float time;uniform float travel;uniform float pixels;uniform float aspect;varying float light;void main(){float depth=.22+fract(seed.z-travel)*.9;vec2 q=seed.xy/depth;q.x+=sin(time*.04+seed.z*6.)*.035;q.y+=cos(time*.03+seed.w*5.)*.02;gl_Position=vec4(q,0.,1.);gl_PointSize=clamp((1.+seed.w*2.8)/depth*pixels,1.,9.);light=(.25+seed.w*.6)*smoothstep(.10,.28,length(vec2(q.x*aspect,q.y)));}`,
 fragmentShader:`varying float light;void main(){vec2 q=gl_PointCoord-.5;float r=length(q);if(r>.5)discard;float glow=exp(-r*r*30.);gl_FragColor=vec4(vec3(.72,.84,1.),glow*light);}`});
 const stars=new THREE.Points(starGeo,starMat);stars.frustumCulled=false;stars.renderOrder=-6;world.add(stars);
 const rotate=`mat3 spin(float a,float b){float c=cos(a),s=sin(a),d=cos(b),e=sin(b);return mat3(c,0.,s,0.,1.,0.,-s,0.,c)*mat3(1.,0.,0.,0.,d,-e,0.,e,d);}`;
 const surface=`varying vec3 surface;varying vec3 faceNormal;uniform float boost;uniform vec3 tint;float noise(vec3 p){return sin(p.x*7.+sin(p.z*9.))*sin(p.y*11.+p.z*4.);}
 void main(){vec3 n=normalize(faceNormal);float light=max(.04,dot(n,normalize(vec3(-.5,.65,1.))));float ridge=noise(surface*1.8)*.5+.5;float fine=noise(surface*5.)*.5+.5;float crater=pow(abs(sin(surface.x*16.+surface.y*12.)*sin(surface.z*19.-surface.x*8.)),7.);vec3 rock=mix(vec3(.025,.035,.07),vec3(.21,.31,.46),ridge);rock*=.55+fine*.4-crater*.25;float rim=pow(1.-max(0.,n.z),3.);vec3 c=rock*light*1.65+tint*rim*(.23+boost*.55);gl_FragColor=vec4(c,1.);}`;
 const planets=[];
 for(let i=0;i<2;i++){const uniforms={time:U.time,aspect:U.aspect,boost:U.boost,tint:U.tint,center:{value:new THREE.Vector2(i?.83:-.73,i?.35:-.05)},radius:{value:i?.37:.66},offset:{value:i*3.1}};
  const material=new THREE.ShaderMaterial({uniforms,depthWrite:false,depthTest:false,vertexShader:`uniform float time;uniform float aspect;uniform vec2 center;uniform float radius;uniform float offset;varying vec3 surface;varying vec3 faceNormal;${rotate}void main(){mat3 rot=spin(time*.09+offset,time*.025+.3);vec3 p=rot*position;faceNormal=rot*normal;surface=position;vec2 q=center+vec2(p.x/aspect,p.y)*radius;gl_Position=vec4(q,.2,1.);}`,fragmentShader:surface});
  const planet=new THREE.Mesh(new THREE.SphereGeometry(1,40,28),material);planet.frustumCulled=false;planet.renderOrder=-5;world.add(planet);planets.push(planet);
 }
 const rockBase=new THREE.IcosahedronGeometry(1,1),rockGeo=new THREE.InstancedBufferGeometry();rockGeo.index=rockBase.index;for(const [key,attr]of Object.entries(rockBase.attributes))rockGeo.setAttribute(key,attr);rockGeo.setAttribute('seed',new THREE.InstancedBufferAttribute(seedArray(10,7),4));
 const rockMat=new THREE.ShaderMaterial({uniforms:U,depthWrite:false,depthTest:false,vertexShader:`attribute vec4 seed;uniform float time;uniform float aspect;varying vec3 surface;varying vec3 faceNormal;${rotate}void main(){float p=fract(seed.z+time*(.027+seed.w*.018));mat3 rot=spin(time*(.4+seed.w)+seed.x*4.,time*.23+seed.y*3.);vec3 q=rot*position;float size=.018+seed.w*.055;vec2 center=vec2(mix(-aspect*1.4,aspect*1.4,p),seed.y*.9+sin(p*3.14+seed.x)*.15);center+=q.xy*size;gl_Position=vec4(center.x/aspect,center.y,.1,1.);surface=position*2.;faceNormal=rot*normal;}`,fragmentShader:surface});
 const rocks=new THREE.Mesh(rockGeo,rockMat);rocks.frustumCulled=false;rocks.renderOrder=-3;world.add(rocks);
 const meteorBase=new THREE.PlaneGeometry(1,1),meteorGeo=new THREE.InstancedBufferGeometry();meteorGeo.index=meteorBase.index;for(const [key,attr]of Object.entries(meteorBase.attributes))meteorGeo.setAttribute(key,attr);meteorGeo.setAttribute('seed',new THREE.InstancedBufferAttribute(seedArray(7,11),4));
 const meteorMat=new THREE.ShaderMaterial({uniforms:U,transparent:true,depthWrite:false,depthTest:false,blending:THREE.AdditiveBlending,
 vertexShader:`attribute vec4 seed;uniform float time;uniform float aspect;uniform float boost;varying vec2 ribbon;varying float fade;void main(){float p=fract(seed.z+time*(.07+seed.w*.04));vec2 dir=normalize(vec2(1.,-.45));vec2 q=vec2(mix(-aspect*1.3,aspect*1.3,p),mix(1.25,-1.25,p)+seed.y*.45);q-=dir*(1.-uv.x)*(.45+boost*.35);q+=vec2(-dir.y,dir.x)*(uv.y-.5)*(.012+boost*.012);gl_Position=vec4(q.x/aspect,q.y,0.,1.);ribbon=uv;fade=smoothstep(.02,.15,p)*(1.-smoothstep(.8,.98,p))*smoothstep(.12,.3,length(q));}`,
 fragmentShader:`uniform vec3 tint;uniform float boost;varying vec2 ribbon;varying float fade;void main(){float crossGlow=exp(-pow((ribbon.y-.5)*6.,2.));float tail=pow(ribbon.x,2.2);vec3 c=mix(vec3(.65,.77,1.),tint,.45);gl_FragColor=vec4(c*(1.+boost),crossGlow*tail*fade*(.65+boost*.35));}`});
 const meteors=new THREE.Mesh(meteorGeo,meteorMat);meteors.frustumCulled=false;meteors.renderOrder=-2;world.add(meteors);
 const renderStatus=(text)=>{if(text!==lastLabel){lastLabel=text;status.textContent=text;}};
 renderStatus('KOZMİK UÇUŞ / DEKOR · SHADOW KAYNAK GÖZLEMİ');
 if(preview){const button=document.createElement('button');button.id='cosmicDemo';button.textContent='Kozmik coşku örneği / SİMÜLASYON';button.onclick=()=>{demoAge=0;renderStatus('GÖRSEL DEMO / SİMÜLASYON — KÂR VEYA FIRSAT KAYDI DEĞİL');};document.querySelector('.settings-row').append(button);}
 return {
  observe(raw,graph,options){if(!!options?.replay!==imported||!graph?.fresh){signal=null;signalAge=99;}imported=!!options?.replay;const next=tracker.observe(raw,graph,options);if(next){signal={...next,message:next.label+' '+new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',signDisplay:'always'}).format(next.delta)+' · '+next.note};signalAge=0;signalCount++;}},
  update(dt,{policy,mobile,quality,aspect,pixelScale,mood,fresh,replay,pulse=0,visible=true}){
   budget=cosmicBudget({mobile,quality});const moving=policy.moving;if(moving){clock+=dt;signalAge+=dt;demoAge+=dt;}
   if(!fresh||replay)signalAge=99;const actual=Math.max(cosmicExcitement(mood,signal,signalAge,{fresh,replay}),fresh&&!replay?pulse*.12:0);excitement=demoAge<10?Math.max(actual,Math.sin(Math.min(1,demoAge/10)*Math.PI)):actual;
   if(moving)travel+=dt*(.065+excitement*.14);
   U.time.value=clock;U.travel.value=travel;U.aspect.value=aspect;U.pixels.value=pixelScale;U.boost.value=excitement;
   U.tint.value.set(['critical','fault','caution'].includes(mood.key)?mood.color:excitement>.1?'#ffd478':mood.color);
   world.visible=visible&&!document.hidden;starGeo.setDrawRange(0,budget.stars);rockGeo.instanceCount=budget.rocks;meteorGeo.instanceCount=budget.meteors;
   planets.forEach((p,i)=>{p.visible=i<budget.planets;p.material.uniforms.radius.value=mobile?.42:i?.37:.66;p.material.uniforms.center.value.set(mobile?-.92:i?.83:-.73,mobile?-.40:i?.35:-.05);});
   const key=moving?'running':'paused';if(stage.dataset.cosmicMotion!==key)stage.dataset.cosmicMotion=key;
   const glow=excitement>.15?'high':'normal';if(stage.dataset.cosmicEnergy!==glow){stage.dataset.cosmicEnergy=glow;stage.style.setProperty('--cosmic-color',U.tint.value.getStyle());}
   if(demoAge<10)renderStatus('GÖRSEL DEMO / SİMÜLASYON — KÂR VEYA FIRSAT KAYDI DEĞİL');
   else if(!fresh||replay)renderStatus(replay?'KAYIT İNCELEME / CANLI KAZANÇ VURGUSU YOK':'KOZMİK DEKOR / GÜNCEL VERİ BEKLENİYOR');
   else if(signal&&signalAge<12)renderStatus(signal.message);
   else renderStatus(mood.key==='opportunity'?'GÜÇLÜ ADAY / SHADOW · Kaynak STRONG bildirdi; kazanç doğrulanmadı':'KOZMİK UÇUŞ / DEKOR · Gerçek kayıtlar atmosferi etkiler');
  },
  snapshot(){return {...budget,excitement,shadowChanges:signalCount,signal:signal?.kind||'none',simulation:demoAge<10,extraDrawCalls:world.visible?3+budget.planets:0};}
 };
}
