import * as THREE from 'three';
import {createCosmicSignals,cosmicExcitement} from './cosmic-signals.mjs';

// One shared-renderer pass deforms the original art; no extra canvas or render target.
// Ambient currents are decoration. Event emphasis uses fresh source records only.
export function createReefScene(stage){
 const scene=new THREE.Scene(),camera=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
 const uniforms={art:{value:null},time:{value:0},aspect:{value:3},energy:{value:0},tint:{value:new THREE.Color('#b9c9e8')},mobile:{value:0}};
 let ready=false,elapsed=0,signal=null,signalAge=99,excitement=0,moving=false,paintKey='';
 const signals=createCosmicSignals();
 const status=document.createElement('div');status.id='reefStatus';status.className='reef-status';status.setAttribute('role','status');stage.after(status);
 stage.classList.remove('cosmic-flight');stage.classList.add('reef-currents');
 stage.parentElement.classList.remove('cosmic-shell');stage.parentElement.classList.add('reef-shell');
 const material=new THREE.ShaderMaterial({uniforms,transparent:true,depthTest:false,depthWrite:false,blending:THREE.AdditiveBlending,
 vertexShader:`varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}`,
 fragmentShader:`precision highp float;
 uniform sampler2D art;uniform float time,aspect,energy,mobile;uniform vec3 tint;varying vec2 vUv;
 void main(){
  vec2 uv=vUv;float t=time;float edge=smoothstep(.04,.48,abs(uv.x-.5));
  // Different currents bend individual coral features, rather than scaling the image.
  vec2 drift=vec2(sin(uv.y*13.+t*.48)+.45*sin(uv.x*23.-uv.y*9.-t*.31),cos(uv.x*16.-t*.39)+.4*sin(uv.y*27.+t*.57));
  uv+=drift*(.007+edge*.010)*(1.+energy*.65);
  uv+=vec2(sin(t*.13)*.006,cos(t*.17)*.004);
  float imageAspect=3.;vec2 cover=vec2(min(1.,aspect/imageAspect),min(1.,imageAspect/aspect));
  vec2 sampleUv=clamp((uv-.5)*cover+.5,.001,.999);
  vec3 color=texture2D(art,sampleUv).rgb;
  float ripple=sin(uv.x*34.+uv.y*23.-t*1.1)+sin(uv.x*19.-uv.y*31.+t*.73);
  float caustic=pow(max(0.,ripple*.5),4.);
  float sweep=pow(max(0.,sin(uv.x*8.+uv.y*5.-t*.65)),10.);
  float luminance=dot(color,vec3(.2126,.7152,.0722));
  color*=.88+caustic*(.3+energy*.35)+sweep*.20;
  color+=tint*(caustic*.045+energy*.13)*smoothstep(.06,.5,luminance);
  // Preserve the dark center and the existing accretion field beneath this pass.
  gl_FragColor=vec4(color,.88);
 }`});
 const mesh=new THREE.Mesh(new THREE.PlaneGeometry(2,2),material);mesh.frustumCulled=false;scene.add(mesh);
 new THREE.TextureLoader().load(new URL('./assets/organic-reef-v1.png',import.meta.url).href,texture=>{
  texture.colorSpace=THREE.SRGBColorSpace;texture.minFilter=THREE.LinearFilter;texture.generateMipmaps=false;
  uniforms.art.value=texture;ready=true;stage.dataset.reefReady='true';
 },undefined,()=>{stage.dataset.reefReady='false';});
 return {
  render(renderer){if(ready)renderer.render(scene,camera);},
  observe(raw,graph,options){const next=signals.observe(raw,graph,options);if(next){signal=next;signalAge=0;}},
  update(dt,{policy,mobile,aspect,mood,fresh,replay,pulse}){
   moving=policy.moving;if(moving){elapsed+=dt;signalAge+=dt;}
   excitement=cosmicExcitement(mood,signal,signalAge,{fresh,replay});
   uniforms.time.value=elapsed;uniforms.aspect.value=aspect;uniforms.mobile.value=mobile?1:0;
   uniforms.energy.value=fresh&&!replay?Math.max(excitement,pulse*.25):0;uniforms.tint.value.set(mood.color);
   const text=replay?'İÇE AKTARILAN KAYIT · OLAY VURGUSU KAPALI':!fresh?'KAYNAK GÜNCEL DEĞİL · OLAY VURGUSU KAPALI':signal&&signalAge<12?`${signal.label} ${signal.delta>0?'+':''}$${signal.delta.toFixed(2)} · ${signal.note}`:'ORGANİK AKINTILAR: DEKORATİF · OLAY VURGULARI: GERÇEK KAYIT';
   const key=text+'|'+moving;if(key!==paintKey){paintKey=key;status.textContent=text+(moving?'':' · HAREKET DURAKLATILDI');}
  },
  snapshot(){return {ready,moving,elapsed,excitement,texture:'organic-reef-v1.png',deformation:'spatial-uv-currents',extraDrawCalls:ready?1:0,extraRenderTargets:0};}
 };
}
