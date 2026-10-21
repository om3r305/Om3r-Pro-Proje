import * as THREE from 'three';
import {createReefFilaments} from './reef-filaments.js';
import {createCosmicSignals,cosmicExcitement} from './cosmic-signals.mjs';

// One shared-renderer pass deforms the original art; no extra canvas or render target.
// Ambient currents are decoration. Event emphasis uses fresh source records only.
export function createReefScene(stage){
 const scene=new THREE.Scene(),camera=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
 const uniforms={art:{value:null},time:{value:0},aspect:{value:3},energy:{value:0},tint:{value:new THREE.Color('#b9c9e8')},pixelScale:{value:1}};
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
  // The reef silhouette stays fixed. Only narrow light veins travel over it.
  vec2 uv=vUv;float t=time;
  float imageAspect=3.;vec2 cover=vec2(min(1.,aspect/imageAspect),min(1.,imageAspect/aspect));
  vec2 sampleUv=clamp((uv-.5)*cover+.5,.001,.999);
  vec3 color=texture2D(art,sampleUv).rgb;
  float luminance=dot(color,vec3(.2126,.7152,.0722));
  float veins=pow(max(0.,sin(sampleUv.x*67.+sampleUv.y*39.-t*.24)),24.);
  color*=.94+veins*.08;
  color+=tint*energy*.065*smoothstep(.08,.45,luminance);
  // Preserve the dark center and the existing accretion field beneath this pass.
  gl_FragColor=vec4(color,.92);
 }`});
 const mesh=new THREE.Mesh(new THREE.PlaneGeometry(2,2),material);mesh.frustumCulled=false;scene.add(mesh);
 const filaments=createReefFilaments(scene,uniforms);
 new THREE.TextureLoader().load(new URL('./assets/organic-reef-v1.png',import.meta.url).href,texture=>{
  texture.colorSpace=THREE.SRGBColorSpace;texture.minFilter=THREE.LinearFilter;texture.generateMipmaps=false;
  uniforms.art.value=texture;ready=true;stage.dataset.reefReady='true';
 },undefined,()=>{stage.dataset.reefReady='false';});
 return {
  render(renderer){if(ready)renderer.render(scene,camera);},
  observe(raw,graph,options){const next=signals.observe(raw,graph,options);if(next){signal=next;signalAge=0;}},
  update(dt,{policy,mobile,quality,aspect,pixelScale,mood,fresh,replay,pulse}){
   moving=policy.moving;if(moving){elapsed+=dt;signalAge+=dt;}
   excitement=cosmicExcitement(mood,signal,signalAge,{fresh,replay});
   uniforms.time.value=elapsed;uniforms.aspect.value=aspect;uniforms.pixelScale.value=pixelScale;filaments.update({mobile,quality});
   uniforms.energy.value=fresh&&!replay?Math.max(excitement,pulse*.25):0;uniforms.tint.value.set(mood.color);
   const text=replay?'İÇE AKTARILAN KAYIT · OLAY VURGUSU KAPALI':!fresh?'KAYNAK GÜNCEL DEĞİL · OLAY VURGUSU KAPALI':signal&&signalAge<12?`${signal.label} ${signal.delta>0?'+':''}$${signal.delta.toFixed(2)} · ${signal.note}`:'ORGANİK AKINTILAR: DEKORATİF · OLAY VURGULARI: GERÇEK KAYIT';
   const key=text+'|'+moving;if(key!==paintKey){paintKey=key;status.textContent=text+(moving?'':' · HAREKET DURAKLATILDI');}
  },
  snapshot(){return {ready,moving,elapsed,excitement,texture:'organic-reef-v1.png',deformation:'stable-art / independent-filaments',filaments:84,spores:180,extraDrawCalls:ready?3:0,extraRenderTargets:0};}
 };
}
