import * as THREE from 'three';

// Decorative anatomy, deliberately separate from source-bound event packets.
export function createReefFilaments(scene,uniforms){
 let seed=813;const rand=()=>{seed=(seed*1664525+1013904223)>>>0;return seed/4294967296;};
 const positions=[],info=[],colors=[];
 const palette=['#718caa','#6aa8b2','#ad8aaf','#b0a68d'].map(c=>new THREE.Color(c));
 for(let root=0;root<28;root++){
  const side=root%2?1:-1,y=-.88+(Math.floor(root/2)/13)*1.76,length=.25+rand()*.4,phase=rand()*6.28;
  for(let branch=0;branch<3;branch++){
   const c=palette[root%4],bend=(rand()-.5)*.4;
   const point=u=>[side*(1.025-length*u),y+Math.sin(u*2.7+phase)*.1*u+bend*u*u+(branch-1)*.055*u*u,0];
   for(let j=0;j<36;j++)for(const u of [j/36,(j+1)/36]){positions.push(...point(u));info.push(u,phase,side,branch);colors.push(c.r,c.g,c.b);}
  }
 }
 const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('anatomy',new THREE.Float32BufferAttribute(info,4));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
 const material=new THREE.ShaderMaterial({uniforms,vertexColors:true,transparent:true,depthTest:false,depthWrite:false,blending:THREE.AdditiveBlending,
 vertexShader:`attribute vec4 anatomy;uniform float time;varying vec3 c;varying float u;varying float phase;void main(){u=anatomy.x;phase=anatomy.y;c=color;vec3 p=position;p.y+=sin(time*.32+u*5.+phase)*.018*u*u;p.x+=cos(time*.24+phase+u*3.)*.009*u*u;gl_Position=vec4(p,1.);}`,
 fragmentShader:`uniform float time,energy;uniform vec3 tint;varying vec3 c;varying float u;varying float phase;void main(){float vein=pow(max(0.,cos(u*10.-time*.72+phase)),18.);float taper=(1.-smoothstep(.72,1.,u));vec3 lit=mix(c,tint,energy*.55);gl_FragColor=vec4(lit*(.50+vein*(.7+energy)),taper*(.22+energy*.18));}`});
 const lines=new THREE.LineSegments(geometry,material);lines.frustumCulled=false;scene.add(lines);
 const dust=new Float32Array(180*3),traits=new Float32Array(180*3);
 for(let i=0;i<180;i++){dust.set([(rand()-.5)*2,(rand()-.5)*2,0],i*3);traits.set([rand()*6.28,rand(),rand()],i*3);}
 const dustGeo=new THREE.BufferGeometry();dustGeo.setAttribute('position',new THREE.BufferAttribute(dust,3));dustGeo.setAttribute('traits',new THREE.BufferAttribute(traits,3));
 const dustMat=new THREE.ShaderMaterial({uniforms,transparent:true,depthTest:false,depthWrite:false,blending:THREE.AdditiveBlending,
 vertexShader:`attribute vec3 traits;uniform float time,pixelScale;varying float a;void main(){vec3 p=position;p.x+=sin(time*.10+traits.x)*.035;p.y=mod(p.y+1.+time*(.006+traits.y*.01),2.)-1.;gl_Position=vec4(p,1.);gl_PointSize=(1.+traits.z*2.)*pixelScale;a=.18+traits.z*.25;}`,
 fragmentShader:`uniform float energy;uniform vec3 tint;varying float a;void main(){float r=length(gl_PointCoord-.5);if(r>.5)discard;gl_FragColor=vec4(mix(vec3(.6,.73,.8),tint,energy),exp(-r*r*22.)*a);}`});
 const spores=new THREE.Points(dustGeo,dustMat);spores.frustumCulled=false;scene.add(spores);
 return {update({mobile,quality}){geometry.setDrawRange(0,mobile||quality<.7?36*2*3*12:positions.length/3);dustGeo.setDrawRange(0,mobile||quality<.7?50:180);},drawCalls:2};
}
