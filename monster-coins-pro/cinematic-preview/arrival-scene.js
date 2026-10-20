import * as THREE from 'three';
// Preserve the preferred module artwork; coordinate actual source-arrival ripples in 3D.
export function createArrivalScene(parent,nodes,positions){
 const ripples=Array.from({length:3},()=>{const ring=new THREE.Mesh(new THREE.RingGeometry(1,1.025,128),new THREE.MeshBasicMaterial({color:0xe6f1ff,transparent:true,opacity:0,side:THREE.DoubleSide,blending:THREE.AdditiveBlending,depthWrite:false}));parent.add(ring);return ring;});
 return {update(t,elapsed,changed,camera){
  const age=elapsed-2.8,impact=age>=0&&age<3?Math.exp(-age*1.7):0;
  for(let j=0;j<ripples.length;j++){const ring=ripples[j],a=age-j*.16;ring.visible=a>=0&&a<2.6;ring.position.copy(positions[0]||new THREE.Vector3());ring.quaternion.copy(camera.quaternion);ring.scale.setScalar(1.1+Math.max(0,a)*2.7);ring.material.opacity=ring.visible?Math.exp(-a*1.9)*.8:0;}
  return impact;
 }};
}
