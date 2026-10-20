export function reasonFields(raw,path='',depth=0){
 if(!raw||typeof raw!=='object'||depth>3)return [];
 const fields=[];
 for(const [key,value]of Object.entries(raw)){
  const at=path+'/'+key;
  if(/reason|rationale|explanation|evidence|veto|gate|constraint|cost|threshold|confidence|score|risk|action|decision|direction|blocked/i.test(key)&&value!==null&&value!==undefined)fields.push({path:at,value});
  else if(typeof value==='object'&&!Array.isArray(value))fields.push(...reasonFields(value,at,depth+1));
 }
 return fields.slice(0,30);
}
export function createEventHistory(limit=180){
 const records=[];
 return {
  observe(graph,{replay=false,now=new Date().toISOString()}={}){
   if(!graph?.valid)return;
   for(const node of graph.nodes.filter(n=>n.id!=='brian')){
    const signature=JSON.stringify([node.at,node.raw]);
    const previous=[...records].reverse().find(r=>r.nodeId===node.id&&r.imported===replay);
    if(previous?.signature===signature)continue;
    const refs=[];
    for(const [key,value]of Object.entries(node.raw||{}))if(key.endsWith('_id')&&typeof value==='string'&&value){const matches=graph.nodes.filter(n=>n.id!==node.id&&n.record_id===value);if(matches.length===1)refs.push({path:node.path+'/'+key,label:matches[0].label,raw:structuredClone(matches[0].raw),targetPath:matches[0].path});}
    if(node.id==='asset'){const alpha=graph.nodes.find(n=>n.id==='alpha'&&n.raw?.asset_id===node.raw.asset_id);if(alpha)refs.push({path:'/alpha/asset_id',label:alpha.label,raw:structuredClone(alpha.raw),targetPath:alpha.path});}
    records.push({key:node.id+'|'+now+'|'+records.length,nodeId:node.id,label:node.label,path:node.path,sourceAt:graph.at,recordAt:node.at,recordId:node.record_id,sourceFresh:graph.fresh===true,observedState:node.state,firstSeenAt:now,arrivalAt:null,imported:replay,signature,raw:structuredClone(node.raw),refs});
   }
   if(records.length>limit)records.splice(0,records.length-limit);
  },
  list(id){return records.filter(r=>r.nodeId===id).slice().reverse();},
  arrive(id,recordAt,now=new Date().toISOString()){
   const r=[...records].reverse().find(r=>r.nodeId===id&&r.recordAt===recordAt&&!r.imported);if(r&&!r.arrivalAt)r.arrivalAt=now;
  }
 };
}
export function reviewStages(record){return [
 {label:'Kaynak kaydı',at:record.recordAt},
 {label:'Public yanıt zarfı',at:record.sourceAt},
 {label:record.imported?'İçe aktarma gözlemi':'Önizlemede gözlendi',at:record.firstSeenAt},
 ...(record.arrivalAt?[{label:'Görsel çekirdeğe ulaştı',at:record.arrivalAt}]:[])
];}
