// Only explicit membership, asset fields and exact record-ID matches are shown.
export function healthFor(node,graph,{replay=false,transportDegraded=false}={}){
 if(replay)return {key:'replay',label:'KAYIT İNCELEME',flow:0};
 if(transportDegraded)return {key:'offline',label:'BAĞLANTI YOK',flow:0};
 if(!graph?.valid||!graph.fresh)return {key:'stale',label:'YANIT ESKİ',flow:0};
 if(node.state==='failed')return {key:'failed',label:'KAYNAK HATASI',flow:0};
 if(node.state!=='current')return {key:'stale',label:'KAYIT ESKİ',flow:0};
 return {key:'current',label:'GÜNCEL',flow:1};
}
export function journeyFor(graph,selected){
 const nodes=graph?.nodes||[];
 const node=nodes.find(n=>n.id===selected&&n.id!=='brian')||[...nodes].filter(n=>n.id!=='brian'&&Number.isFinite(Date.parse(n.at||''))).sort((a,b)=>Date.parse(b.at)-Date.parse(a.at))[0];
 if(!node)return null;
 const relations=(graph.edges||[]).filter(e=>e.to===node.id||e.from===node.id).map(e=>({from:e.from,to:e.to,path:e.path,kind:e.type==='explicit-asset-reference'?'AÇIK VARLIK ALANI':'YANIT ÜYELİĞİ'}));
 for(const [key,value]of Object.entries(node.raw||{})){
  if(!key.endsWith('_id')||typeof value!=='string'||!value)continue;
  const matches=nodes.filter(n=>n.id!==node.id&&n.record_id===value);
  if(matches.length===1)relations.push({from:node.id,to:matches[0].id,path:node.path+'/'+key,kind:'KAYIT KİMLİĞİ EŞLEŞMESİ'});
 }
 return {node,relations,causalChain:false};
}
