const finite=v=>typeof v==='number'&&Number.isFinite(v);
export function createCosmicSignals(){
 let previous=null,imported=false;
 return {observe(raw,graph,{replay=false}={}){
  if(replay!==imported){previous=null;imported=replay;}
  if(replay||!graph?.valid||!graph.fresh)return null;
  const n=graph.nodes.find(n=>n.id==='treasury'&&n.state==='current'),t=n?.raw;
  if(!t||!finite(t.equity_usd)||!Number.isFinite(Date.parse(t.observed_at)))return null;
  const next={at:Date.parse(t.observed_at),equity:t.equity_usd};
  if(previous&&next.at<=previous.at)return null;
  const prior=previous;previous=next;
  if(!prior||next.at-prior.at>600000)return null;
  const delta=next.equity-prior.equity;if(Math.abs(delta)<.005)return null;
  return {kind:delta>0?'shadow-rise':'shadow-fall',delta,at:t.observed_at,power:Math.min(1,Math.max(.3,Math.log1p(Math.abs(delta)/Math.max(1,prior.equity)*10000)/4)),label:delta>0?'SHADOW ÖZKAYNAK ARTIŞI':'SHADOW ÖZKAYNAK AZALIŞI',note:'Gerçekleşmiş kâr değil · özkaynak değişimi'};
 }};
}
export function cosmicBudget({mobile=false,quality=1}={}){
 return quality<.7?{stars:160,rocks:2,meteors:1,planets:1}:mobile?{stars:360,rocks:4,meteors:2,planets:1}:{stars:1400,rocks:10,meteors:7,planets:2};
}
export function cosmicExcitement(mood,signal,signalAge,{fresh=false,replay=false}={}){
 if(!fresh||replay)return 0;
 const strong=mood?.key==='opportunity'?1:0;
 const gain=signal?.kind==='shadow-rise'?signal.power*Math.max(0,1-signalAge/12):0;
 return Math.max(strong,gain);
}
