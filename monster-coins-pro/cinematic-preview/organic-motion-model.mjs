export function organicPolicy({fresh=false,replay=false,paused=false,visible=true,mobile=false,quality=1}={}){
 const moving=!paused&&visible,low=quality<.7;
 return {moving,flowing:moving&&fresh&&!replay,paths:low?4:mobile?8:24,breathing:moving&&!low,tier:low?'low':mobile?'mobile':'desktop'};
}
export function organicEventAllowed(mode,node,graph,{replay=false}={}){
 return !replay&&graph?.valid===true&&graph.fresh===true&&node?.state==='current'&&['live','revision'].includes(mode);
}
