import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const VERSION="brian.multiasset-opportunity-engine.v3-registry";
const COLLECTOR_ID="brian-multiasset-opportunity-engine-v1";
const EXPORT_URL="https://dliediwlldojkfjzlznm.supabase.co/functions/v1/brian-realtime-multiasset-export";

type Mark={asset_id:string;asset_class:string;provider_time:string;price:number;return_5m:number|null;return_1h:number|null;session_state:string;data_latency_seconds:number;provider_quality:string;metadata?:Record<string,unknown>};
type EventRow={event_id:string;observed_at:string;published_at:string|null;event_kind:string;source_id:string;claim:string;primary_asset:string|null};
type Link={event:EventRow;weight:number;theme:string};

function out(b:unknown,s=200){return new Response(JSON.stringify(b),{status:s,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
function n(v:unknown,d=0){const x=Number(v);return Number.isFinite(x)?x:d}
function clip(v:number,lo=0,hi=1){return Math.max(lo,Math.min(hi,v))}
async function sha(s:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(s)));return [...d].map(b=>b.toString(16).padStart(2,"0")).join("")}
function errorText(e:unknown){return e instanceof Error?`${e.name}: ${e.message}`:String(e)}
async function auth(req:Request){
  const supplied=(req.headers.get("x-brian-cron-key")??"").trim();
  if(!supplied)throw new Error("UNAUTHORIZED_CRON");
  const q=await db.from("brian_dashboard_auth").select("cron_key_sha256").eq("auth_id","control-v3").single();
  if(q.error||!q.data)throw new Error("CRON_AUTH_UNAVAILABLE");
  const digest=await sha(supplied);
  const expected=String(q.data.cron_key_sha256??"");
  if(digest.length!==expected.length)throw new Error("UNAUTHORIZED_CRON");
  let diff=0;for(let i=0;i<digest.length;i++)diff|=digest.charCodeAt(i)^expected.charCodeAt(i);
  if(diff!==0)throw new Error("UNAUTHORIZED_CRON");
  return supplied;
}
function classify(claim:string,source:string){
  const x=`${claim} ${source}`.toLowerCase();
  const themes=new Set<string>();
  if(/\b(ecb|fed|federal reserve|bank of england|boe|bank of japan|boj|bank of canada|boc|interest rate|policy rate|monetary|inflation|yield|liquidity)\b/.test(x))themes.add("MONETARY");
  if(/\b(war|missile|houthi|sanction|russia|ukraine|iran|saudi|nato|geopolit|attack|conflict|air raid)\b/.test(x))themes.add("GEOPOLITICAL");
  if(/\b(oil|crude|brent|wti|opec|gas|pipeline|energy|refinery|blackout|power)\b/.test(x))themes.add("ENERGY");
  if(/\b(ai|artificial intelligence|nvidia|semiconductor|chip|gemini|microsoft|meta|google|data center)\b/.test(x))themes.add("AI_TECH");
  if(/\b(bank|banking|credit|financial|systemic|treasury|bond|debt|liquidity)\b/.test(x))themes.add("FINANCIAL");
  if(/\b(dollar|usd|euro|eur|yen|jpy|sterling|gbp|currency|fx)\b/.test(x))themes.add("FX");
  return [...themes];
}
function markThemes(mark:Mark):string[]{
  const raw=mark.metadata?.themes;
  return Array.isArray(raw)?raw.map(String).filter(Boolean):[];
}
function threshold(assetClass:string){
  return assetClass==="fx"?0.0007:
    assetClass==="index"?0.0015:
    assetClass==="commodity"?0.0020:
    assetClass==="etf"?0.0020:0.0025;
}
function costBps(assetClass:string){
  return assetClass==="fx"?6:
    assetClass==="index"?10:
    assetClass==="commodity"?14:
    assetClass==="etf"?12:18;
}
function ticket(score:number){return score>=.82?20:score>=.68?10:score>=.55?5:3}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  const startedAt=new Date().toISOString();
  let key="";
  try{key=await auth(req)}catch(e){return out({status:"UNAUTHORIZED",error:errorText(e)},401)}
  try{
    const exportResp=await fetch(EXPORT_URL,{
      method:"POST",
      headers:{"content-type":"application/json","x-brian-cron-key":key},
      body:"{}",
      signal:AbortSignal.timeout(12000)
    });
    const exportJson=await exportResp.json().catch(()=>({}));
    if(!exportResp.ok||exportJson?.status!=="SUCCESS")throw new Error(`realtime_export:${exportResp.status}:${String(exportJson?.status??"")}`);
    const marks=(Array.isArray(exportJson.marks)?exportJson.marks:[]) as Mark[];

    const since=new Date(Date.now()-6*60*60_000).toISOString();
    const eventsQ=await db.from("brian_world_event_frames")
      .select("event_id,observed_at,published_at,event_kind,source_id,claim,primary_asset")
      .gte("observed_at",since)
      .order("observed_at",{ascending:false})
      .limit(300);
    if(eventsQ.error)throw new Error(`world_events:${eventsQ.error.message}`);
    const events=(eventsQ.data??[]) as EventRow[];

    const linksByAsset=new Map<string,Link[]>();
    const marksByTheme=new Map<string,Mark[]>();
    for(const mark of marks){
      for(const theme of markThemes(mark)){
        const arr=marksByTheme.get(theme)??[];
        arr.push(mark);
        marksByTheme.set(theme,arr);
      }
    }
    for(const ev of events){
      const themes=classify(String(ev.claim??""),String(ev.source_id??""));
      if(!themes.length)continue;
      const eventAt=Date.parse(String(ev.published_at??ev.observed_at));
      const ageH=Math.max(0,(Date.now()-eventAt)/3600000);
      const sourceWeight=String(ev.source_id).startsWith("official:") ? 1 : String(ev.source_id).startsWith("institutional:") ? .85 : .72;
      const decay=Math.exp(-ageH/4);
      for(const theme of themes){
        for(const mark of marksByTheme.get(theme)??[]){
          const asset=String(mark.asset_id);
          const arr=linksByAsset.get(asset)??[];
          arr.push({event:ev,weight:sourceWeight*decay,theme});
          linksByAsset.set(asset,arr);
        }
      }
    }

    const decisions:any[]=[];
    for(const m of marks){
      const price=n(m.price);
      if(!(price>0))continue;
      const links=(linksByAsset.get(String(m.asset_id))??[]).sort((a,b)=>b.weight-a.weight).slice(0,8);
      const eventStrength=clip(links.reduce((s,l)=>s+l.weight,0)/2.2);
      const r5=n(m.return_5m,0),r1=n(m.return_1h,0);
      const reaction=Math.abs(r5)*.45+Math.abs(r1)*.55;
      const th=threshold(String(m.asset_class));
      const reactionScore=clip(reaction/(th*2.2));
      const signed=r1!==0?r1:r5;
      const direction=signed>0?1:signed<0?-1:0;
      const fresh=n(m.data_latency_seconds,1e9)<=15*60 && String(m.session_state).toUpperCase()==="REGULAR";
      const relevant=links.length>0&&eventStrength>=.18;
      const moving=reaction>=th;
      const groups:string[]=[];
      if(relevant)groups.push("event_link");
      if(moving)groups.push("market_reaction");
      const score=clip(.48*eventStrength+.52*reactionScore);
      let action="WAIT",veto:string|null=null;
      if(!fresh)veto="MARKET_CLOSED_OR_STALE";
      else if(!relevant)veto="NO_RELEVANT_EVENT";
      else if(!moving||direction===0)veto="REACTION_NOT_CONFIRMED";
      else if(groups.length<2)veto="INSUFFICIENT_INDEPENDENT_GROUPS";
      else if(score<.52)veto="SCORE_BELOW_GATE";
      else action=direction===1?"OPEN_LONG":"OPEN_SHORT";
      const observedAt=new Date().toISOString();
      const decisionId=await sha(`${VERSION}|${m.asset_id}|${observedAt.slice(0,16)}|${action}|${direction}|${score.toFixed(6)}`);
      decisions.push({
        decision_id:decisionId,observed_at:observedAt,asset_id:String(m.asset_id),asset_class:String(m.asset_class),
        provider_time:String(m.provider_time),observed_reference_price:price,action,direction:action==="WAIT"?0:direction,
        evidence_score:score,independent_group_count:groups.length,support_groups:groups,conflict_groups:[],
        linked_event_ids:[...new Set(links.map(l=>String(l.event.event_id)))].slice(0,12),
        requested_virtual_notional_usd:action==="WAIT"?0:ticket(score),
        estimated_round_trip_cost_bps:costBps(String(m.asset_class)),
        reason:action==="WAIT"
          ?`multi-asset watch: ${veto}; event_strength=${eventStrength.toFixed(3)} reaction=${(reaction*100).toFixed(3)}%`
          :`event-linked market reaction confirmed across event_link + market_reaction; score=${score.toFixed(3)}`,
        veto_reason:veto,session_state:String(m.session_state),data_latency_seconds:n(m.data_latency_seconds,1e9),
        metadata:{
          version:VERSION,provider_quality:m.provider_quality,event_strength:eventStrength,reaction_score:reactionScore,
          return_5m:r5,return_1h:r1,threshold:th,linked_themes:[...new Set(links.map(l=>l.theme))],
          direction_source:"OBSERVED_MARKET_REACTION_NOT_HEADLINE_GUESS",
          data_quality_gate:"REGULAR_SESSION_AND_15M_FRESHNESS",
          execution_grade:false,shadow_lane:"MULTIASSET_EVENT_REACTION"
        },
        evidence_class:"PROSPECTIVE_DEVELOPMENT_SHADOW",shadow_only:true,live_execution:false
      });
    }

    if(decisions.length){
      const ins=await db.from("brian_multiasset_alpha_decisions").insert(decisions);
      if(ins.error)throw new Error(`persist:${ins.error.message}`);
    }
    const finishedAt=new Date().toISOString();
    const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|SUCCESS`);
    await db.from("brian_collector_runs").insert({
      run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status:"SUCCESS",
      observed_records:marks.length+events.length,stored_records:decisions.length,degraded_sources:[],
      metadata:{version:VERSION,marks:marks.length,events:events.length,actionable:decisions.filter(d=>d.action!=="WAIT").length,shadow_only:true,live_execution:false},
      evidence_class:"PROSPECTIVE_DEVELOPMENT_SHADOW",shadow_only:true,live_execution:false
    });
    return out({
      status:"SUCCESS",version:VERSION,marks:marks.length,events:events.length,
      decisions:decisions.length,actionable:decisions.filter(d=>d.action!=="WAIT").length,
      results:decisions.map(d=>({asset_id:d.asset_id,action:d.action,evidence_score:d.evidence_score,veto_reason:d.veto_reason,session_state:d.session_state})),
      shadow_only:true,live_execution:false
    });
  }catch(e){
    const finishedAt=new Date().toISOString();
    const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|FAILED`);
    await db.from("brian_collector_runs").insert({
      run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status:"FAILED",
      observed_records:0,stored_records:0,degraded_sources:[],error_class:"MULTIASSET_OPPORTUNITY_ERROR",error_message:errorText(e).slice(0,1200),
      metadata:{version:VERSION,shadow_only:true,live_execution:false},evidence_class:"PROSPECTIVE_DEVELOPMENT_SHADOW",shadow_only:true,live_execution:false
    });
    return out({status:"FAILED_CLOSED",error:errorText(e),shadow_only:true,live_execution:false},500);
  }
});