import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import postgres from "npm:postgres@3.4.7";
const URL=Deno.env.get("SUPABASE_URL")!, KEY=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,KEY,{auth:{persistSession:false,autoRefreshToken:false}});
const DB_URL=Deno.env.get("SUPABASE_DB_URL")!;
const sql=postgres(DB_URL,{prepare:false,max:1,connect_timeout:5,idle_timeout:10});
const V="brian.alpha-catalyst-recheck.v1.cf-shadow.6", E="PROSPECTIVE_DEVELOPMENT_SHADOW";
const MICRO=new Set(["micro_velocity","micro_volume","micro_breakout","micro_reclaim","micro_taker_flow"]), EVENT=new Set(["catalyst_sentinel_reaction","world_event_reaction"]);
const out=(x:unknown,s=200)=>new Response(JSON.stringify(x),{status:s,headers:{"content-type":"application/json","cache-control":"no-store"}}), fin=(x:unknown,d=0)=>Number.isFinite(Number(x))?Number(x):d, clip=(x:number)=>Math.max(0,Math.min(1,x));
async function sha(s:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(s)));return [...d].map(x=>x.toString(16).padStart(2,"0")).join("")}
const CF_KEY_SHA256="8d348396f3da9bbffde9bef6f6f8d802af542bdcb3354743f92d3ece260fea51";
async function auth(r:Request){
  const k=(r.headers.get("x-brian-cloudflare-key")??"").trim();
  if(!k)throw Error("UNAUTHORIZED_CLOUDFLARE");
  const a=await sha(k),b=CF_KEY_SHA256;
  if(a.length!==b.length)throw Error("UNAUTHORIZED_CLOUDFLARE");
  let z=0;
  for(let i=0;i<a.length;i++)z|=a.charCodeAt(i)^b.charCodeAt(i);
  if(z)throw Error("UNAUTHORIZED_CLOUDFLARE");
}
function asset(x:unknown){const s=String(x??"").toUpperCase().replace(/^CRYPTO:/,"");if(!/^[A-Z0-9]{2,20}USDT$/.test(s))throw Error("INVALID_TRIGGER_ASSET");return `crypto:${s}`}
function hms(h:string){return h==="MICRO_1_5M"?300000:h==="FAST_5_30M"?1800000:h==="EVENT_DRIVEN"?7200000:h==="DAILY"?129600000:0}
function grp(g:string){return MICRO.has(g)?"intrabar_tape":EVENT.has(g)?"event_reaction":g}
function dir(n:number):-1|0|1{return n>0?1:n<0?-1:0}
async function evidence(a:string){
  if(!DB_URL)throw Error("DB_EVIDENCE:SUPABASE_DB_URL_MISSING");
  try{
    return await sql`
      select observation_id, independent_group, horizon, observed_at,
             direction, strength, confidence, reliability
      from brian_sensor_observations
      where asset_id=${a}
        and available=true
        and independent_group <> 'news_gdelt'
        and observed_at >= now() - interval '36 hours'
      order by observed_at desc
      limit 500
    `;
  }catch(e){throw Error("DB_EVIDENCE_DIRECT:"+String(e))}
}
function compile(rows:any[]){const now=Date.now(),by=new Map<string,any>(),ignored:string[]=[];for(const r of rows){const t=Date.parse(r.observed_at),h=hms(r.horizon),d=dir(Number(r.direction)),s=fin(r.strength,-1),c=fin(r.confidence,-1),re=fin(r.reliability,-1);if(!Number.isFinite(t)||!h||now-t>h||t>now+5000||!d||s<0||s>1||c<0||c>1||re<0||re>1){ignored.push(r.observation_id);continue}const g=grp(r.independent_group),q=s*c*re,p=by.get(g);if(!p||q>p.q||(q===p.q&&r.observed_at>p.observed_at)){if(p)ignored.push(p.observation_id);by.set(g,{...r,g,q})}else ignored.push(r.observation_id)}const v=[...by.values()];if(!v.length)return{action:"WAIT",direction:0,score:0,support:[],conflict:[],ids:[],ignored,reason:"no fresh directional independent evidence",veto:null};const ag=v.reduce((n,x)=>n+dir(x.direction)*x.q,0)/v.length,d=dir(ag),sup=v.filter(x=>dir(x.direction)===d),con=v.filter(x=>dir(x.direction)!==d),ratio=sup.length/v.length,breadth=.4+.6*Math.min(1,v.length/3),score=clip(Math.abs(ag)*(.5+.5*ratio)*breadth),base={direction:d,score,support:sup.map(x=>x.g).sort(),conflict:con.map(x=>x.g).sort(),ids:v.map(x=>x.observation_id).sort(),ignored};return !d||sup.length<2||score<.18?{...base,action:"WAIT",reason:`consensus below preregistered gate: support=${sup.length}, score=${score.toFixed(6)}`,veto:null}:{...base,action:d>0?"OPEN_LONG":"OPEN_SHORT",reason:"fresh independent evidence passed catalyst-priority preregistered consensus",veto:null}}
async function intrabar(a:string){try{const q=await sql`
  select event_id,direction,status,late_chase
  from brian_intrabar_reaction_events
  where asset_id=${a}
    and observed_at >= now() - interval '5 minutes'
  order by observed_at desc
  limit 1
`;return q[0]??null}catch(e){throw Error("DB_INTRABAR_DIRECT:"+String(e))}}
async function book(a:string){const s=a.slice(7),r=await fetch(`https://api.binance.com/api/v3/ticker/bookTicker?symbol=${s}`,{signal:AbortSignal.timeout(5000)});if(!r.ok)throw Error(`BOOK_${r.status}`);const p=await r.json(),b=fin(p.bidPrice),x=fin(p.askPrice);if(!(b>0)||x<b)throw Error("INVALID_BOOK");return{mid:(b+x)/2,spread:10000*(x-b)/((b+x)/2)}}
async function depth(a:string,d:-1|1,n:number,at:string){const s=a.slice(7),r=await fetch(`https://api.binance.com/api/v3/depth?symbol=${s}&limit=100`,{signal:AbortSignal.timeout(6000)});if(!r.ok)throw Error(`DEPTH_${r.status}`);const p=await r.json(),b=fin(p.bids?.[0]?.[0]),x=fin(p.asks?.[0]?.[0]);if(!(b>0)||x<b)throw Error("INVALID_DEPTH");const mid=(b+x)/2,lv=d>0?p.asks:p.bids;let rem=n,fill=0,qty=0;for(const z of lv){const pr=fin(z[0]),q=fin(z[1]);if(pr<=0||q<=0)continue;const take=Math.min(rem,pr*q);fill+=take;qty+=take/pr;rem-=take;if(rem<=1e-9)break}const ok=rem<=Math.max(1e-6,n*.001),vwap=qty?fill/qty:mid,slip=ok?Math.max(0,d>0?(vwap/mid-1)*10000:(1-vwap/mid)*10000):9999,spread=10000*(x-b)/mid,rt=20+spread+2*slip,id=await sha(`${V}|cost|${a}|${at}|${d}|${n}|${p.lastUpdateId??""}`);return{id,mid,spread,slip,rt,ok,fill,n,src:`binance_public_rest_depth:${s}:${p.lastUpdateId??""}`}}
async function macro(at:string){try{const q=await sql`
  select observation_id,observed_at,source_ids,reason,metadata
  from brian_sensor_observations
  where asset_id='global:MACRO'
    and sensor_family='official_macro_event'
    and available=true
    and observed_at >= ${new Date(Date.parse(at)-3600000).toISOString()}::timestamptz
    and observed_at <= ${at}::timestamptz
  order by observed_at desc
  limit 24
`;return{role:"context_only_no_direction_vote",direction_vote:0,event_count:q.length,events:q.map((r:any)=>({observation_id:r.observation_id,observed_at:r.observed_at,event_id:r.metadata?.event_id??null,correlation_id:r.metadata?.correlation_id??r.metadata?.event_id??null,title:r.metadata?.title??null,provenance_uri:r.metadata?.provenance_uri??null}))}}catch(e){throw Error("DB_MACRO_DIRECT:"+String(e))}}
Deno.serve(async r=>{if(r.method!=="POST")return out({error:"POST required"},405);try{await auth(r)}catch(e){return out({status:"UNAUTHORIZED",error:String(e)},401)}try{const b=await r.json().catch(()=>({})),event=String(b.event_id??b.trigger_event_id??"").trim(),alert=String(b.alert_id??b.trigger_alert_id??"").trim(),requestId=String(b.request_id??"").trim(),a=asset(b.asset_id??b.trigger_asset_id);if(!event)throw Error("MISSING_EVENT_ID");const rows=await evidence(a),c:any=compile(rows),ib=await intrabar(a).catch(()=>null);if((c.action==="OPEN_LONG"||c.action==="OPEN_SHORT")&&ib?.late_chase&&ib.status==="VETOED_LATE_CHASE"&&dir(ib.direction)===c.direction){c.action="VETO";c.reason="independent evidence aligned but current intrabar state is a late-chase veto";c.veto="LATE_CHASE"}const at=new Date().toISOString();let bk:any=null,cost:any=null;try{bk=await book(a)}catch{}if(c.action==="OPEN_LONG"||c.action==="OPEN_SHORT"){try{const n=c.score>=.8?20:c.score>=.6?10:c.score>=.4?5:3;cost=await depth(a,c.direction,n,at);if(!cost.ok){c.action="VETO";c.reason="visible liquidity cannot fill requested shadow notional";c.veto="INSUFFICIENT_VISIBLE_DEPTH"}}catch(e){c.action="VETO";c.reason=`synchronized L2 cost unavailable: ${String(e)}`;c.veto="COST_UNAVAILABLE"}}let cq:string|null=null,costRow:any=null;if(cost){if(requestId)cost.id=await sha(`${V}|cfreq|${requestId}|cost`);cq=cost.id;costRow={quote_id:cq,compiler_version:V,asset_id:a,observed_at:at,side:c.direction>0?"BUY":"SELL",requested_notional_usd:cost.n,filled_notional_usd:cost.fill,fill_ratio:clip(cost.fill/cost.n),fillable:cost.ok,fee_bps:10,spread_bps:cost.spread,depth_slippage_bps:cost.slip,one_way_cost_bps:10+cost.spread/2+cost.slip,estimated_round_trip_cost_bps:cost.rt,quality:"L2_OBSERVED",source_ids:[cost.src],reason:"catalyst-priority synchronized Binance L2 shadow cost",metadata:{trigger_event_id:event,trigger_alert_id:alert}}}const watchRows=await sql`select watch_id,status,reaction_score,direction from brian_catalyst_sentinel_watches where event_id=${event} and asset_id=${a} limit 1`;const watch={data:watchRows[0]??null};const id=requestId?await sha(`${V}|cfreq|${requestId}`):await sha(`${V}|${event}|${a}|${at}|${c.action}|${c.direction}|${c.score}`),m=await macro(at),row={decision_id:id,compiler_version:V,observed_at:at,asset_id:a,observed_reference_price:cost?.mid??bk?.mid??null,action:c.action,direction:c.direction,evidence_score:c.score,independent_group_count:c.support.length+c.conflict.length,support_groups:c.support,conflict_groups:c.conflict,source_observation_ids:c.ids,source_intrabar_event_ids:ib?.event_id?[ib.event_id]:[],source_cost_quote_id:cq,requested_virtual_notional_usd:cost?.n??0,gross_edge_bps:null,estimated_round_trip_cost_bps:cost?.rt??null,net_edge_bps:null,veto_reason:c.veto,reason:c.reason,metadata:{ignored_observation_ids:c.ignored,trigger_source:"catalyst_sentinel",cloudflare_request_id:requestId||null,trigger_event_id:event,correlation_id:event,trigger_alert_id:alert||null,trigger_watch_id:watch.data?.watch_id??null,catalyst_watch_status:watch.data?.status??null,catalyst_reaction_score:watch.data?.reaction_score??null,event_linkage:{event_id:event,correlation_id:event,asset_id:a},reference_price:{value:cost?.mid??bk?.mid??null,captured_at:at},official_macro_context:m,hard_risk_gates_bypassed:false,scheduling_priority_bypass_only:true,score_is_not_expected_return_bps:true},evidence_class:E,shadow_only:true,live_execution:false};try{await sql`select public.brian_cloudflare_alpha_shadow_write_v1(${sql.json(costRow)}::jsonb,${sql.json(row)}::jsonb) as result`}catch(e){throw Error("DB_ALPHA_WRITE_DIRECT:"+String(e))}return out({status:"CAPTURED",version:V,event_id:event,correlation_id:event,asset_id:a,decision_id:id,action:c.action,direction:c.direction,evidence_score:c.score,hard_risk_gates_bypassed:false,scheduling_priority_bypass_only:true,shadow_only:true,live_execution:false})}catch(e){return out({status:"FAILED_CLOSED",error:String(e),version:V,shadow_only:true,live_execution:false},500)}});