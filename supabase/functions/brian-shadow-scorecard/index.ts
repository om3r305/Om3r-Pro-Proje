import postgres from 'npm:postgres@3.4.5';
import {summarize, alphaSummary, type Row} from './model.ts';

const headers={'content-type':'application/json; charset=utf-8','cache-control':'no-store',
  'access-control-allow-origin':'*','access-control-allow-headers':'content-type,x-brian-dashboard-key,x-brian-cron-key','access-control-allow-methods':'POST,OPTIONS'};
const response=(body:unknown,status=200)=>new Response(JSON.stringify(body),{status,headers});
async function digest(s:string){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(s)))].map(b=>b.toString(16).padStart(2,'0')).join('')}
function equal(a:string,b:string){if(a.length!==b.length)return false;let d=0;for(let i=0;i<a.length;i++)d|=a.charCodeAt(i)^b.charCodeAt(i);return d===0}
let cache: Row|null=null, cacheAt=0;

Deno.serve(async(req:Request)=>{
  if(req.method==='OPTIONS')return new Response(null,{status:204,headers});
  if(req.method!=='POST')return response({error:'METHOD_NOT_ALLOWED'},405);
  const key=(req.headers.get('x-brian-dashboard-key')||req.headers.get('x-brian-cron-key')||'').trim();
  if(!key)return response({error:'UNAUTHORIZED_DASHBOARD'},401);
  const sql=postgres(Deno.env.get('SUPABASE_DB_URL')!,{prepare:false,max:1,idle_timeout:1,connect_timeout:8,max_lifetime:30});
  try{
    const [auth]=await sql`select dashboard_key_sha256,cron_key_sha256 from public.brian_dashboard_auth where auth_id='control-v3'`;
    const hash=await digest(key);
    if(!auth||![auth.dashboard_key_sha256,auth.cron_key_sha256].some(v=>v&&equal(hash,String(v))))return response({error:'UNAUTHORIZED_DASHBOARD'},401);
    if(cache&&Date.now()-cacheAt<60000)return response({...cache,cache_age_seconds:(Date.now()-cacheAt)/1000});
    const payload=await sql.begin(async(tx:any)=>{
      await tx`set local statement_timeout='8s'`;
      await tx`set transaction isolation level repeatable read, read only`;
      const protocols=await tx`select * from public.brian_shadow_protocols order by engine_id limit 3`;
      const reports=[];
      for(const p of protocols){
        const points=await tx`select * from public.brian_shadow_observations where protocol_id=${p.protocol_id} order by captured_at limit 2201`;
        const cutoff=points.at(-1)?.source_at;
        let trades:any[]=[];
        if(cutoff&&p.engine_id==='treasury'){
          trades=await tx`select a.action_id id,a.observed_at,a.asset_id symbol,a.reason,a.source_decision_id,
            a.metadata->>'treasury_version' policy_version,a.shadow_only,a.live_execution,
            case when o.n=1 and o.asset_id=a.asset_id and o.direction=a.direction and a.direction in (-1,1)
              and o.capital=a.capital_usd and o.capital>0 and o.price>0 and a.reference_price>0 and a.cost_usd>=0 and o.cost>=0
              and o.opened_at>=${p.starts_at} and o.mode_ok
              then a.direction*(a.reference_price/o.price-1)*a.capital_usd-a.cost_usd-o.cost else null end net,
            d.net_edge_bps expected_net_bps,d.reason entry_reason
            from (select * from public.brian_treasury_shadow_actions where kind='EXIT' and observed_at>=${p.starts_at} and observed_at<=${cutoff} order by observed_at,action_id limit 1001) a
            left join lateral (select count(*) n,min(x.asset_id) asset_id,min(x.direction) direction,min(x.capital_usd) capital,
              min(x.reference_price) price,min(x.cost_usd) cost,min(x.observed_at) opened_at,
              bool_and(x.shadow_only and not x.live_execution) mode_ok
              from (select * from public.brian_treasury_shadow_actions where position_id=a.position_id and kind='OPEN' and observed_at<=a.observed_at limit 2) x) o on true
            left join public.brian_alpha_decisions d on d.decision_id=a.source_decision_id order by a.observed_at,a.action_id`;
        }else if(cutoff){
          trades=await tx`select event_id id,observed_at,symbol,pnl net,reason,metadata->>'policy_version' policy_version,
            metadata->'shadow_only' shadow_only,metadata->'live_execution' live_execution,
            metadata->>'entry_forecast_net_bps' expected_net_bps
            from public.brian_dip_multiasset_events where engine_id=${p.engine_id} and observed_at>=${p.starts_at}
            and observed_at<=${cutoff} and action='SELL' order by observed_at,event_id limit 1001`;
        }
        reports.push(summarize(p,points.slice(0,2200),trades.slice(0,1000),trades.length>1000||points.length>2200));
      }
      const start=protocols.find((p:Row)=>p.engine_id==='treasury')?.starts_at;
      const end=protocols.find((p:Row)=>p.engine_id==='treasury')?.ends_at;
      const decisions=start?await tx`select d.decision_id,d.asset_id,d.observed_at,d.action,d.reason,d.net_edge_bps,d.compiler_version,
        d.estimated_round_trip_cost_bps,o.direction_adjusted_return
        from (select decision_id,asset_id,observed_at,action,reason,net_edge_bps,compiler_version,estimated_round_trip_cost_bps
          from public.brian_alpha_decisions where observed_at>=${start} and observed_at<=least(now(),${end}::timestamptz) and shadow_only and not live_execution order by observed_at,decision_id limit 501) d
        left join public.brian_alpha_decision_outcomes o on o.decision_id=d.decision_id and o.horizon_seconds=3600 and o.shadow_only and not o.live_execution
        order by d.observed_at,d.decision_id`:[];
      return {version:'forward-scorecard-v1',generated_at:new Date().toISOString(),reports,alpha:alphaSummary(decisions),shadow_only:true,live_execution:false};
    });
    cache=payload;cacheAt=Date.now();return response(payload);
  }catch(error){console.error('shadow-scorecard',error instanceof Error?error.message:'QUERY_FAILED');return response({error:'SCORECARD_UNAVAILABLE',shadow_only:true,live_execution:false},503)}
  finally{await sql.end({timeout:1}).catch(()=>{})}
});
