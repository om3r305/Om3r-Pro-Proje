import { createClient } from "npm:@supabase/supabase-js@2";
import { requireCronAuth } from "../_shared/cron_auth.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const FOCUS=["XRPUSDT","ETHUSDT","DOGEUSDT"] as const;
const HORIZON_MIN=8;
const EVIDENCE="AGGRESSIVE_DIP_FORESIGHT_SHADOW";
const ORIGIN=/^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const EXACT=new Set(["https://monster-coins-pro-seven.vercel.app","https://monster-coins-pro-oemer-yildirim.vercel.app","http://localhost:3000","http://127.0.0.1:3000"]);
type J=Record<string,unknown>;
type Bar={t:number;o:number;h:number;l:number;c:number;v:number};

function cors(o:string|null){const a=o&&(EXACT.has(o)||ORIGIN.test(o))?o:"https://monster-coins-pro-oemer-yildirim.vercel.app";return{"access-control-allow-origin":a,"access-control-allow-headers":"content-type,x-brian-dashboard-key,x-brian-cron-key,authorization,apikey","access-control-allow-methods":"POST,OPTIONS","vary":"Origin"};}
function out(x:unknown,status=200,o:string|null=null){return new Response(JSON.stringify(x),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store",...cors(o)}});}
function n(v:unknown,d=0){const x=Number(v);return Number.isFinite(x)?x:d;}
function clip(v:number,a:number,b:number){return Math.max(a,Math.min(b,v));}
function pct(a:number,b:number){return b?(a/b-1)*100:0;}
function mean(xs:number[]){return xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:0;}
function shaHex(s:string){return crypto.subtle.digest("SHA-256",new TextEncoder().encode(s)).then(b=>[...new Uint8Array(b)].map(x=>x.toString(16).padStart(2,"0")).join(""));}
function same(a:string,b:string){if(a.length!==b.length)return false;let d=0;for(let i=0;i<a.length;i++)d|=a.charCodeAt(i)^b.charCodeAt(i);return d===0;}
async function auth(req:Request){
  const dashboard=(req.headers.get("x-brian-dashboard-key")??"").trim();
  if(dashboard){const q=await db.from("brian_dashboard_auth").select("dashboard_key_sha256,created_at").order("created_at",{ascending:false}).limit(1).maybeSingle();if(q.error||!q.data||!same(await shaHex(dashboard),String(q.data.dashboard_key_sha256)))throw Error("UNAUTHORIZED_DASHBOARD");return;}
  await requireCronAuth(req,db);
}
async function latestSession(){const q=await db.from("brian_dip_session_events").select("session_id,event_kind,requested_at").order("requested_at",{ascending:false}).order("event_id",{ascending:false}).limit(1).maybeSingle();if(q.error)throw q.error;if(!q.data||q.data.event_kind!=="START")return null;return String(q.data.session_id);}
async function latestState(sessionId:string){const q=await db.from("brian_dip_snapshots").select("observed_at,state").eq("session_id",sessionId).order("observed_at",{ascending:false}).limit(1).maybeSingle();if(q.error)throw q.error;return q.data?.state&&typeof q.data.state==="object"?q.data.state as J:{};}
async function bars(sym:string){let last:unknown=null;for(const host of ["https://api.binance.com","https://api1.binance.com","https://api3.binance.com"]){try{const r=await fetch(`${host}/api/v3/klines?symbol=${sym}&interval=1m&limit=100`,{headers:{accept:"application/json"},signal:AbortSignal.timeout(7000)});if(!r.ok)throw Error(`HTTP_${r.status}`);const a=await r.json();return (Array.isArray(a)?a:[]).map((x:unknown)=>Array.isArray(x)?({t:n(x[0]),o:n(x[1]),h:n(x[2]),l:n(x[3]),c:n(x[4]),v:n(x[5])}):null).filter((x:Bar|null):x is Bar=>Boolean(x&&x.c>0));}catch(e){last=e;}}throw Error(`BINANCE_KLINES:${sym}:${String(last)}`);}
function atr(rows:Bar[],p=14){const xs:number[]=[];for(let i=Math.max(1,rows.length-p);i<rows.length;i++){const r=rows[i],pc=rows[i-1].c;xs.push(Math.max(r.h-r.l,Math.abs(r.h-pc),Math.abs(r.l-pc)));}return mean(xs);}
function vote(d:J,name:string){const a=Array.isArray(d.votes)?d.votes:[];const x=a.find((v:unknown)=>v&&typeof v==="object"&&String((v as J).name)===name) as J|undefined;return n(x?.bias);}
function makeForecast(sym:string,rows:Bar[],state:J){
  const symbols=state.symbols&&typeof state.symbols==="object"?state.symbols as J:{},st=symbols[sym]&&typeof symbols[sym]==="object"?symbols[sym] as J:{},v4=st.v4&&typeof st.v4==="object"?st.v4 as J:{},d=v4.brainV5&&typeof v4.brainV5==="object"?v4.brainV5 as J:{};
  const price=rows.at(-1)?.c||n(st.last),a=atr(rows),atrPct=price?a/price*100:0;
  const closes=rows.slice(-10).map(x=>x.c),mom=closes.length>4?clip(pct(closes.at(-1)!,closes[0])/Math.max(.05,atrPct*2.2),-1,1):0;
  const edge=clip(n(d.edge),-1,1),structure=clip(vote(d,"structure"),-1,1),trend=clip(vote(d,"trend"),-1,1),micro=clip(vote(d,"microstructure"),-1,1);
  const signal=clip(.48*edge+.18*mom+.16*structure+.12*trend+.06*micro,-1,1);
  const direction=Math.abs(signal)<.10?"RANGE":signal>0?"UP":"DOWN";
  let confidence=clip(.44*n(d.confidence,.48)+.30*n(d.agreement,.48)+.26*n(d.brainQuality??d.quality,.48),.18,.95);
  if(String(d.action??"WAIT")==="WAIT")confidence*=.90;
  const gross=Math.max(0,n(d.grossEdgeBps))/100,targetPct=clip(Math.max(atrPct*Math.sqrt(HORIZON_MIN)*.72,gross*.38)*(.52+.48*confidence),Math.max(.04,atrPct*.6),2.2);
  const signed=direction==="RANGE"?signal*.28:direction==="UP"?1:-1;
  const candles:J[]=[];let prev=price;
  for(let i=1;i<=HORIZON_MIN;i++){
    const progress=i/HORIZON_MIN,damp=.72+.28*Math.sin(progress*Math.PI/2),wave=Math.sin(i*1.73+signal*2.4)*atrPct*.10*(1-progress*.35),center=price*(1+(signed*targetPct*progress*damp+wave)/100),o=prev,c=center,body=Math.max(price*atrPct*.055/100,Math.abs(c-o)*.28),h=Math.max(o,c)+body*(.65+.2*Math.abs(Math.sin(i))),l=Math.min(o,c)-body*(.65+.2*Math.abs(Math.cos(i)));candles.push({i,o,h,l,c});prev=c;
  }
  const peak=Math.max(price,...candles.map(x=>n(x.h))),trough=Math.min(price,...candles.map(x=>n(x.l))),predictedClose=n(candles.at(-1)?.c,price);
  return{symbol:sym,generated_at:new Date().toISOString(),horizon_min:HORIZON_MIN,price,direction,signal,confidence,setup:String(d.setup??"WAIT_DATA"),regime:String(d.regime??"WAIT_DATA"),brain_action:String(d.action??"WAIT"),brain_quality:n(d.brainQuality??d.quality),agreement:n(d.agreement),peak,trough,predicted_close:predictedClose,candles};
}
async function resolveDue(prices:Record<string,number>){const now=new Date().toISOString(),q=await db.from("brian_dip_foresight").select("id,symbol,entry_price,direction,predicted_close").is("resolved_at",null).lte("due_at",now).limit(60);if(q.error)throw q.error;for(const r of q.data??[]){const actual=prices[String(r.symbol)];if(!(actual>0))continue;const entry=n(r.entry_price),ret=pct(actual,entry),dir=String(r.direction),hit=dir==="RANGE"?Math.abs(ret)<=.12:dir==="UP"?ret>0:ret<0,err=Math.abs(pct(actual,n(r.predicted_close,entry)));await db.from("brian_dip_foresight").update({resolved_at:now,actual_price:actual,hit,abs_error_pct:err}).eq("id",r.id);}}
async function stats(sessionId:string,sym:string){const q=await db.from("brian_dip_foresight").select("hit,abs_error_pct").eq("session_id",sessionId).eq("symbol",sym).not("resolved_at","is",null).order("resolved_at",{ascending:false}).limit(50);if(q.error)throw q.error;const rows=q.data??[],hits=rows.filter(x=>x.hit===true).length;return{samples:rows.length,accuracy:rows.length?hits/rows.length:null,avg_error_pct:rows.length?mean(rows.map(x=>n(x.abs_error_pct))):null};}
async function persist(sessionId:string,f:J){const sym=String(f.symbol),cutoff=new Date(Date.now()-45_000).toISOString(),q=await db.from("brian_dip_foresight").select("id").eq("session_id",sessionId).eq("symbol",sym).gte("created_at",cutoff).order("created_at",{ascending:false}).limit(1).maybeSingle();if(q.error)throw q.error;if(q.data)return;const created=new Date(),due=new Date(created.getTime()+HORIZON_MIN*60_000);const ins=await db.from("brian_dip_foresight").insert({id:`foresight-${crypto.randomUUID()}`,session_id:sessionId,symbol:sym,created_at:created.toISOString(),due_at:due.toISOString(),entry_price:f.price,direction:f.direction,confidence:f.confidence,horizon_min:HORIZON_MIN,predicted_peak:f.peak,predicted_trough:f.trough,predicted_close:f.predicted_close,path:f.candles,evidence_class:EVIDENCE,shadow_only:true,live_execution:false});if(ins.error)throw ins.error;}

Deno.serve(async(req:Request)=>{const o=req.headers.get("origin");if(req.method==="OPTIONS")return new Response("ok",{headers:cors(o)});if(req.method!=="POST")return out({status:"METHOD_NOT_ALLOWED"},405,o);try{await auth(req);const sid=await latestSession();if(!sid)return out({status:"NO_ACTIVE_SESSION",forecasts:{},focus:FOCUS,shadow_only:true,live_execution:false},200,o);const state=await latestState(sid),forecasts:Record<string,J>={},prices:Record<string,number>={};for(const sym of FOCUS){const r=await bars(sym);const f=makeForecast(sym,r,state);forecasts[sym]=f;prices[sym]=n(f.price);}await resolveDue(prices);for(const sym of FOCUS){await persist(sid,forecasts[sym]);Object.assign(forecasts[sym],await stats(sid,sym));}return out({status:"OK",session_id:sid,focus:FOCUS,forecasts,meaning:{confidence:"yüksek daha güçlü mevcut tahmin",accuracy:"yüksek daha iyi geçmiş isabet"},shadow_only:true,live_execution:false},200,o);}catch(e){const m=String(e instanceof Error?e.message:e),u=m.includes("UNAUTHORIZED");console.error("brian-dip-foresight",m);return out({status:u?"UNAUTHORIZED":"FAILED_CLOSED",error:m,shadow_only:true,live_execution:false},u?401:500,o);}});
