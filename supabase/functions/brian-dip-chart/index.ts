const EXACT=new Set([
  'https://monster-coins-pro-seven.vercel.app',
  'https://monster-coins-pro-oemer-yildirim.vercel.app',
  'http://localhost:3000','http://127.0.0.1:3000',
]);
const ORIGIN=/^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const HOSTS=['https://fapi.binance.com','https://fapi1.binance.com','https://fapi2.binance.com','https://fapi3.binance.com'];
function cors(o:string|null){const a=o&&(EXACT.has(o)||ORIGIN.test(o))?o:'https://monster-coins-pro-seven.vercel.app';return{'access-control-allow-origin':a,'access-control-allow-methods':'GET,OPTIONS','access-control-allow-headers':'content-type','vary':'Origin'};}
function out(x:unknown,status=200,o:string|null=null){return new Response(JSON.stringify(x),{status,headers:{'content-type':'application/json; charset=utf-8','cache-control':'no-store',...cors(o)}});}
async function fetchJson(path:string){let last='BINANCE_USDM_UNAVAILABLE';for(const h of HOSTS){try{const r=await fetch(h+path,{headers:{accept:'application/json'},signal:AbortSignal.timeout(4500)});if(!r.ok){last=`${h} HTTP ${r.status}`;continue;}return await r.json();}catch(e){last=e instanceof Error?e.message:String(e);}}throw Error(last);}
Deno.serve(async(req:Request)=>{
  const o=req.headers.get('origin');if(req.method==='OPTIONS')return new Response('ok',{headers:cors(o)});if(req.method!=='GET')return out({status:'METHOD_NOT_ALLOWED'},405,o);
  try{const u=new URL(req.url),limit=Math.max(30,Math.min(240,Number(u.searchParams.get('limit')||180)||180));const raw=await fetchJson(`/fapi/v1/klines?symbol=ETHUSDT&interval=1m&limit=${limit}`);if(!Array.isArray(raw)||raw.length<20)throw Error('INVALID_KLINES');const candles=raw.map((x:any[])=>({t:Number(x[0]),o:Number(x[1]),h:Number(x[2]),l:Number(x[3]),c:Number(x[4]),v:Number(x[5]),ct:Number(x[6])}));return out({status:'OK',source:'BINANCE_USDM_PERP',symbol:'ETHUSDT',interval:'1m',generated_at:new Date().toISOString(),last_price:candles.at(-1)?.c??null,candles},200,o);}catch(e){return out({status:'FAILED',source:'BINANCE_USDM_PERP',error:e instanceof Error?e.message:String(e)},502,o);}
});
