const EXACT=new Set([
  'https://monster-coins-pro-seven.vercel.app',
  'https://monster-coins-pro-oemer-yildirim.vercel.app',
  'http://localhost:3000','http://127.0.0.1:3000',
]);
const ORIGIN=/^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const HOSTS=['https://fapi.binance.com','https://fapi1.binance.com','https://fapi2.binance.com'];
const REGION='eu-central-1';
function cors(o:string|null){const a=o&&(EXACT.has(o)||ORIGIN.test(o))?o:'https://monster-coins-pro-seven.vercel.app';return{'access-control-allow-origin':a,'access-control-allow-methods':'GET,OPTIONS','access-control-allow-headers':'content-type','vary':'Origin'};}
function out(x:unknown,status=200,o:string|null=null){return new Response(JSON.stringify(x),{status,headers:{'content-type':'application/json; charset=utf-8','cache-control':'no-store',...cors(o)}});}
async function fetchJson(path:string){
  let last='MARKET_UNAVAILABLE';
  for(const host of HOSTS){
    try{
      const r=await fetch(host+path,{method:'GET',headers:{accept:'application/json','user-agent':'Brian-DIP-V8.3-Dual-Shadow/1.0'},signal:AbortSignal.timeout(5500)});
      if(!r.ok){last='HTTP_'+r.status;if([418,429].includes(r.status))throw Error('RATE_LIMIT');continue;}
      const text=await r.text();if(!text.trim()){last='EMPTY_RESPONSE';continue;}
      try{return JSON.parse(text);}catch{last='INVALID_JSON';continue;}
    }catch(e){last=e instanceof Error?e.message:String(e);if(last==='RATE_LIMIT')break;}
  }
  throw Error('BINANCE_USDM:'+last);
}
Deno.serve(async(req:Request)=>{
  const o=req.headers.get('origin');if(req.method==='OPTIONS')return new Response('ok',{headers:cors(o)});if(req.method!=='GET')return out({status:'METHOD_NOT_ALLOWED'},405,o);
  const here=Deno.env.get('SB_REGION')||'';
  if(here!==REGION){
    const u=new URL(req.url);
    if(u.searchParams.get('forceFunctionRegion')!==REGION){u.searchParams.set('forceFunctionRegion',REGION);return new Response(null,{status:307,headers:{location:u.toString(),'cache-control':'no-store',...cors(o)}});}
  }
  try{
    const u=new URL(req.url),limit=Math.max(30,Math.min(240,Number(u.searchParams.get('limit')||180)||180));
    const raw=await fetchJson(`/fapi/v1/klines?symbol=ETHUSDT&interval=1m&limit=${limit}`);
    if(!Array.isArray(raw)||raw.length<20)throw Error('INVALID_KLINES');
    const candles=raw.map((x:any[])=>({t:Number(x[0]),o:Number(x[1]),h:Number(x[2]),l:Number(x[3]),c:Number(x[4]),v:Number(x[5]),ct:Number(x[6])}));
    if(candles.some(x=>![x.t,x.o,x.h,x.l,x.c,x.v,x.ct].every(Number.isFinite)||x.l<=0||x.h<x.l))throw Error('INVALID_CANDLE_VALUES');
    return out({status:'OK',source:'BINANCE_USDM_PERP',region:Deno.env.get('SB_REGION')||null,symbol:'ETHUSDT',interval:'1m',generated_at:new Date().toISOString(),last_price:candles.at(-1)?.c??null,candles},200,o);
  }catch(e){return out({status:'FAILED',source:'BINANCE_USDM_PERP',region:Deno.env.get('SB_REGION')||null,error:e instanceof Error?e.message:String(e)},502,o);}
});
