const EXACT=new Set([
  'https://monster-coins-pro-seven.vercel.app',
  'https://monster-coins-pro-oemer-yildirim.vercel.app',
  'http://localhost:3000','http://127.0.0.1:3000',
]);
const ORIGIN=/^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const FUTURES_HOSTS=['https://fapi.binance.com','https://fapi1.binance.com','https://fapi2.binance.com'];
const SPOT_HOSTS=['https://api.binance.com','https://api1.binance.com','https://api2.binance.com','https://api3.binance.com'];
const REGION='eu-central-1';
const PUBLIC='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-dip-chart';
function cors(o:string|null){const a=o&&(EXACT.has(o)||ORIGIN.test(o))?o:'https://monster-coins-pro-seven.vercel.app';return{'access-control-allow-origin':a,'access-control-allow-methods':'GET,OPTIONS','access-control-allow-headers':'content-type','vary':'Origin'};}
function out(x:unknown,status=200,o:string|null=null){return new Response(JSON.stringify(x),{status,headers:{'content-type':'application/json; charset=utf-8','cache-control':'no-store, max-age=0','pragma':'no-cache',...cors(o)}});}
async function fetchJson(hosts:string[],path:string,label:string){
  let last='MARKET_UNAVAILABLE';
  for(const host of hosts){
    try{
      const r=await fetch(host+path,{method:'GET',headers:{accept:'application/json','user-agent':'Brian-DIP-V8.3-Chart/1.0'},signal:AbortSignal.timeout(5500)});
      if(!r.ok){last='HTTP_'+r.status;if([418,429].includes(r.status))throw Error('RATE_LIMIT');continue;}
      const text=await r.text();if(!text.trim()){last='EMPTY_RESPONSE';continue;}
      try{return JSON.parse(text);}catch{last='INVALID_JSON';continue;}
    }catch(e){last=e instanceof Error?e.message:String(e);if(last==='RATE_LIMIT')break;}
  }
  throw Error(label+':'+last);
}
function normalize(raw:any[]){
  if(!Array.isArray(raw)||raw.length<20)throw Error('INVALID_KLINES');
  const candles=raw.map((x:any[])=>({t:Number(x[0]),o:Number(x[1]),h:Number(x[2]),l:Number(x[3]),c:Number(x[4]),v:Number(x[5]),ct:Number(x[6])}));
  if(candles.some(x=>![x.t,x.o,x.h,x.l,x.c,x.v,x.ct].every(Number.isFinite)||x.l<=0||x.h<x.l))throw Error('INVALID_CANDLE_VALUES');
  return candles;
}
Deno.serve(async(req:Request)=>{
  const o=req.headers.get('origin');if(req.method==='OPTIONS')return new Response('ok',{headers:cors(o)});if(req.method!=='GET')return out({status:'METHOD_NOT_ALLOWED'},405,o);
  const incoming=new URL(req.url),here=Deno.env.get('SB_REGION')||'',view=incoming.searchParams.get('view')==='spot1s'?'spot1s':'perp1m';
  if(here!==REGION&&incoming.searchParams.get('forceFunctionRegion')!==REGION){
    const target=new URL(PUBLIC);target.searchParams.set('forceFunctionRegion',REGION);target.searchParams.set('view',view);target.searchParams.set('limit',incoming.searchParams.get('limit')||(view==='spot1s'?'600':'180'));
    return new Response(null,{status:307,headers:{location:target.toString(),'cache-control':'no-store',...cors(o)}});
  }
  try{
    if(view==='spot1s'){
      const limit=Math.max(120,Math.min(1000,Number(incoming.searchParams.get('limit')||600)||600));
      const raw=await fetchJson(SPOT_HOSTS,`/api/v3/klines?symbol=ETHUSDT&interval=1s&limit=${limit}`,'BINANCE_SPOT');
      const candles=normalize(raw);
      return out({status:'OK',source:'BINANCE_SPOT',region:here||null,symbol:'ETHUSDT',interval:'1s',generated_at:new Date().toISOString(),last_price:candles.at(-1)?.c??null,candles},200,o);
    }
    const limit=Math.max(30,Math.min(240,Number(incoming.searchParams.get('limit')||180)||180));
    const raw=await fetchJson(FUTURES_HOSTS,`/fapi/v1/klines?symbol=ETHUSDT&interval=1m&limit=${limit}`,'BINANCE_USDM');
    const candles=normalize(raw);
    return out({status:'OK',source:'BINANCE_USDM_PERP',region:here||null,symbol:'ETHUSDT',interval:'1m',generated_at:new Date().toISOString(),last_price:candles.at(-1)?.c??null,candles},200,o);
  }catch(e){return out({status:'FAILED',source:view==='spot1s'?'BINANCE_SPOT':'BINANCE_USDM_PERP',region:here||null,error:e instanceof Error?e.message:String(e)},502,o);}
});
