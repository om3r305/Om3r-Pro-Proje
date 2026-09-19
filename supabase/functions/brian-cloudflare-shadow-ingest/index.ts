const NAME="brian-cloudflare-shadow-ingest";
const REPLACEMENT="brian-realtime-official-eye";
Deno.serve((_req:Request)=>new Response(JSON.stringify({
  status:"DECOMMISSIONED",
  name:NAME,
  replacement:REPLACEMENT,
  shadow_only:true,
  live_execution:false
}),{
  status:410,
  headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}
}));
