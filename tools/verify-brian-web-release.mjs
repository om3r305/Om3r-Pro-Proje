// Read-only deployment smoke check. No credentials, runs, experiments or controls.
const input=process.argv[2];
let origin;
try{const u=new URL(input);if(u.protocol!=='https:'||u.origin!==input||!u.hostname.endsWith('.vercel.app'))throw Error();origin=u.origin;}
catch{console.error('Usage: node tools/verify-brian-web-release.mjs https://<deployment>.vercel.app');process.exit(2);}
const probes=[
 ['/intelligence','GET',200,'html'],
 ['/api/brian/brian-frontier-heartbeat-public','POST',200,'heartbeat'],
 ['/api/brian/brian-prediction-status','POST',401],
 ['/api/scout-feed','GET',401]
];
const results=await Promise.all(probes.map(async([path,method,expected,kind])=>{
 try{
  const r=await fetch(origin+path,{method,headers:{'content-type':'application/json'},...(method==='POST'?{body:'{}'}:{}),redirect:'error',signal:AbortSignal.timeout(20000)});
  let valid=r.status===expected;
  if(valid&&kind==='heartbeat'){const h=await r.json();valid=h.status==='OK'&&h.read_only===true&&h.shadow_only===true&&h.live_execution===false;}
  if(valid&&kind==='html')valid=(await r.text()).includes('resonance.js');
  return {path,status:r.status,expected,passed:valid};
 }catch(e){return {path,passed:false,error:e.name};}
}));
console.log(JSON.stringify({origin,passed:results.every(r=>r.passed),results},null,2));
if(results.some(r=>!r.passed))process.exitCode=1;
