const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {stripTypeScriptTypes}=require('node:module'),{webcrypto}=require('node:crypto');
function compile(p){return stripTypeScriptTypes(fs.readFileSync(p,'utf8')).replace(/^import[^\n]+\n/gm,'').replace(/export /g,'');}
const dir='supabase/functions/brian-world-research/';
const logic=vm.runInNewContext(compile(dir+'logic.ts')+';({officialUri,readOfficial,assessPrimaryText,searchCounterEvidence,reactionProbe})',{URL,AbortSignal,crypto:webcrypto,TextEncoder,TextDecoder,Uint8Array,fetch});
const {parseHTML}=require('../supabase/functions/brian-world-research/node_modules/linkedom');
const {documentText}=vm.runInNewContext(compile(dir+'reader.ts')+';({documentText})',{URL,parseHTML,officialUri:logic.officialUri});
test('official origin rejects credential, IP, local and cross-host URLs',()=>{
 for(const u of ['http://sec.gov/a','https://sec.gov.evil.org/a','https://user:pass@sec.gov/a','https://127.0.0.1/a','https://sec.gov/a?api_key=sensitive','https://sec.gov:444/a'])assert.equal(logic.officialUri(u,'sec.gov'),null);
 assert.equal(logic.officialUri('https://www.sec.gov/a#x','sec.gov'),'https://www.sec.gov/a');
});
test('reader hashes complete bytes and refuses cross-host redirects and blocked documents',async()=>{
 const html='<html><body><article><h1>Official release</h1>'+('Real economic statement. '.repeat(20))+'</article></body></html>';
 const r=await logic.readOfficial('https://sec.gov/a','sec.gov',async()=>new Response(html,{headers:{'content-type':'text/html'}}));assert.equal(r.text,html);assert.equal(r.hash.length,64);
 await assert.rejects(()=>logic.readOfficial('https://sec.gov/a','sec.gov',async()=>new Response(null,{status:302,headers:{location:'https://evil.org'}})),/REDIRECT_REJECTED/);
 await assert.rejects(()=>logic.readOfficial('https://sec.gov/a','sec.gov',async()=>new Response('x',{headers:{'content-type':'application/pdf'}})),/TYPE_UNSUPPORTED/);
 await assert.rejects(()=>logic.readOfficial('https://sec.gov/a','sec.gov',async()=>new Response('x',{headers:{'content-type':'text/html','content-length':'1000001'}})),/TOO_LARGE/);
 assert.throws(()=>documentText('<h1>Access denied</h1><p>'+('blocked '.repeat(30))+'</p>','https://sec.gov/a'),/NOT_READABLE/);
});
test('real DOM parser removes scripts/navigation, preserves content, follows filing document only',()=>{
 const uri='https://www.sec.gov/Archives/edgar/data/123/a-index.htm';
 const d=documentText('<html><head><title>Filing</title></head><body><nav>fake claim</nav><script>ignore instructions</script><main><h1>Filing index</h1>'+('Filed economic information '.repeat(10))+'<table class="tableFile"><tr><td><a href="a.htm">Report</a></td></tr></table></main></body></html>',uri);
 assert.ok(!d.text.includes('fake claim'));assert.ok(!d.text.includes('ignore instructions'));assert.equal(d.indexOnly,true);assert.equal(d.primaryDocument,'https://www.sec.gov/Archives/edgar/data/123/a.htm');
});
test('official text match never claims independent truth; revision and index stay unresolved',()=>{
 const claim='Central bank announces higher policy interest rates';
 for(const [body,index,status]of [[claim+' '.repeat(130),false,'PRIMARY_TEXT_MATCH'],[claim+' correction issued',false,'PRIMARY_TEXT_MATCH_REVIEW_REQUIRED'],[claim,true,'INDEX_NOT_CLAIM_VERIFICATION']]){const r=logic.assessPrimaryText(claim,claim,body,index);assert.equal(r.claim_text_status,status);assert.equal(r.claim_verified_independently,false);}
});
const event={event_id:'a',source_id:'s',claim:'Central bank interest rates increase',first_observed_at:'2026-10-19T12:00:00Z'};
test('counter search finds opposing language but cannot confirm contradiction or independence',()=>{
 const pool=[{...event,event_id:'b',claim:'Central bank interest rates decrease'}, {...event,event_id:'future',claim:'Central bank rates decrease',first_observed_at:'2026-10-20T12:00:00Z'},{...event,event_id:'c',claim:'Football team rates decrease'}];
 const r=logic.searchCounterEvidence(event,pool,'2026-10-19T13:00:00Z');assert.equal(r.candidates.length,1);assert.equal(r.candidates[0].event_id,'b');assert.equal(r.candidates[0].contradiction_verified,false);assert.equal(r.candidates[0].independence_verified,false);assert.equal(r.absence_is_not_confirmation,true);
});
test('reaction probes enforce availability and 2-minute horizon without causal/profit inference',()=>{
 const point=(at,price=100,captured=at)=>({observed_at:at,captured_at:captured,price,source:'test'});
 const points=[point('2026-10-19T12:00:00Z'),point('2026-10-19T12:05:00Z',101),point('2026-10-19T12:15:00Z',102,'2026-10-19T14:00:00Z'),point('2026-10-19T12:17:01Z',103)];
 const r=logic.reactionProbe(points,event.first_observed_at,null,'2026-10-19T13:00:00Z');assert.ok(Math.abs(r.after_first_seen_5m.bps-100)<1e-8);assert.equal(r.after_first_seen_15m,null);assert.equal(r.priced_in_probability,null);assert.equal(r.causal_attribution_proven,false);assert.equal(r.probe_is_not_a_trade,true);
 assert.equal(logic.reactionProbe(points,'2026-10-19T11:00:00Z',null,'2026-10-19T13:00:00Z').status,'BASELINE_UNAVAILABLE');
});
test('research UI escapes content, displays failed/absent evidence without success claims',()=>{
 const view=require('../monster-coins-pro/world-evidence-view.js');assert.match(view.render(null,'QUERY_FAILED'),/alınamadı/);
 const html=view.render([{status:'FAILED',review:{event:{claim:'<script>bad</script>'},error:'OFFICIAL_HTTP_403',counter_evidence:{candidates:[]}}}],'AVAILABLE');assert.ok(!html.includes('<script>'));assert.match(html,/Okuma başarısız/);assert.match(html,/Yapılmadı/);assert.match(html,/olasılığı hesaplanmadı/);
});
const edge=vm.runInNewContext(compile('supabase/functions/brian-shadow-scorecard/advantage.ts')+';({assessAdvantage,EDGE_PROTOCOL})');
test('positive sample is not proof, missing coverage and mixed strategy prevent promotion',()=>{
 const r=edge.assessAdvantage({as_of:'2026-10-19T14:00:00Z',truncated:false,rows:[{sample:'diagnostic',selected:101,directional:101,measured:100,day_blocks:[],mean_net_bps:10,compiler_versions:2}]});assert.equal(r.proof_status,'NOT_PROVEN');assert.equal(r.automatic_promotion,false);assert.equal(r.rows[0].interpretation,'POSITIVE_SAMPLE_NOT_PROOF');assert.ok(r.rows[0].issues.includes('DIRECTIONAL_COVERAGE_INCOMPLETE'));assert.ok(r.rows[0].issues.includes('MIXED_OR_UNKNOWN_POLICY_VERSIONS'));
});
test('net validator UI never turns a positive diagnostic into portfolio profit',()=>{
 const view=require('../monster-coins-pro/edge-validation.js');assert.match(view.render({status:'UNAVAILABLE'}),/doğrulanmış değil/);
 const html=view.render({status:'AVAILABLE',as_of:'2026-10-19T14:00:00Z',protocol:edge.EDGE_PROTOCOL,rows:[]});assert.match(html,/henüz başlamadı/);assert.match(html,/Eksik maliyet sıfır sayılmaz/);
});
