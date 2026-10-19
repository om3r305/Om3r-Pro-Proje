export const RESEARCH_VERSION = 'brian.world-evidence-review.v2-domain-cooldown';
export function officialUri(value: unknown, domain: string): string | null {
  try {
    const u = new URL(String(value)), host = domain.toLowerCase().replace(/^www\./, '');
    if (!/^[a-z][a-z0-9.-]+\.[a-z]{2,}$/.test(host) || /(^|\.)(local|localhost|internal|invalid|test)$/.test(host)) return null;
    if (u.protocol !== 'https:' || u.username || u.password || u.port || ![host, 'www.' + host].includes(u.hostname.toLowerCase())) return null;
    if ([...u.searchParams.keys()].some(k => /token|password|secret|api.?key|signature|authorization/i.test(k))) return null;
    u.hash = ''; return u.toString();
  } catch { return null; }
}
export async function readOfficial(uri: string, domain: string, fetcher: typeof fetch = fetch) {
  let current = officialUri(uri, domain); if (!current) throw Error('OFFICIAL_URI_REJECTED');
  const signal = AbortSignal.timeout(7000);
  for (let hop = 0; hop <= 2; hop++) {
    const r = await fetcher(current, { redirect: 'manual', signal, headers: { accept: 'text/html,application/xhtml+xml,text/plain,application/xml;q=0.8', 'user-agent': 'BrianEvidence/1.0 (+https://github.com/om3r305/Om3r-Pro-Proje)' } });
    if ([301,302,303,307,308].includes(r.status)) {
      const next = officialUri(new URL(r.headers.get('location') || '', current).toString(), domain);
      await r.body?.cancel(); if (!next || hop === 2) throw Error('OFFICIAL_REDIRECT_REJECTED'); current = next; continue;
    }
    if (!r.ok) { await r.body?.cancel(); throw Error(`OFFICIAL_HTTP_${r.status}`); }
    const contentType = r.headers.get('content-type') || '';
    if (!/text\/html|text\/plain|application\/(xhtml\+xml|xml)/i.test(contentType)) { await r.body?.cancel(); throw Error('OFFICIAL_DOCUMENT_TYPE_UNSUPPORTED'); }
    if (Number(r.headers.get('content-length')) > 1_000_000) { await r.body?.cancel(); throw Error('OFFICIAL_DOCUMENT_TOO_LARGE'); }
    const reader = r.body?.getReader(); if (!reader) throw Error('OFFICIAL_BODY_EMPTY');
    const chunks: Uint8Array[] = []; let bytes = 0;
    try { while (true) { const p = await reader.read(); if (p.done) break; bytes += p.value.byteLength; if (bytes > 1_000_000) throw Error('OFFICIAL_DOCUMENT_TOO_LARGE'); chunks.push(p.value); } }
    finally { await reader.cancel().catch(() => {}); }
    const buffer = new Uint8Array(bytes); let offset = 0; for (const chunk of chunks) { buffer.set(chunk,offset); offset += chunk.length; }
    const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))].map(v=>v.toString(16).padStart(2,'0')).join('');
    return { uri: current, text: new TextDecoder().decode(buffer), bytes, hash, contentType, received_at: new Date().toISOString() };
  }
  throw Error('OFFICIAL_REDIRECT_REJECTED');
}
const normalized = (s: string) => s.normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu,' ').trim().replace(/\s+/g,' ');
const tokens = (s: string) => [...new Set(normalized(s).split(' ').filter(t=>t.length>=4&&!['this','that','with','from','have','will','their','about','after','before','into','says','said','more','than','been','were','which','would','could','press','release','announces','statement'].includes(t)))];
export function assessPrimaryText(claim: string, title: string, body: string, indexOnly = false) {
  const needle = normalized(claim), haystack = normalized(body), revision = /\b(retracted|retraction|erratum|withdrawn|correction|not accurate|incorrectly reported)\b/i.test(body);
  const exact = needle.length >= 24 && haystack.includes(needle);
  return { origin_status:'REGISTERED_OFFICIAL_ORIGIN_CONFIRMED', body_received:body.trim().length>=120,
    document_scope:indexOnly?'INDEX_PAGE_ONLY':'DOCUMENT_TEXT', claim_text_status:indexOnly?'INDEX_NOT_CLAIM_VERIFICATION':exact?(revision?'PRIMARY_TEXT_MATCH_REVIEW_REQUIRED':'PRIMARY_TEXT_MATCH'):normalized(title).includes(needle)&&needle.length>=24?'TITLE_MATCH_ONLY':'CLAIM_TEXT_NOT_MATCHED',
    claim_verified_independently:false, revision_language_present:revision,
    excerpt:body.trim().split(/\s+/).slice(0,24).join(' ').slice(0,240), title:title.slice(0,180), text_characters:body.length };
}
export type CounterEvent = { event_id:string; source_id:string; claim:string; provenance_uri?:string; published_at?:string; first_observed_at:string };
export function searchCounterEvidence(event: CounterEvent, pool: CounterEvent[], asOf: string) {
  const at = Date.parse(asOf), start = Date.parse(event.first_observed_at)-86400000, wanted = tokens(event.claim);
  const up=/\b(rise|rises|rising|increase|increases|higher|expands|growth|approval|approved|easing|cut|cuts)\b/i;
  const down=/\b(fall|falls|falling|decrease|decreases|lower|contracts|decline|rejection|rejected|tightening|hike|hikes)\b/i;
  const sign = (s:string) => up.test(s)&&!down.test(s)?1:down.test(s)&&!up.test(s)?-1:0;
  const direction=sign(event.claim), valid=pool.filter(p=>p.event_id!==event.event_id&&Number.isFinite(Date.parse(p.first_observed_at))&&Date.parse(p.first_observed_at)>=start&&Date.parse(p.first_observed_at)<=at);
  const candidates=valid.flatMap(p=>{const shared=tokens(p.claim).filter(t=>wanted.includes(t));const revision=/\b(retract|retracted|correction|erratum|withdrawn|incorrect)\b/i.test(p.claim);
    if(shared.length<2||(!revision&&(!direction||sign(p.claim)!==-direction)))return[];
    return[{...p,shared_terms:shared.slice(0,8),reason:revision?'REVISION_LANGUAGE_CANDIDATE':'OPPOSING_LANGUAGE_CANDIDATE',independence_verified:false,contradiction_verified:false}];}).slice(0,6);
  return { status:candidates.length?'CANDIDATES_FOUND':'SEARCHED_NO_MATCHES',scope:'BOUNDED_STORED_EVENT_HEADLINES_NOT_ALL_WEB',searched_records:valid.length,pool_limit:100,candidates,absence_is_not_confirmation:true,search_completed_at:asOf };
}
export type ReactionPoint = {observed_at:string;captured_at:string;price:number|string;source:string};
export function reactionProbe(points:ReactionPoint[], firstSeen:string, published:string|null, asOf:string) {
  const at=Date.parse(asOf),seen=Date.parse(firstSeen),pub=published?Date.parse(published):NaN;
  const valid=points.filter(p=>Number.isFinite(Date.parse(p.observed_at))&&Number.isFinite(Date.parse(p.captured_at))&&Date.parse(p.observed_at)<=at&&Date.parse(p.captured_at)<=at&&Number.isFinite(Number(p.price))&&Number(p.price)>0).sort((a,b)=>Date.parse(a.observed_at)-Date.parse(b.observed_at));
  const before=(target:number)=>valid.filter(p=>Date.parse(p.observed_at)<=target&&Date.parse(p.observed_at)>=target-120000).at(-1);
  const terminal=(target:number)=>valid.find(p=>Date.parse(p.observed_at)>=target&&Date.parse(p.observed_at)<=Math.min(at,target+120000));
  const delta=(a:ReactionPoint|undefined,b:ReactionPoint|undefined)=>a&&b?{bps:(Number(b.price)/Number(a.price)-1)*10000,from:a.observed_at,to:b.observed_at,source_from:a.source,source_to:b.source}:null;
  const base=Number.isFinite(seen)&&seen<=at?before(seen):undefined;
  return {status:base?'MEASURED_WITH_AVAILABLE_WINDOWS':'BASELINE_UNAVAILABLE',anchor:'BRIAN_FIRST_SEEN',first_seen_at:firstSeen,publication_time_valid:Number.isFinite(pub)&&pub<=seen,
    pre_publication_15m:Number.isFinite(pub)&&pub<=seen?delta(before(pub-900000),before(pub)):null,
    publication_to_first_seen:Number.isFinite(pub)&&pub<=seen?delta(before(pub),base):null,
    after_first_seen_5m:delta(base,terminal(seen+300000)),after_first_seen_15m:delta(base,terminal(seen+900000)),
    priced_in_probability:null,fully_priced_in_proven:false,causal_attribution_proven:false,probe_is_not_a_trade:true,
    horizon_seconds:900,horizon_kind:'PREDECLARED_MEASUREMENT_NOT_OPTIMAL_TRADE_HORIZON'};
}
