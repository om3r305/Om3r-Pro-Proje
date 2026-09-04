(() => {
  const API = 'https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-control-center';
  const KEY_STORAGE = 'mcp-dashboard-key-v1';
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  const clock = value => value ? new Intl.DateTimeFormat('de-DE',{timeZone:'Europe/Berlin',hour:'2-digit',minute:'2-digit',second:'2-digit'}).format(new Date(value)) : '—';
  const num = (value, digits=4) => Number.isFinite(Number(value)) ? Number(value).toLocaleString('en-US',{maximumFractionDigits:digits}) : '—';
  const bps = value => Number.isFinite(Number(value)) ? `${Number(value).toFixed(2)} bps` : '—';
  const pct = value => Number.isFinite(Number(value)) ? `${Number(value)>=0?'+':''}${(Number(value)*100).toFixed(2)}%` : '—';
  const actionClass = action => action === 'OPEN_LONG' ? 'pos' : action === 'OPEN_SHORT' ? 'neg' : action === 'VETO' ? 'amber' : '';

  function costMap(alpha){ return new Map((alpha?.costs || []).map(row => [String(row.quote_id), row])); }
  function comparisonMap(alpha){
    const map = new Map();
    for(const row of alpha?.phase37_comparisons || []){
      const id = String(row.decision_id);
      const list = map.get(id) || [];
      list.push(row); map.set(id,list);
    }
    return map;
  }
  function renderDecisions(alpha){
    const root = $('alphaDecisionList'); if(!root) return;
    const rows = alpha?.decisions || [];
    if(!rows.length){ root.innerHTML='<div class="row"><div><div class="rtitle">ALPHA kararı bekleniyor</div><div class="rmeta">1 dakikalık unified shadow compiler henüz receipt üretmedi.</div></div><span>—</span></div>'; return; }
    const costs = costMap(alpha), comparisons = comparisonMap(alpha);
    root.innerHTML = rows.slice(0,12).map(d => {
      const cost = costs.get(String(d.source_cost_quote_id || ''));
      const rel = (comparisons.get(String(d.decision_id)) || []).map(x => `${x.phase37_policy_kind}:${x.relationship}`).join(' · ');
      const groups = (d.support_groups || []).join(' + ') || 'destek grubu yok';
      const conflicts = (d.conflict_groups || []).length ? `<br>Conflict: ${esc((d.conflict_groups||[]).join(', '))}` : '';
      const quality = cost?.quality || d.metadata?.cost_quality || 'NO_COST';
      const reason = String(d.reason || '').slice(0,180);
      return `<div class="row decision ${String(d.action).startsWith('OPEN_')?'active':''}"><div><div class="rtitle">${clock(d.observed_at)} · ${esc(String(d.asset_id).replace('crypto:',''))} · ${esc(d.action)}</div><div class="rmeta">Score ${Number(d.evidence_score||0).toFixed(4)} · ${esc(groups)}<br>Ref ${num(d.observed_reference_price,8)} · ${esc(quality)} · ${bps(d.estimated_round_trip_cost_bps)}${rel?`<br>Phase3.7 ${esc(rel)}`:''}${conflicts}<br>${esc(reason)}</div></div><b class="${actionClass(d.action)}">${esc(d.action)}</b></div>`;
    }).join('');
  }
  function renderPositions(alpha){
    const root = $('alphaPositionList'); if(!root) return;
    const rows = alpha?.positions || [];
    if(!rows.length){ root.innerHTML='<div class="row"><div><div class="rtitle">ALPHA pozisyonu yok</div><div class="rmeta">OPEN oluştuğunda direction-only shadow book burada görünür.</div></div><span>—</span></div>'; return; }
    root.innerHTML = rows.map(p => {
      const side = Number(p.position) > 0 ? 'LONG' : 'SHORT';
      return `<div class="row"><div><div class="rtitle">${esc(String(p.asset_id).replace('crypto:',''))} · ${side}</div><div class="rmeta">Entry ${num(p.entry_price,8)} · ${clock(p.entry_ts)}<br>Son ${esc(p.last_event_type||'—')} · ref ${num(p.last_reference_price,8)}</div></div><b class="${side==='LONG'?'pos':'neg'}">${side}</b></div>`;
    }).join('');
  }
  function renderOutcomes(alpha){
    const root = $('alphaOutcomeList'); if(!root) return;
    const rows = alpha?.outcomes || [];
    if(!rows.length){ root.innerHTML='<div class="row"><div><div class="rtitle">Audit horizon bekleniyor</div><div class="rmeta">5m / 15m / 60m terminal noktası causal olarak çözülünce burada görünür. Eksik fiyat/cost varsa auditor fail-closed kalır.</div></div><span>—</span></div>'; return; }
    root.innerHTML = rows.slice(0,12).map(o => `<div class="row"><div><div class="rtitle">${esc(String(o.asset_id).replace('crypto:',''))} · ${Math.round(Number(o.horizon_seconds)/60)}m</div><div class="rmeta">${esc(o.classification)}<br>Gross ${pct(o.gross_return)} · Dir ${pct(o.direction_adjusted_return)} · MFE ${pct(o.mfe)} · MAE ${pct(o.mae)}<br>${clock(o.observed_at)} → ${clock(o.resolved_at)}</div></div><b class="${Number(o.direction_adjusted_return)>0?'pos':Number(o.direction_adjusted_return)<0?'neg':''}">${pct(o.direction_adjusted_return)}</b></div>`).join('');
  }
  function renderHealth(alpha){
    const status = $('healthAlpha'); const meta = $('healthAlphaMeta');
    if(!status || !meta) return;
    status.textContent = alpha?.status || 'STALE';
    status.className = alpha?.status === 'ONLINE' ? 'pos' : alpha?.status === 'DEGRADED' ? 'amber' : 'neg';
    meta.textContent = alpha?.last_decision_at ? `Son karar ${clock(alpha.last_decision_at)} · ${alpha.decision_age_seconds ?? '—'}s` : 'ALPHA decision yok';
    const compiler = $('alphaCompilerState');
    if(compiler) compiler.textContent = `${alpha?.status || 'STALE'} · ${alpha?.compiler_version || 'v2'}`;
  }
  async function refresh(){
    const key = localStorage.getItem(KEY_STORAGE) || '';
    if(!key) return;
    try{
      const response = await fetch(API,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':key},body:JSON.stringify({action:'status'})});
      if(!response.ok) return;
      const data = await response.json(); const alpha = data.alpha_v2;
      renderHealth(alpha); renderDecisions(alpha); renderPositions(alpha); renderOutcomes(alpha);
    }catch(error){
      const status=$('healthAlpha'); if(status){status.textContent='ERROR';status.className='neg';}
      const meta=$('healthAlphaMeta'); if(meta) meta.textContent=String(error?.message||error).slice(0,120);
    }
  }
  window.addEventListener('load',()=>{ refresh(); setInterval(()=>{ if(!document.hidden) refresh(); },15000); });
  document.addEventListener('visibilitychange',()=>{ if(!document.hidden) refresh(); });
})();
