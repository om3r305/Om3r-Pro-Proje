const TREASURY_API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-evolution-treasury-status';
const KEY_STORAGE='mcp-dashboard-key-v1';
const $=id=>document.getElementById(id);
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function key(){return(localStorage.getItem(KEY_STORAGE)||'').trim();}
function clock(v){if(!v)return'—';try{return new Intl.DateTimeFormat('tr-TR',{timeZone:'Europe/Berlin',day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit'}).format(new Date(v));}catch{return'—';}}
function usd(v){const n=Number(v);return Number.isFinite(n)?new Intl.NumberFormat('tr-TR',{style:'currency',currency:'USD',minimumFractionDigits:2,maximumFractionDigits:2}).format(n):'—';}
function pct(v){const n=Number(v);return Number.isFinite(n)?`${(n*100).toFixed(1)}%`:'—';}
function bps(v){const n=Number(v);return Number.isFinite(n)?`${n.toFixed(2)} bps`:'—';}
function pill(text,tone=''){return`<span class="tr-pill ${tone}">${esc(text)}</span>`;}
function toast(message){const el=$('toast');if(!el)return;el.textContent=String(message);el.classList.add('show');clearTimeout(toast.t);toast.t=setTimeout(()=>el.classList.remove('show'),3200);}
function showLock(v){$('trLock')?.classList.toggle('show',v);}
async function api(){const k=key();if(!k)throw Error('Anahtar yok');const r=await fetch(TREASURY_API,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':k},body:'{}',cache:'no-store'});let data={};try{data=await r.json();}catch{}if(!r.ok){if(r.status===401){localStorage.removeItem(KEY_STORAGE);showLock(true);}throw Error(data.error||`HTTP ${r.status}`);}return data;}
function render(data){
  const s=data.summary||{},snap=data.snapshot||{},positions=data.positions||[],actions=data.actions||[],runs=data.runs||[];
  $('trSync').textContent=`Son durum ${clock(snap.observed_at||data.observed_at)}`;
  $('trState').textContent=`◉ ${data.status||'—'}`;
  const gate=s.promotion_gate_open===true;$('trGate').textContent=gate?'✓ Layer-4 gate açık':'🔒 Layer-4 gate kapalı';$('trGate').className=`cc-chip ${gate?'ok':''}`;
  $('trEquity').textContent=usd(s.equity_usd);$('trPnl').textContent=`Toplam P&L ${usd(s.total_pnl_usd)} · realized ${usd(s.realized_pnl_usd)}`;
  $('trCash').textContent=usd(s.cash_usd);$('trReserve').textContent=`Nakit ${pct(s.cash_reserve_pct)} · sabit rezerv kotası yok`;
  $('trDeployment').textContent=usd(s.deployment_usd);$('trDeploymentPct').textContent=`Deployment ${pct(s.deployment_pct)} · Brian conviction ile 0–100% karar verir`;
  $('trPositionsCount').textContent=String(s.open_positions??0);$('trCosts').textContent=`Kümülatif maliyet ${usd(s.cumulative_costs_usd)}`;

  const equity=Number(s.equity_usd);
  $('trPositions').innerHTML=positions.map(p=>{const dir=Number(p.direction)>0?'LONG':'SHORT';const tone=Number(p.direction)>0?'tr-pos-long':'tr-pos-short';const capital=Number(p.capitalUsd);const share=Number.isFinite(equity)&&equity>0&&Number.isFinite(capital)?capital/equity:null;return`<div class="tr-row"><div class="tr-title ${tone}">${esc(p.assetId)} · ${dir} ${pill(usd(p.capitalUsd))}</div><div class="tr-meta">Kasa payı ${share==null?'—':pct(share)} · giriş ${Number(p.entryPrice).toLocaleString('tr-TR')} · açılış ${clock(p.openedAt)}<br>Entry edge ${bps(p.entryExpectedNetEdgeBps)} · son edge ${bps(p.latestExpectedNetEdgeBps)} · maliyet ${bps(p.roundTripCostBps)}<br>Kaynak karar ${esc(p.sourceDecisionId)}</div></div>`;}).join('')||'<div class="tr-row">Açık sermaye pozisyonu yok. Treasury nakitte.</div>';

  $('trGateDetail').innerHTML=`<div class="tr-row"><div class="tr-title">${gate?pill('AUTHORIZED','ok'):pill('CLOSED','warn')}</div><div class="tr-meta">${esc(s.promotion_gate_reason||'—')}<br>Evidence: ${esc(s.promotion_gate_ref||'—')}<br>Gate kapanırsa yeni allocation durur ve mevcut SHADOW allocation flat edilir. Gate açıksa büyüklük sabit ticket ile değil; net edge + prospective reliability + evidence maturity ile belirlenir.</div></div>${(snap.blocked_reasons||[]).map(r=>`<div class="tr-row"><div class="tr-title">${pill('BLOCK','warn')} ${esc(r)}</div></div>`).join('')}`;

  $('trActionList').innerHTML=actions.slice(0,40).map(a=>{const tone=a.kind==='OPEN'?'ok':a.reason==='RISK_STOP'?'bad':'warn';return`<div class="tr-row"><div class="tr-title">${esc(a.asset_id)} ${pill(a.kind,tone)} ${Number(a.direction)>0?'LONG':'SHORT'}</div><div class="tr-meta">${clock(a.observed_at)} · ${usd(a.capital_usd)} @ ${Number(a.reference_price).toLocaleString('tr-TR')}<br>${esc(a.reason)} · edge ${bps(a.expected_net_edge_bps)} · cost ${usd(a.cost_usd)}</div></div>`;}).join('')||'<div class="tr-row">Henüz Treasury action yok.</div>';

  $('trRuns').innerHTML=runs.slice(0,20).map(r=>`<div class="tr-row"><div class="tr-title">${clock(r.started_at)} ${pill(r.status,r.status==='SUCCESS'?'ok':r.status==='FAILED'?'bad':'warn')}</div><div class="tr-meta">observed ${esc(r.observed_records)} · stored ${esc(r.stored_records)}${r.error_message?`<br>${esc(r.error_message)}`:''}</div></div>`).join('')||'<div class="tr-row">Treasury worker henüz çalışmadı.</div>';
}
async function refresh(){try{const data=await api();showLock(false);render(data);}catch(e){$('trSync').textContent='Treasury durumu alınamadı';toast(e.message||String(e));}}
function bind(){
  $('trUnlock')?.addEventListener('click',()=>{const v=($('trKey')?.value||'').trim();if(!v)return;localStorage.setItem(KEY_STORAGE,v);refresh();});
  $('trKey')?.addEventListener('keydown',e=>{if(e.key==='Enter')$('trUnlock')?.click();});
}
bind();if(!key())showLock(true);else refresh();setInterval(()=>{if(key())refresh();},15000);
