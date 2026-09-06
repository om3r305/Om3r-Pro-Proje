from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"expected source fragment missing in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


# 1) DIP: preserve/recover the last healthy universe and bridge high-quality V5
# brain-only decisions into the existing V4.1 SHADOW executor without bypassing
# cloud, cost, spread, heat, position-count or live-execution safety gates.
guard = ROOT / "monster-coins-pro" / "dip-expert-v4-runtime-guard.js"
append_once(
    guard,
    "BRIAN_DIP_LIVE_V6",
    r'''
/* BRIAN_DIP_LIVE_V6 — production recovery + V5 reasoner/executor bridge.
   SHADOW/PAPER ONLY. Does not add any live order endpoint and does not mutate MAIN/Phase 3.7. */
let v6LastHealthyUniverse = [];

function v6UniqueSymbols(items){
  const out=[];
  for(const s of items||[]){if(typeof s==='string'&&s.endsWith('USDT')&&s!=='BTCUSDT'&&!out.includes(s))out.push(s);}
  return out;
}
function v6RememberHealthyUniverse(){
  const ready=v6UniqueSymbols(v4Universe).filter(v4RuntimeSymbolReady);
  if(ready.length>=V4_RUNTIME_MIN_READY_MARKETS)v6LastHealthyUniverse=ready.slice(0,V4_UNIVERSE_SIZE);
}
function v6RecoverUniverse(){
  const open=Object.keys(states).filter(s=>states[s]?.pos);
  const configured=Array.isArray(session?.config?.symbols)?session.config.symbols:[];
  const runtimeReady=Object.keys(states).filter(s=>s!=='BTCUSDT'&&v4RuntimeSymbolReady(s));
  const currentReady=v6UniqueSymbols(v4Universe).filter(v4RuntimeSymbolReady);
  const rememberedReady=v6UniqueSymbols(v6LastHealthyUniverse).filter(v4RuntimeSymbolReady);
  const next=v6UniqueSymbols([...open,...currentReady,...rememberedReady,...configured,...runtimeReady]).slice(0,V4_UNIVERSE_SIZE);
  if(next.length<V4_RUNTIME_MIN_READY_MARKETS)return false;
  v4Universe=next;v4UniverseUpdatedAt=v4Now();v4Universe.forEach(v4Ensure);v6LastHealthyUniverse=[...next];
  if(!selected||!v4Universe.includes(selected))selected=v4Universe[0]||'ETHUSDT';
  if($('radarStatus'))$('radarStatus').textContent=running?'BRIAN V5 LIVE':`DATA ${next.length}/${V4_UNIVERSE_SIZE}`;
  try{renderRadar();}catch{}
  return true;
}

const _v6PriorLoadHistory=v4LoadHistory;
v4LoadHistory=async function(){
  const before=v6UniqueSymbols(v4Universe);
  if(before.length>=V4_RUNTIME_MIN_READY_MARKETS)v6LastHealthyUniverse=[...before];
  try{
    const result=await _v6PriorLoadHistory();
    if(!v4Universe.length)v6RecoverUniverse();
    v6RememberHealthyUniverse();
    return result;
  }catch(e){
    if(before.length){v4Universe=[...before];v4UniverseUpdatedAt=v4Now();}
    v6RecoverUniverse();
    throw e;
  }finally{
    if(!v4Universe.length)v6RecoverUniverse();
    try{renderRadar();}catch{}
  }
};
historyLoad=v4LoadHistory;

// The V5 layer is loaded after this guard. Install the bridge on the next task so
// it sees V5 globals and can extend only the reasoner entry path.
setTimeout(()=>{
  if(typeof v5ReasonerEntry!=='function'||typeof v5ControllerDecision!=='function')return;
  v5ReasonerEntry=function(st,ctx,p,d){
    if(!st||!ctx||!d||!['BUY','SELL'].includes(d.action))return false;
    if(d.hardDrift||d.ood||(d.vetoReasons||[]).length)return false;
    const cost=Math.max(0,Number(d.costBps||0)),net=Number(d.netEdgeBps||0);
    const setup=String(d.setup||'');
    const confFloor=['PULLBACK_CONTINUATION','TREND_EXHAUSTION'].includes(setup)?.62:.64;
    if(Number(d.confidence||0)<confFloor||Number(d.agreement||0)<.56)return false;
    if(!(net>Math.max(3,cost*.45))||Number(ctx.edgeRatio||0)<V4_COST_EDGE_MULT)return false;

    if(d.action==='BUY'){
      const allowed=['LIQUIDITY_SWEEP_REVERSAL','RANGE_REJECTION','PULLBACK_CONTINUATION','BREAKOUT_RETEST_CONTINUATION','DIP_RECLAIM'];
      if(!allowed.includes(setup))return false;
      const flowOk=Number(ctx.flow?.ofi||0)>=.03&&Number(ctx.bk?.pressure||0)>=1.01;
      const trendFloor=['LIQUIDITY_SWEEP_REVERSAL','RANGE_REJECTION','DIP_RECLAIM'].includes(setup)?-.28:-.12;
      if(!flowOk||Number(ctx.btcLongRisk||0)<=-.42||Number(ctx.htfLong||0)<=trendFloor)return false;
      return v4Open(st,ctx,'LONG',`V5_${setup}`);
    }

    if(!v4FuturesSymbols.has(st.symbol))return false;
    const pctx=v4Context(st.symbol,'USDM_PERP');if(!pctx)return false;
    const pd=v5ControllerDecision(st.symbol,pctx,pctx.bk.mid||Number(p));
    v5DecisionBySymbol[st.symbol]=pd;st.v4.brainV5=v5SlimDecision(pd);
    if(pd.action!=='SELL'||pd.hardDrift||pd.ood||(pd.vetoReasons||[]).length)return false;
    const psetup=String(pd.setup||setup),allowed=['TREND_EXHAUSTION','DOWNTREND_BREAK','FAILED_BREAK_REVERSAL'];
    if(!allowed.includes(psetup))return false;
    const pcost=Math.max(0,Number(pd.costBps||0)),pnet=Number(pd.netEdgeBps||0);
    if(Number(pd.confidence||0)<.62||Number(pd.agreement||0)<.56||!(pnet>Math.max(3,pcost*.45)))return false;
    if(Number(pctx.edgeRatio||0)<V4_COST_EDGE_MULT)return false;
    const flowOk=Number(pctx.flow?.ofi||0)<=-.03&&Number(pctx.bk?.pressure||1)<=.99;
    const htfOk=Number(pctx.htfShort||0)>-.08||psetup==='FAILED_BREAK_REVERSAL';
    const fundingOk=Number(pctx.fundingRate||0)>-0.0005;
    if(!flowOk||!htfOk||!fundingOk)return false;
    return v4Open(st,pctx,'SHORT',`V5_${psetup}`);
  };
},0);

setInterval(()=>{
  if(!session||session.status!=='RUNNING')return;
  if(v4Universe.length>=V4_RUNTIME_MIN_READY_MARKETS){v6RememberHealthyUniverse();return;}
  v6RecoverUniverse();
},3000);
'''
)

# 2) Dashboard: make General Overview useful by defaulting the performance view
# to the actual live ALPHA directional-shadow telemetry instead of a frozen/inactive
# Phase 3.7 tracking snapshot that can legitimately remain all zeroes.
index = ROOT / "monster-coins-pro" / "index.html"
replace_once(
    index,
    '<select id="policyView"><option value="PROFIT">Kâr Modu</option><option value="NATIVE">Native Kontrol</option></select>',
    '<select id="policyView"><option value="ALPHA" selected>ALPHA Canlı</option><option value="PROFIT">Kâr Modu (Frozen 3.7)</option><option value="NATIVE">Native Kontrol (Frozen 3.7)</option></select>',
)

js = ROOT / "monster-coins-pro" / "dashboard.js"
replace_once(
    js,
    "function currentPolicy(){const v=$('policyView')?.value||'PROFIT';return v==='NATIVE'?'NATIVE':'PROFIT';}",
    "function currentPolicy(){const v=$('policyView')?.value||'ALPHA';return v==='NATIVE'?'NATIVE':v==='PROFIT'?'PROFIT':'ALPHA';}",
)

insert_marker = "function renderOverview(data){const s=data.session;const snap=s?data.policies?.[currentPolicy()]||null:null;"
alpha_helper = r'''function setOverviewLabels(alphaMode){
  const labels=['kpiEquity','kpiPnl','kpiPositions','kpiWin','kpiActions','kpiCost'];
  const alpha=['ALPHA Durumu','Yön Dağılımı','Açık Shadow Yön','Outcome Pozitif','ALPHA Aksiyon','Ort. Maliyet'];
  const legacy=['Shadow Bakiye','Oturum K/Z','Açık Pozisyon','Başarı Oranı','Shadow Aksiyon','İşlem Maliyeti'];
  labels.forEach((id,i)=>{const el=$(id)?.previousElementSibling;if(el)el.textContent=(alphaMode?alpha:legacy)[i];});
  const chart=$('equityChart')?.closest('.cc-card');if(chart){const title=chart.querySelector('.cc-card-title'),sub=chart.querySelector('.cc-card-sub');if(title)title.textContent=alphaMode?'ALPHA Karar Aktivitesi':'Shadow Equity';if(sub)sub.textContent=alphaMode?'Gerçek ALPHA directional-shadow state ve son karar akışı · direction-only/no notional':'Seçili takip modunun sanal bakiye eğrisi';}
  const pos=$('positionList')?.closest('.cc-card');if(pos){const sub=pos.querySelector('.cc-card-sub');if(sub)sub.textContent=alphaMode?'ALPHA position book · LONG/SHORT yön state':'Seçili takip modunun mevcut shadow pozisyonları';}
  const trades=$('tradeList')?.closest('.cc-card');if(trades){const sub=trades.querySelector('.cc-card-sub');if(sub)sub.textContent=alphaMode?'Son OPEN_LONG / OPEN_SHORT ALPHA kararları':'Telefon için yatay tablo yerine okunabilir kart akışı';}
}
function renderAlphaActivity(decisions,active){
  const root=$('equityChart');if(!root)return;const rows=(decisions||[]).slice(0,24).reverse();
  if(!rows.length){root.innerHTML='<div class="cc-chart-empty">ALPHA karar akışı bekleniyor.</div>';return;}
  const vals=rows.map(x=>Math.max(0,Number(x.evidence_score||0))),max=Math.max(.01,...vals),w=900,h=280,pad=18;
  const pts=vals.map((v,i)=>`${pad+(w-2*pad)*(vals.length===1?0:i/(vals.length-1))},${h-pad-(h-2*pad)*(v/max)}`).join(' ');
  root.innerHTML=`<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="ALPHA karar aktivitesi">${[.25,.5,.75].map(k=>`<line class="cc-gridline" x1="${pad}" x2="${w-pad}" y1="${h*k}" y2="${h*k}"/>`).join('')}<polyline class="cc-line" points="${pts}"/></svg><div class="cc-card-sub" style="padding-top:8px">${active.length} açık yön · ${rows.filter(d=>d.action==='OPEN_LONG'||d.action==='OPEN_SHORT').length} açma kararı / son ${rows.length}</div>`;
}
function renderAlphaOverview(data){
  setOverviewLabels(true);const a=data.alpha_v2||{},ds=a.decisions||[],ps=(a.positions||[]).filter(p=>Number(p.position)!==0),os=a.outcomes||[];
  const opens=ds.filter(d=>d.action==='OPEN_LONG'||d.action==='OPEN_SHORT'),longs=ps.filter(p=>Number(p.position)>0).length,shorts=ps.filter(p=>Number(p.position)<0).length;
  const resolved=os.filter(o=>Number.isFinite(Number(o.direction_adjusted_return))),positive=resolved.filter(o=>Number(o.direction_adjusted_return)>0).length,hit=resolved.length?100*positive/resolved.length:0;
  const costs=ds.map(d=>Number(d.estimated_round_trip_cost_bps)).filter(Number.isFinite),avgCost=costs.length?costs.reduce((s,x)=>s+x,0)/costs.length:null;
  $('kpiEquity').textContent=a.online?'ALPHA LIVE':statusTr(a.status||'STALE');$('kpiEquity').className=`cc-kpi-value ${a.online?'ok':'warn'}`;$('kpiEquityMeta').textContent=`Son karar ${age(a.decision_age_seconds)} önce · ${a.compiler_version||'compiler'}`;
  $('kpiPnl').textContent=`${longs} LONG / ${shorts} SHORT`;$('kpiPnl').className='cc-kpi-value';$('kpiPnlMeta').textContent='Direction-only shadow state · gerçek emir yok';
  $('kpiPositions').textContent=String(ps.length);$('kpiPositionsMeta').textContent=ps.slice(0,8).map(p=>sym(p.asset_id)).join(' · ')||'Açık yön yok';
  $('kpiWin').textContent=resolved.length?`${hit.toFixed(1)}%`:'—';$('kpiWinMeta').textContent=`${positive} pozitif / ${resolved.length} resolved prospective outcome`;
  $('kpiActions').textContent=String(opens.length);$('kpiActionsMeta').textContent=`OPEN_LONG/SHORT · son ${ds.length} ALPHA kararı`;
  $('kpiCost').textContent=avgCost==null?'—':`${avgCost.toFixed(1)} bps`;$('kpiCostMeta').textContent='Son kararların ortalama round-trip cost tahmini';
  renderAlphaActivity(ds,ps);
  $('positionList').innerHTML=ps.slice(0,16).map(p=>{const side=Number(p.position)>0?'LONG':'SHORT',entry=Number(p.entry_price),last=Number(p.last_reference_price),move=entry>0&&last>0?((last/entry-1)*10000*(Number(p.position)>0?1:-1)):null;return`<div class="cc-row"><div class="cc-row-main"><div class="cc-row-title">${esc(sym(p.asset_id))} · ${side}</div><div class="cc-row-meta">Giriş ${num(p.entry_price,8)} · son ${num(p.last_reference_price,8)} · ${clock(p.last_action_at)}</div></div><div class="cc-row-side ${move==null?'':move>=0?'ok':'bad'}">${move==null?'—':bps(move)}</div></div>`}).join('')||'<div class="cc-row">ALPHA açık yön bekliyor.</div>';
  $('tradeList').innerHTML=opens.slice(0,18).map(d=>`<div class="cc-row"><div class="cc-row-main"><div class="cc-row-title">${esc(sym(d.asset_id))} · <span class="cc-action ${actionClass(d.action)}">${esc(actionTr(d.action))}</span></div><div class="cc-row-meta">${clock(d.observed_at)} · fiyat ${num(d.observed_reference_price,8)} · skor ${num(d.evidence_score,4)}</div></div><div class="cc-row-side">${d.estimated_round_trip_cost_bps==null?'—':num(d.estimated_round_trip_cost_bps,1)+' bps'}</div></div>`).join('')||'<div class="cc-row">Son ALPHA penceresinde OPEN aksiyonu yok.</div>';
  renderReports(data.reports||[]);
}
'''
replace_once(js, insert_marker, alpha_helper + "\nfunction renderOverview(data){if(currentPolicy()==='ALPHA')return renderAlphaOverview(data);setOverviewLabels(false);const s=data.session;const snap=s?data.policies?.[currentPolicy()]||null:null;")

# 3) PWA cache bump so installed/mobile clients do not stay on the stale JS shell.
sw = ROOT / "monster-coins-pro" / "sw.js"
replace_once(sw, "const CACHE='monster-coins-pro-shell-v7';", "const CACHE='monster-coins-pro-shell-v8';")

# 4) Keep existing regression tests aligned and add explicit live behavior checks.
dash_test = ROOT / "tests" / "test_brian2026_dashboard_mobile_tr.py"
replace_once(dash_test, 'assert "monster-coins-pro-shell-v7" in sw', 'assert "monster-coins-pro-shell-v8" in sw')

dip_test = ROOT / "tests" / "test_brian2026_dip_runtime_stability.py"
replace_once(dip_test, 'assert "monster-coins-pro-shell-v7" in SW', 'assert "monster-coins-pro-shell-v8" in SW')

live_test = ROOT / "tests" / "test_brian2026_live_ui_execution.py"
live_test.write_text(r'''from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DASH = (ROOT / "monster-coins-pro" / "dashboard.js").read_text(encoding="utf-8")
INDEX = (ROOT / "monster-coins-pro" / "index.html").read_text(encoding="utf-8")
GUARD = (ROOT / "monster-coins-pro" / "dip-expert-v4-runtime-guard.js").read_text(encoding="utf-8")
SW = (ROOT / "monster-coins-pro" / "sw.js").read_text(encoding="utf-8")


def test_general_overview_defaults_to_live_alpha_instead_of_zero_frozen_tracker():
    assert '<option value="ALPHA" selected>ALPHA Canlı</option>' in INDEX
    assert "currentPolicy(){const v=$('policyView')?.value||'ALPHA'" in DASH
    assert "renderAlphaOverview" in DASH
    assert "direction-only/no notional" in DASH
    assert "OPEN_LONG/SHORT" in DASH


def test_dip_universe_is_sticky_and_self_recovers_from_partial_refresh():
    assert "BRIAN_DIP_LIVE_V6" in GUARD
    assert "v6LastHealthyUniverse" in GUARD
    assert "v6RecoverUniverse" in GUARD
    assert "if(before.length){v4Universe=[...before]" in GUARD
    assert "setInterval" in GUARD


def test_v5_reasoner_can_bridge_quality_long_and_short_candidates_without_bypassing_safety():
    assert "PULLBACK_CONTINUATION" in GUARD
    assert "BREAKOUT_RETEST_CONTINUATION" in GUARD
    assert "TREND_EXHAUSTION" in GUARD
    assert "DOWNTREND_BREAK" in GUARD
    assert "v4FuturesSymbols.has" in GUARD
    assert "V4_COST_EDGE_MULT" in GUARD
    assert "vetoReasons" in GUARD
    assert "v4Open(st,ctx,'LONG'" in GUARD
    assert "v4Open(st,pctx,'SHORT'" in GUARD
    assert "SHADOW/PAPER ONLY" in GUARD
    assert "live order endpoint" in GUARD


def test_pwa_shell_is_bumped_for_live_fix():
    assert "monster-coins-pro-shell-v8" in SW


def test_modified_javascript_parses_when_node_is_available():
    node = shutil.which("node")
    if node is None:
        return
    for rel in ["monster-coins-pro/dashboard.js", "monster-coins-pro/dip-expert-v4-runtime-guard.js"]:
        result = subprocess.run([node, "--check", str(ROOT / rel)], capture_output=True, text=True)
        assert result.returncode == 0, f"{rel}: {result.stderr}"
''', encoding="utf-8")

print("Brian live UI/execution patch applied")
