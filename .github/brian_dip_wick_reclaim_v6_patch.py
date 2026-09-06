from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def append_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding='utf-8')
    if marker in text:
        return
    path.write_text(text.rstrip() + '\n\n' + block.strip() + '\n', encoding='utf-8')


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding='utf-8')
    if new in text:
        return
    if old not in text:
        raise SystemExit(f'expected fragment missing in {path}: {old!r}')
    path.write_text(text.replace(old, new, 1), encoding='utf-8')


brain = ROOT / 'monster-coins-pro' / 'dip-expert-v5-brain.js'
append_once(
    brain,
    'BRIAN_DIP_WICK_RECLAIM_V6',
    r'''
/* BRIAN_DIP_WICK_RECLAIM_V6
   Intrabar chart-pattern perception for the isolated DIP brain.
   SHADOW/PAPER ONLY. This changes pattern detection, never adds a live order endpoint. */
function v5Esc(s){
  if(typeof v4Esc==='function')return v4Esc(s);
  return String(s??'').replace(/[&<>\"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[m]));
}

const v6BaseSweepState=v5SweepState;
const v6BaseSetup=v5Setup;
const v6BaseControllerDecision=v5ControllerDecision;
const v6LastPatternEvent={};

function v6LiveWickState(sym,p){
  const closed=(typeof v4Closed==='function'?v4Closed(sym,'1m'):[]).slice(-18);
  if(closed.length<7)return null;
  const raw=v4Bars?.[sym]?.['1m']||[];
  const tail=raw.at(-1);
  const currentBar=tail&&tail.closed===false?tail:null;
  const local=closed.slice(-6),structural=closed.slice(-12);
  const localLow=Math.min(...local.map(x=>Number(x.l))),localHigh=Math.max(...local.map(x=>Number(x.h)));
  const structuralLow=Math.min(...structural.map(x=>Number(x.l))),structuralHigh=Math.max(...structural.map(x=>Number(x.h)));
  const tape=(v4TapeSpot?.[sym]||[]).filter(q=>Date.now()-Number(q.t)<=30000&&Number(q.p)>0);
  const tapeLow=tape.length?Math.min(...tape.map(q=>Number(q.p))):Infinity;
  const tapeHigh=tape.length?Math.max(...tape.map(q=>Number(q.p))):-Infinity;
  const ref=Number(p||currentBar?.c||closed.at(-1)?.c||0);if(!(ref>0))return null;
  const liveLow=Math.min(ref,Number(currentBar?.l||Infinity),tapeLow);
  const liveHigh=Math.max(ref,Number(currentBar?.h||-Infinity),tapeHigh);
  const atr=Math.max(0,Number(v4Atr(closed,14)||0)),atrPct=atr/ref*100;
  const sweepNeedPct=v4Clamp(atrPct*.10,.012,.12);
  const reclaimNeedPct=v4Clamp(atrPct*.035,.006,.06);
  const body=currentBar?Math.abs(Number(currentBar.c)-Number(currentBar.o)):0;
  const lowerWick=currentBar?Math.max(0,Math.min(Number(currentBar.o),Number(currentBar.c))-Number(currentBar.l)):0;
  const upperWick=currentBar?Math.max(0,Number(currentBar.h)-Math.max(Number(currentBar.o),Number(currentBar.c))):0;
  const wickFloor=Math.max(body,atr*.04,ref*.00001);
  const lowerWickRatio=lowerWick/wickFloor,upperWickRatio=upperWick/wickFloor;

  // Recent local support/resistance is intentionally used in addition to the deeper
  // structural range. This catches the visual liquidity wick a human sees even when
  // an older 10-12 bar extreme sits farther away.
  const bullDepthPct=localLow>0?Math.max(0,(localLow-liveLow)/localLow*100):0;
  const bullReclaimPct=localLow>0?(ref-localLow)/localLow*100:0;
  const bearDepthPct=localHigh>0?Math.max(0,(liveHigh-localHigh)/localHigh*100):0;
  const bearReclaimPct=localHigh>0?(localHigh-ref)/localHigh*100:0;
  const tapeRich=tape.length>=3;
  const bullSweep=bullDepthPct>=sweepNeedPct&&(lowerWickRatio>=1.05||tapeRich);
  const bearSweep=bearDepthPct>=sweepNeedPct&&(upperWickRatio>=1.05||tapeRich);
  const bull=bullSweep&&bullReclaimPct>=reclaimNeedPct;
  const bear=bearSweep&&bearReclaimPct>=reclaimNeedPct;
  return{bull,bear,bullSweep,bearSweep,localLow,localHigh,structuralLow,structuralHigh,liveLow,liveHigh,
    bullDepthPct,bullReclaimPct,bearDepthPct,bearReclaimPct,lowerWickRatio,upperWickRatio,
    atrPct,sweepNeedPct,reclaimNeedPct,currentBarAt:Number(currentBar?.t||0),source:'INTRABAR_WICK'};
}

v5SweepState=function(sym,p){
  const base=v6BaseSweepState(sym,p)||{bull:false,bear:false};
  const liveState=v6LiveWickState(sym,p);
  if(!liveState)return base;
  return{...base,...liveState,bull:Boolean(base.bull||liveState.bull),bear:Boolean(base.bear||liveState.bear),
    bullSweep:Boolean(base.bull||liveState.bullSweep),bearSweep:Boolean(base.bear||liveState.bearSweep)};
};

v5Setup=function(sym,ctx,p,dir){
  const sweep=v5SweepState(sym,p);
  // Pattern classification happens before execution confirmation. The existing V4.1
  // executor still requires flow/book/HTF/cost/heat checks before opening a shadow trade.
  if(dir>0&&sweep?.bull)return'LIQUIDITY_SWEEP_REVERSAL';
  if(dir<0&&sweep?.bear)return'FAILED_BREAK_REVERSAL';
  return v6BaseSetup(sym,ctx,p,dir);
};

v5ControllerDecision=function(sym,ctx,p){
  const d=v6BaseControllerDecision(sym,ctx,p),sweep=v5SweepState(sym,p);
  if(!d||!sweep)return d;
  const pattern=sweep.bull?'WICK_SWEEP_RECLAIM_LONG':sweep.bear?'WICK_SWEEP_RECLAIM_SHORT':
    sweep.bullSweep?'WICK_SWEEP_WAIT_RECLAIM_LONG':sweep.bearSweep?'WICK_SWEEP_WAIT_RECLAIM_SHORT':null;
  if(!pattern)return d;
  d.marketPattern=pattern;
  d.sweep={source:sweep.source,depthPct:sweep.bullSweep?sweep.bullDepthPct:sweep.bearDepthPct,
    reclaimPct:sweep.bullSweep?sweep.bullReclaimPct:sweep.bearReclaimPct,
    wickRatio:sweep.bullSweep?sweep.lowerWickRatio:sweep.upperWickRatio,
    localLow:sweep.localLow,localHigh:sweep.localHigh,atrPct:sweep.atrPct};
  if(sweep.bull&&Number(d.rawDirection||0)>=0)d.setup='LIQUIDITY_SWEEP_REVERSAL';
  if(sweep.bear&&Number(d.rawDirection||0)<=0)d.setup='FAILED_BREAK_REVERSAL';

  const barKey=Number(sweep.currentBarAt||Math.floor(Date.now()/60000));
  const key=`${pattern}:${barKey}`;
  if(v6LastPatternEvent[sym]!==key&&typeof running!=='undefined'&&running&&typeof sid!=='undefined'&&sid){
    v6LastPatternEvent[sym]=key;
    try{event('INFO',sym,p,{metadata:{expert_v4:true,brain_v5:true,info:'WICK_RECLAIM_CANDIDATE',pattern,
      setup:d.setup,action:d.action,confidence:d.confidence,agreement:d.agreement,net_edge_bps:d.netEdgeBps,
      depth_pct:d.sweep.depthPct,reclaim_pct:d.sweep.reclaimPct,wick_ratio:d.sweep.wickRatio,atr_pct:d.sweep.atrPct}});}catch{}
  }
  return d;
};
'''
)

sw = ROOT / 'monster-coins-pro' / 'sw.js'
replace_once(sw, "const CACHE='monster-coins-pro-shell-v8';", "const CACHE='monster-coins-pro-shell-v9';")

test = ROOT / 'tests' / 'test_brian2026_dip_wick_reclaim_v6.py'
test.write_text(r'''from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAIN = (ROOT / "monster-coins-pro" / "dip-expert-v5-brain.js").read_text(encoding="utf-8")
SW = (ROOT / "monster-coins-pro" / "sw.js").read_text(encoding="utf-8")


def test_v5_runtime_escape_bug_is_closed():
    assert "function v5Esc(s)" in BRAIN
    assert "v5Esc(d.setup||'WAIT')" in BRAIN


def test_intrabar_wick_reclaim_uses_live_bar_local_structure_and_atr():
    assert "BRIAN_DIP_WICK_RECLAIM_V6" in BRAIN
    assert "tail&&tail.closed===false" in BRAIN
    assert "closed.slice(-6)" in BRAIN
    assert "v4TapeSpot?.[sym]" in BRAIN
    assert "sweepNeedPct=v4Clamp(atrPct*.10,.012,.12)" in BRAIN
    assert "reclaimNeedPct=v4Clamp(atrPct*.035,.006,.06)" in BRAIN
    assert "WICK_SWEEP_WAIT_RECLAIM_LONG" in BRAIN
    assert "WICK_SWEEP_RECLAIM_LONG" in BRAIN


def test_pattern_classification_does_not_bypass_executor_gates():
    assert "return v6BaseSetup(sym,ctx,p,dir)" in BRAIN
    assert "LIQUIDITY_SWEEP_REVERSAL" in BRAIN
    assert "FAILED_BREAK_REVERSAL" in BRAIN
    # No real execution endpoint may be introduced in the isolated browser brain.
    assert "/api/v3/order" not in BRAIN
    assert "live_execution=true" not in BRAIN


def test_wick_candidates_are_append_only_telemetry_and_cache_refreshes():
    assert "WICK_RECLAIM_CANDIDATE" in BRAIN
    assert "monster-coins-pro-shell-v9" in SW
''', encoding='utf-8')
