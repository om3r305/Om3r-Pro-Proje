/* Brian Dip Intelligence V5 — isolated copy-on-write expert layer.
   SHADOW/PAPER ONLY. This file never calls the main Brian Control Center, never writes
   Phase 3.7 state/checkpoints, and never shares the main cashbox. It consumes only the
   Dip page's existing V4 market context and persists only inside brian-dip-trader snapshots/events. */

const DIP_BRAIN_V5 = 'dip-isolated-brian-intelligence-v5';
const DIP_BRAIN_CASHBOX = 'DIP_SHADOW_CASHBOX_V5';
const DIP_BRAIN_MAIN_RUNTIME_MUTATION = false;
const DIP_BRAIN_MIN_CONF = 0.52;
const DIP_BRAIN_MIN_AGREEMENT = 0.54;
const DIP_BRAIN_MIN_EDGE = 0.18;
const DIP_BRAIN_MISS_HORIZON_MS = 5 * 60 * 1000;
const DIP_BRAIN_MAX_PENDING = 36;
const DIP_BRAIN_MAX_AUDITS = 60;
const DIP_BRAIN_BASE_WEIGHTS = Object.freeze({
  structure: 0.32,
  trend: 0.24,
  momentum: 0.18,
  volume: 0.12,
  mean_reversion: 0.14,
  microstructure: 0.20,
});

let v5Brain = v5FreshBrain();
const v5DecisionBySymbol = {};
const v5LastMissCandidateAt = {};
const v5LastLearnVectorAt = {};

function v5FreshBrain(){
  return {
    schemaVersion:'brian.dip-intelligence-state.v1',
    intelligenceVersion:DIP_BRAIN_V5,
    cashboxId:DIP_BRAIN_CASHBOX,
    mainRuntimeMutation:false,
    createdAt:Date.now(),
    lastLearningAt:Date.now(),
    experts:{},
    setups:{},
    vetoes:{},
    familiarity:{},
    pending:[],
    audits:[],
    outcomes:[],
    ab:{
      champion:'V4_RULE_EXECUTOR',
      challenger:'BRIAN_EXPERT_CLONE_V5',
      policy:'HYBRID_GATED',
      attempted:0,accepted:0,brainVetoed:0,brainOnlyCandidates:0,
      savedLosses:0,missedWins:0,correctAbstentions:0,
      promotedInsideDip:false
    }
  };
}
function v5Num(x,d=0){const n=Number(x);return Number.isFinite(n)?n:d;}
function v5Clip(x,a,b){return Math.max(a,Math.min(b,v5Num(x)));}
function v5Sign(x){return x>0?1:x<0?-1:0;}
function v5BetaReliability(row){
  const n=v5Num(row?.n),wins=v5Num(row?.wins),wr=(wins+3)/(n+6);
  return v5Clip(0.72 + (wr-.5)*1.25,.65,1.35);
}
function v5Freshness(){
  const age=Math.max(0,Date.now()-v5Num(v5Brain.lastLearningAt,v5Brain.createdAt));
  if(age<=6*3600000)return 1;
  if(age<=24*3600000)return .90;
  if(age<=72*3600000)return .78;
  return .68;
}
function v5ExpertRow(name){return v5Brain.experts[name]||(v5Brain.experts[name]={n:0,wins:0,pnl:0});}
function v5SetupRow(name){return v5Brain.setups[name]||(v5Brain.setups[name]={n:0,wins:0,pnl:0,mfe:0,mae:0});}
function v5VetoRow(name){return v5Brain.vetoes[name]||(v5Brain.vetoes[name]={n:0,savedLoss:0,missedWin:0,neutral:0});}

function v5FeatureVector(ctx){
  const pressure=v5Clip((v5Num(ctx?.bk?.pressure,1)-1)/.55,-1,1);
  return [
    v5Clip(ctx?.f1?.trend,-1,1),v5Clip(ctx?.f5?.trend,-1,1),
    v5Clip(ctx?.f15?.trend,-1,1),v5Clip(ctx?.f1h?.trend,-1,1),
    v5Clip((v5Num(ctx?.f1?.rsi,50)-50)/50,-1,1),v5Clip(v5Num(ctx?.f1?.z)/3,-1,1),
    v5Clip((v5Num(ctx?.f1?.volRel,1)-1)/2,-1,1),v5Clip(ctx?.flow?.ofi,-1,1),
    pressure,v5Clip(ctx?.btcLongRisk,-1,1),v5Clip(v5Num(ctx?.idioZ)/3,-1,1),
    v5Clip((v5Num(ctx?.edgeRatio)-2.5)/4,-1,1)
  ];
}
function v5FamiliarityState(sym){
  return v5Brain.familiarity[sym]||(v5Brain.familiarity[sym]={n:0,mean:[],m2:[],recent:[],lastAt:0});
}
function v5AssessFamiliarity(sym,vector){
  const s=v5FamiliarityState(sym);
  if(s.n<20||!Array.isArray(s.mean)||s.mean.length!==vector.length){return{familiarity:1,ood:false,drift:false,hardDrift:false,score:0,driftScore:0};}
  const z=vector.map((x,i)=>{const variance=s.n>1?v5Num(s.m2[i])/(s.n-1):1,sd=Math.sqrt(Math.max(1e-4,variance));return v5Clip((x-v5Num(s.mean[i]))/sd,-8,8);});
  const score=Math.sqrt(v4Mean(z.map(x=>x*x)));
  const recent=(s.recent||[]).slice(-16);
  let driftScore=0;
  if(recent.length>=8){
    const means=vector.map((_,i)=>v4Mean(recent.map(r=>v5Num(r[i]))));
    const dz=means.map((x,i)=>{const variance=s.n>1?v5Num(s.m2[i])/(s.n-1):1,sd=Math.sqrt(Math.max(1e-4,variance));return v5Clip((x-v5Num(s.mean[i]))/sd,-8,8);});
    driftScore=Math.sqrt(v4Mean(dz.map(x=>x*x)));
  }
  const hard=score>4.8||driftScore>3.8,drift=hard||driftScore>2.35,ood=score>3.75;
  return{familiarity:v5Clip(Math.exp(-Math.max(0,score-1.25)*.28),.35,1),ood,drift,hardDrift:hard,score,driftScore};
}
function v5LearnFeatureReference(sym,vector){
  const now=Date.now();if(now-v5Num(v5LastLearnVectorAt[sym])<5000)return;
  v5LastLearnVectorAt[sym]=now;const s=v5FamiliarityState(sym);s.n++;
  if(!s.mean.length){s.mean=[...vector];s.m2=vector.map(()=>0);}else{
    vector.forEach((x,i)=>{const d=x-s.mean[i];s.mean[i]+=d/s.n;const d2=x-s.mean[i];s.m2[i]+=d*d2;});
  }
  s.recent=(s.recent||[]);s.recent.push([...vector]);while(s.recent.length>24)s.recent.shift();s.lastAt=now;
}

function v5Tape(sym,ms=15000){
  if(typeof v4TapePulse==='function')return v4TapePulse(sym,ms);
  const a=(v4TapeSpot[sym]||[]),cut=Date.now()-ms,x=a.filter(q=>q.t>=cut&&Number(q.p)>0);
  if(x.length<2)return{retPct:0,ofi:0,ratio:1,prints:x.length};
  const buy=v4Sum(x.map(q=>q.buy)),sell=v4Sum(x.map(q=>q.sell)),tot=buy+sell;
  return{retPct:v4Pct(Number(x.at(-1).p),Number(x[0].p)),ofi:tot?(buy-sell)/tot:0,ratio:sell?buy/sell:(buy?9:1),prints:x.length};
}
function v5SweepState(sym,p){
  const rows=(typeof v4RecentOneMin==='function'?v4RecentOneMin(sym,12):v4Closed(sym,'1m').slice(-12));
  if(rows.length<7)return{bull:false,bear:false};
  const prior=rows.slice(0,-2),tail=rows.slice(-2),priorLow=Math.min(...prior.map(x=>Number(x.l))),priorHigh=Math.max(...prior.map(x=>Number(x.h)));
  const tailLow=Math.min(...tail.map(x=>Number(x.l))),tailHigh=Math.max(...tail.map(x=>Number(x.h)));
  return{bull:tailLow<priorLow&&p>priorLow,bear:tailHigh>priorHigh&&p<priorHigh,priorLow,priorHigh};
}
function v5Regime(ctx){
  const h=.58*v5Num(ctx?.f15?.trend)+.42*v5Num(ctx?.f1h?.trend),btc=v5Num(ctx?.btcLongRisk),vol=v5Num(ctx?.f5?.atrPct);
  if(btc<-.55&&h<-.12)return'STRESS_DOWN';
  if(h>.34)return'TREND_UP';if(h<-.34)return'TREND_DOWN';
  if(vol>1.6)return'VOLATILE_RANGE';return'RANGE';
}
function v5RegimeWeight(name,regime){
  const m={structure:1,trend:1,momentum:1,volume:1,mean_reversion:1,microstructure:1};
  if(regime==='TREND_UP'||regime==='TREND_DOWN'){m.trend=1.35;m.momentum=1.20;m.structure=1.15;m.mean_reversion=.72;}
  else if(regime==='RANGE'){m.mean_reversion=1.38;m.structure=1.18;m.trend=.82;}
  else if(regime==='VOLATILE_RANGE'){m.microstructure=1.30;m.structure=1.15;m.mean_reversion=1.12;m.momentum=.90;}
  else if(regime==='STRESS_DOWN'){m.microstructure=1.32;m.structure=1.25;m.trend=1.18;m.mean_reversion=.62;}
  return m[name]||1;
}
function v5Vote(name,bias,confidence,reason,regime){
  const rel=v5BetaReliability(v5ExpertRow(name));
  const weight=v5Num(DIP_BRAIN_BASE_WEIGHTS[name],.15)*v5RegimeWeight(name,regime)*rel;
  return{name,bias:v5Clip(bias,-1,1),confidence:v5Clip(confidence,0,1),weight,reason};
}
function v5ExpertVotes(sym,ctx,p,regime){
  const pulse=v5Tape(sym,15000),sweep=v5SweepState(sym,p),pressure=v5Clip((v5Num(ctx.bk.pressure,1)-1)/.50,-1,1);
  let structure=.46*v5Num(ctx.f5.trend)+.30*v5Num(ctx.f15.trend)+.24*v5Num(ctx.f1h.trend);
  if(sweep.bull)structure+=.28;if(sweep.bear)structure-=.28;
  const trend=.25*v5Num(ctx.f5.trend)+.30*v5Num(ctx.f15.trend)+.45*v5Num(ctx.f1h.trend);
  const rsi=v5Clip((v5Num(ctx.f1.rsi,50)-50)/50,-1,1);
  const pulseBias=v5Clip(pulse.retPct/Math.max(.08,v5Num(ctx.f1.atrPct)*.45),-1,1);
  const momentum=.34*v5Num(ctx.f1.trend)+.28*rsi+.38*pulseBias;
  const volume=v5Clip(.50*v5Clip((v5Num(ctx.f1.volRel,1)-1)/1.5,-1,1)*v5Sign(pulse.retPct||ctx.flow.ofi)+.50*v5Num(ctx.flow.ofi),-1,1);
  let meanRev=v5Clip(-v5Num(ctx.f1.z)*.24,-.55,.55);if(Math.abs(trend)>.35)meanRev*=.62;
  const micro=.62*v5Num(ctx.flow.ofi)+.38*pressure;
  return[
    v5Vote('structure',structure,.58+.30*Math.abs(structure),sweep.bull?'bull sweep/reclaim':sweep.bear?'bear sweep/reject':'5m/15m/1h structure',regime),
    v5Vote('trend',trend,.55+.34*Math.abs(trend),'completed 5m/15m/1h trend',regime),
    v5Vote('momentum',momentum,.50+.34*Math.abs(momentum),'1m RSI + 15s tape acceleration',regime),
    v5Vote('volume',volume,.48+.35*Math.abs(volume),'relative volume + taker flow',regime),
    v5Vote('mean_reversion',meanRev,.48+.32*Math.abs(meanRev),'z-score / range location',regime),
    v5Vote('microstructure',micro,.54+.36*Math.abs(micro),'OFI + depth pressure',regime)
  ];
}
function v5Setup(sym,ctx,p,dir){
  const pulse=v5Tape(sym,15000),sweep=v5SweepState(sym,p),htf=v5Num(ctx.htfLong);
  if(dir>0&&sweep.bull&&ctx.flow.ofi>.05&&ctx.bk.pressure>1.01)return'LIQUIDITY_SWEEP_REVERSAL';
  if(dir<0&&sweep.bear&&ctx.flow.ofi<-.05&&ctx.bk.pressure<.99)return'FAILED_BREAK_REVERSAL';
  if(dir>0&&pulse.retPct>=Math.max(.04,v5Num(ctx.f1.atrPct)*.18)&&ctx.f1.trend>.35)return'BREAKOUT_RETEST_CONTINUATION';
  if(dir>0&&htf>.08&&ctx.f5.trend>.24&&p>=ctx.f1.ema9*.998)return'PULLBACK_CONTINUATION';
  if(Math.abs(htf)<.22&&dir>0&&ctx.f1.z<-1.0)return'RANGE_REJECTION';
  if(dir<0&&ctx.htfShort>.20)return'TREND_EXHAUSTION';
  return dir>0?'DIP_RECLAIM':dir<0?'DOWNTREND_BREAK':'NO_CLEAR_SETUP';
}
function v5HorizonEdges(ctx,pulse,dir){
  const align=x=>Math.max(0,x*dir);
  const h15=Math.max(0,Math.abs(v5Num(pulse.retPct))*100 + 10*align(v5Num(pulse.ofi)));
  const h1=Math.max(0,v5Num(ctx.f1.atrPct)*100*(.55+.45*align(v5Num(ctx.f1.trend))));
  const h5=Math.max(0,v5Num(ctx.expectedMoveBps)*(.55+.45*align(v5Num(ctx.f5.trend))));
  const h15m=Math.max(0,v5Num(ctx.f15.atrPct)*100*(.48+.52*align(v5Num(ctx.f15.trend))));
  return{'15s':h15,'1m':h1,'5m':h5,'15m':h15m};
}
function v5ControllerDecision(sym,ctx,p){
  const vector=v5FeatureVector(ctx),fam=v5AssessFamiliarity(sym,vector),regime=v5Regime(ctx),votes=v5ExpertVotes(sym,ctx,p,regime);
  const total=v4Sum(votes.map(v=>v.weight*Math.max(.08,v.confidence))),raw=total?v4Sum(votes.map(v=>v.bias*v.weight*Math.max(.08,v.confidence)))/total:0,dir=v5Sign(raw);
  const same=dir? v4Sum(votes.filter(v=>v5Sign(v.bias)===dir).map(v=>v.weight*Math.max(.08,v.confidence))):0;
  const agreement=total?same/total:0,uncertainty=v5Clip(1-agreement + (fam.drift ? .18 : 0) + (fam.ood ? .15 : 0),0,1);
  const freshness=v5Freshness(),confBase=v5Clip(.45+.36*Math.abs(raw)+.28*Math.max(0,agreement-.45),0,.99);
  let confidence=v5Clip(confBase*freshness*fam.familiarity*(fam.drift ? .78 : 1),0,.99);
  const pulse=v5Tape(sym,15000),horizons=v5HorizonEdges(ctx,pulse,dir||1),gross=Math.max(...Object.values(horizons)),cost=v5Num(ctx.roundTripCostBps),net=gross-cost;
  const veto=[];
  if(fam.hardDrift)veto.push('HARD_DRIFT');else if(fam.ood)veto.push('OUT_OF_DISTRIBUTION');
  if(v5Num(ctx.bk.ageMs)>3500)veto.push('STALE_BOOK');if(v5Num(ctx.bk.spreadBps)>V4_MAX_ENTRY_SPREAD_BPS)veto.push('SPREAD_TOO_WIDE');
  if(v5Num(ctx.edgeRatio)<V4_COST_EDGE_MULT)veto.push('EDGE_BELOW_COST');
  if(dir>0&&v5Num(ctx.btcLongRisk)<-.48)veto.push('BTC_STRESS_LONG');
  if(dir<0&&!v4FuturesSymbols.has(sym))veto.push('NO_PERP_MARKET');
  if(agreement<DIP_BRAIN_MIN_AGREEMENT)veto.push('EXPERT_DISAGREEMENT');
  if(confidence<DIP_BRAIN_MIN_CONF)veto.push('LOW_CONFIDENCE');
  if(Math.abs(raw)<DIP_BRAIN_MIN_EDGE)veto.push('EDGE_TOO_SMALL');
  if(net<=Math.max(2,cost*.35))veto.push('NET_EDGE_TOO_SMALL');
  let action='WAIT';if(!veto.length&&dir)action=dir>0?'BUY':'SELL';
  const setup=v5Setup(sym,ctx,p,dir),setupRel=v5BetaReliability(v5SetupRow(setup));
  confidence=v5Clip(confidence*(.86+.14*setupRel),0,.99);
  const d={
    schemaVersion:'brian.dip-decision.v1',intelligenceVersion:DIP_BRAIN_V5,cashboxId:DIP_BRAIN_CASHBOX,
    ts:Date.now(),symbol:sym,action,rawDirection:dir,edge:v5Clip(raw,-1,1),confidence,agreement,uncertainty,
    setup,regime,grossEdgeBps:gross,costBps:cost,netEdgeBps:net,horizons,freshness,
    familiarity:fam.familiarity,ood:fam.ood,drift:fam.drift,hardDrift:fam.hardDrift,driftScore:fam.driftScore,
    votes:votes.map(v=>({name:v.name,bias:v.bias,confidence:v.confidence,weight:v.weight,reason:v.reason})),vetoReasons:veto,
    shadowOnly:true,liveExecution:false,mainRuntimeMutation:false
  };
  v5LearnFeatureReference(sym,vector);return d;
}
function v5SlimDecision(d){if(!d)return null;return{ts:d.ts,action:d.action,edge:d.edge,confidence:d.confidence,agreement:d.agreement,uncertainty:d.uncertainty,setup:d.setup,regime:d.regime,grossEdgeBps:d.grossEdgeBps,costBps:d.costBps,netEdgeBps:d.netEdgeBps,freshness:d.freshness,familiarity:d.familiarity,ood:d.ood,drift:d.drift,hardDrift:d.hardDrift,vetoReasons:d.vetoReasons,votes:d.votes?.map(v=>({name:v.name,bias:v.bias,confidence:v.confidence,weight:v.weight}))||[]};}

function v5QueueCounterfactual(sym,side,p,decision,reason,kind='NATIVE_VETO'){
  const now=Date.now();if(now-v5Num(v5LastMissCandidateAt[sym])<60000)return;v5LastMissCandidateAt[sym]=now;
  const x={id:`v5-miss-${crypto.randomUUID()}`,sym,side,entry:Number(p),createdAt:now,dueAt:now+DIP_BRAIN_MISS_HORIZON_MS,bestPx:Number(p),worstPx:Number(p),reason:String(reason||'WAIT'),kind,decision:v5SlimDecision(decision)};
  v5Brain.pending.push(x);while(v5Brain.pending.length>DIP_BRAIN_MAX_PENDING)v5Brain.pending.shift();
  if(kind==='NATIVE_VETO')v5Brain.ab.brainVetoed++;else v5Brain.ab.brainOnlyCandidates++;
}
function v5ResolveCounterfactuals(sym,p){
  const now=Date.now(),px=Number(p);if(!(px>0))return;
  for(const x of v5Brain.pending){if(x.sym!==sym||x.resolved)continue;x.bestPx=Math.max(v5Num(x.bestPx,px),px);x.worstPx=Math.min(v5Num(x.worstPx,px),px);if(now<x.dueAt)continue;
    const long=x.side==='LONG',mfe=long?v4Pct(x.bestPx,x.entry):v4Pct(x.entry,x.worstPx),mae=long?Math.max(0,-v4Pct(x.worstPx,x.entry)):Math.max(0,-v4Pct(x.entry,x.bestPx));
    const closeMove=long?v4Pct(px,x.entry):v4Pct(x.entry,px),costPct=v5Num(x.decision?.costBps)/100,missed=mfe>Math.max(costPct*1.35,.18),saved=mae>Math.max(costPct*1.5,.25)&&!missed;
    const result=missed?'MISSED_PROFIT':saved?'SAVED_LOSS':'CORRECT_ABSTENTION',vr=v5VetoRow(x.reason);vr.n++;if(missed){vr.missedWin++;v5Brain.ab.missedWins++;}else if(saved){vr.savedLoss++;v5Brain.ab.savedLosses++;}else{vr.neutral++;v5Brain.ab.correctAbstentions++;}
    x.resolved=true;v5Brain.lastLearningAt=now;const audit={id:x.id,sym,side:x.side,reason:x.reason,kind:x.kind,result,mfePct:mfe,maePct:mae,closeMovePct:closeMove,costPct,at:now,decision:x.decision};v5Brain.audits.push(audit);while(v5Brain.audits.length>DIP_BRAIN_MAX_AUDITS)v5Brain.audits.shift();
    try{event('INFO',sym,px,{metadata:{expert_v4:true,brain_v5:true,info:'MISSED_OPPORTUNITY_AUDIT_V5',cashbox_id:DIP_BRAIN_CASHBOX,result,side:x.side,reason:x.reason,mfe_pct:mfe,mae_pct:mae,close_move_pct:closeMove,cost_pct:costPct,setup:x.decision?.setup||null,confidence:x.decision?.confidence||null}})}catch{}
  }
  v5Brain.pending=v5Brain.pending.filter(x=>!x.resolved||now-x.dueAt<60000);
}
function v5UpdatePositionExcursions(st,p){
  const q=st?.pos;if(!q||!q.brainV5)return;const px=Number(p);if(!(px>0))return;const long=(q.side||'LONG')==='LONG';
  const fav=long?v4Pct(px,q.entry):v4Pct(q.entry,px),adv=long?Math.max(0,-v4Pct(px,q.entry)):Math.max(0,-v4Pct(q.entry,px));q.mfePct=Math.max(v5Num(q.mfePct),fav);q.maePct=Math.max(v5Num(q.maePct),adv);
}
function v5LearnOutcome(q,pnlValue,reason){
  if(!q?.brainV5)return;const won=Number(pnlValue)>0,d=q.brainV5,setup=v5SetupRow(d.setup||q.mode||'UNKNOWN');setup.n++;if(won)setup.wins++;setup.pnl+=Number(pnlValue||0);setup.mfe+=v5Num(q.mfePct);setup.mae+=v5Num(q.maePct);
  for(const vote of d.votes||[]){const row=v5ExpertRow(vote.name);row.n++;const correct=(won&&v5Sign(vote.bias)===v5Sign(d.edge))||(!won&&v5Sign(vote.bias)!==v5Sign(d.edge));if(correct)row.wins++;row.pnl+=Number(pnlValue||0)*Math.abs(v5Num(vote.bias));}
  v5Brain.outcomes.push({t:Date.now(),sym:q.symbol||null,setup:d.setup,pnl:Number(pnlValue||0),reason,mfePct:v5Num(q.mfePct),maePct:v5Num(q.maePct)});while(v5Brain.outcomes.length>50)v5Brain.outcomes.shift();v5Brain.lastLearningAt=Date.now();
  if(v5Brain.ab.attempted>=25){const net=v4Sum(v5Brain.outcomes.slice(-25).map(x=>x.pnl));v5Brain.ab.promotedInsideDip=net>0&&v5Brain.ab.savedLosses>=v5Brain.ab.missedWins;}
}

const _v5NativeOpen=v4Open;
v4Open=function(st,ctx,side,mode){
  const wanted=side==='LONG'?'BUY':'SELL',d=v5DecisionBySymbol[st.symbol]||v5ControllerDecision(st.symbol,ctx,ctx?.bk?.mid||Number(live[st.symbol]));
  v5Brain.ab.attempted++;
  const mismatch=d.action!==wanted||d.hardDrift||d.netEdgeBps<=0;
  if(mismatch){const why=d.vetoReasons?.[0]||`BRAIN_${d.action}_VS_${wanted}`;st.v4.lastVeto=`BRAIN_V5_${why}`;v5QueueCounterfactual(st.symbol,side,ctx?.bk?.mid||Number(live[st.symbol]),d,why,'NATIVE_VETO');return v4Reject(`BRAIN_V5_${why}`);}
  const ok=_v5NativeOpen(st,ctx,side,mode);if(ok&&st.pos){st.pos.brainV5=v5SlimDecision(d);st.pos.cashboxId=DIP_BRAIN_CASHBOX;st.pos.mfePct=0;st.pos.maePct=0;st.pos.symbol=st.symbol;v5Brain.ab.accepted++;}
  return ok;
};
const _v5NativeClose=v4Close;
v4Close=function(st,ctx,reason){
  const q=st?.pos?{...st.pos,brainV5:st.pos.brainV5?JSON.parse(JSON.stringify(st.pos.brainV5)):null}:null,before=v5Num(book.realized);_v5NativeClose(st,ctx,reason);const delta=v5Num(book.realized)-before;if(q)v5LearnOutcome(q,delta,reason);
};
const _v5NativeManage=v4Manage;
v4Manage=function(st,ctx){
  if(!st?.pos||!ctx)return _v5NativeManage(st,ctx);const p=ctx.bk.mid||Number(live[st.symbol]);v5UpdatePositionExcursions(st,p);const d=v5ControllerDecision(st.symbol,ctx,p);v5DecisionBySymbol[st.symbol]=d;
  const side=st.pos.side||'LONG',flip=(side==='LONG'&&d.action==='SELL'&&d.confidence>=.72&&ctx.htfLong<-.15)||(side==='SHORT'&&d.action==='BUY'&&d.confidence>=.72&&ctx.htfShort<-.15);
  if(flip)return v4Close(st,ctx,'BRAIN_V5_STRUCTURE_FLIP');return _v5NativeManage(st,ctx);
};
function v5ReasonerEntry(st,ctx,p,d){
  if(!d||d.action!=='BUY'||d.confidence<.58||d.agreement<.57)return false;
  if(!['LIQUIDITY_SWEEP_REVERSAL','RANGE_REJECTION'].includes(d.setup))return false;
  const flowOk=ctx.flow.ofi>=.07&&ctx.bk.pressure>=1.02,btcOk=ctx.btcLongRisk>-.42;if(!flowOk||!btcOk)return false;
  return v4Open(st,ctx,'LONG',`V5_${d.setup}`);
}
const _v5NativeEvaluate=v4Evaluate;
v4Evaluate=function(sym,p){
  if(!running||v4Booting||v4NeedsRestart||!sid||!v4Universe.includes(sym))return;v5ResolveCounterfactuals(sym,p);
  const st=v4Ensure(sym),ctx=v4Context(sym,st.pos?.venue||'SPOT');if(!ctx)return _v5NativeEvaluate(sym,p);
  const d=v5ControllerDecision(sym,ctx,ctx.bk.mid||Number(p));v5DecisionBySymbol[sym]=d;st.v4.brainV5=v5SlimDecision(d);
  if(st.pos)return v4Manage(st,ctx);
  if(v5ReasonerEntry(st,ctx,ctx.bk.mid||Number(p),d))return;
  const before=Boolean(st.pos),priorAttempt=v5Brain.ab.attempted;_v5NativeEvaluate(sym,p);const after=Boolean(st.pos);
  if(!before&&!after&&d.action!=='WAIT'&&d.confidence>=.56&&d.netEdgeBps>0&&v5Brain.ab.attempted===priorAttempt){v5QueueCounterfactual(sym,d.action==='BUY'?'LONG':'SHORT',ctx.bk.mid||Number(p),d,st.v4.lastVeto||'NO_NATIVE_SETUP','BRAIN_ONLY_CANDIDATE');}
};

const _v5CompactState=v4CompactState;
v4CompactState=function(){
  const base=_v5CompactState(),compact=JSON.parse(JSON.stringify(v5Brain));
  for(const s of Object.values(compact.familiarity||{})){if(Array.isArray(s.recent))s.recent=s.recent.slice(-12);}
  compact.pending=(compact.pending||[]).slice(-DIP_BRAIN_MAX_PENDING);compact.audits=(compact.audits||[]).slice(-DIP_BRAIN_MAX_AUDITS);compact.outcomes=(compact.outcomes||[]).slice(-50);
  return{...base,isolation:{cashbox_id:DIP_BRAIN_CASHBOX,intelligence_version:DIP_BRAIN_V5,main_runtime_mutation:false,main_phase37_mutation:false},v5Brain:compact};
};
const _v5Restore=restore;
restore=function(d){
  _v5Restore(d);const saved=d?.snapshot?.state?.v5Brain;if(saved?.cashboxId===DIP_BRAIN_CASHBOX&&saved?.mainRuntimeMutation===false){v5Brain={...v5FreshBrain(),...saved,experts:saved.experts||{},setups:saved.setups||{},vetoes:saved.vetoes||{},familiarity:saved.familiarity||{},pending:saved.pending||[],audits:saved.audits||[],outcomes:saved.outcomes||[],ab:{...v5FreshBrain().ab,...(saved.ab||{})}};}else if(!d?.session){v5Brain=v5FreshBrain();}
};
const _v5Start=start;
start=async function(restart=false){if(restart||!session)v5Brain=v5FreshBrain();return _v5Start(restart);};

const _v5Note=note;
note=function(e){const m=e?.metadata||{};if(m.brain_v5&&m.info==='MISSED_OPPORTUNITY_AUDIT_V5')return`V5 ${m.result} · ${m.side} · MFE ${v5Num(m.mfe_pct).toFixed(2)}% / MAE ${v5Num(m.mae_pct).toFixed(2)}% · ${m.reason}`;return _v5Note(e);};
const _v5RadarRow=v4RadarRow;
v4RadarRow=function(sym){
  const html=_v5RadarRow(sym),d=states[sym]?.v4?.brainV5;if(!d)return html;
  const brain=`<div class="scoreLabel">Brian V5 ${d.action} · ${v5Esc(d.setup||'WAIT')} · net ${v5Num(d.netEdgeBps).toFixed(0)}bps · C ${(v5Num(d.confidence)*100).toFixed(0)}% · A ${(v5Num(d.agreement)*100).toFixed(0)}%${d.drift?' · DRIFT':''}${d.ood?' · OOD':''}</div>`;
  return html.replace('</div><div class="coinRight">',`${brain}</div><div class="coinRight">`);
};
const _v5UiPatch=v4UiPatch;
v4UiPatch=function(){
  _v5UiPatch();const h=document.querySelector('.desktopTitle h1');if(h)h.textContent='Aggressive Dip · Brian Expert V5';const mb=document.querySelector('.mobileBrand b');if(mb)mb.textContent='Brian Dip V5';
  const sub=document.querySelector('.desktopTitle .sub');if(sub)sub.textContent='AYRI KASA · Brian expert clone · regime MoE · 15s/1m/5m/15m · OOD/drift · net-edge · missed-opportunity learning';
  const banner=document.querySelector('.dipBanner>div:first-child');if(banner)banner.innerHTML='<strong>BRIAN DIP V5 · SHADOW ONLY · AYRI KASA</strong> · Ana Brian/Phase 3.7 state, checkpoint ve kasa yazılmaz. Dip yalnız kendi snapshot/hafızasında öğrenir; expert committee + regime weighting + OOD/drift + maliyet sonrası net-edge + missed-opportunity audit birlikte çalışır.';
  const badge=document.querySelector('.expertModeCard .badge');if(badge)badge.textContent='ISOLATED BRIAN EXPERT V5';const expert=document.querySelector('.expertModeCard');if(expert){const b=expert.querySelector('b');if(b)b.textContent='Tek uzman çekirdeğin kopyası · Dip’e özel hafıza ve reliability';const s=expert.querySelector('small');if(s)s.textContent='DIP_SHADOW_CASHBOX_V5 · main state write = 0 · champion/challenger shadow A/B';}
  if($('startBtn'))$('startBtn').textContent='▶ Brian Dip V5 Başlat';if($('restartBtn'))$('restartBtn').textContent='↻ V5 Temiz Ayrı Kasa ile Restart';
  const rule=document.querySelector('.dipRuleLine');if(rule)rule.innerHTML='V5: <b>Controller</b> = Brian expert clone (structure/trend/momentum/volume/mean-reversion/microstructure + regime + familiarity/drift) → <b>Executor</b> = mevcut V4.1 shadow lifecycle. BUY/SELL kadar WAIT ve veto kararları da 5 dk counterfactual MFE/MAE ile öğrenilir. Net edge ücret+spread+impact sonrasında pozitif değilse işlem yok.';
  const logic=document.querySelector('.logicSteps');if(logic)logic.innerHTML='<div><b>1</b><span>Brian uzman komitesi Dip içinde kopya olarak çalışır; ana dashboard hafızasına/checkpointine hiçbir write yok.</span></div><div><b>2</b><span>Rejim-conditioned mixture-of-experts: trend, range ve stress halinde uzman ağırlıkları değişir; Dip sonucu yalnız Dip reliability’sini günceller.</span></div><div><b>3</b><span>15sn tape + 1m/5m/15m/1h yapı + BTC rejimi + OFI/depth tek decision contract içinde birleşir.</span></div><div><b>4</b><span>Model familiarity/OOD + rolling drift confidence’i düşürür; hard drift yeni girişi veto eder.</span></div><div><b>5</b><span>Gross hareket değil, fee+spread+impact sonrası net edge aranır; executor stop/target/trail/flow/time lifecycle’ını ayrı yönetir.</span></div><div><b>6</b><span>Trade yapılmayan güçlü adaylar da kaydedilir; 5 dk MFE/MAE ile MISSED_PROFIT / SAVED_LOSS / CORRECT_ABSTENTION ayrımı öğrenilir.</span></div>';
  if($('radarStatus'))$('radarStatus').textContent=running?'BRIAN V5 LIVE':'V5 WAIT';
};
const _v5RenderKpi=renderKpi;
renderKpi=function(){_v5RenderKpi();if($('kpiEngine')){$('kpiEngine').textContent=running?'BRIAN V5':'V5 IDLE';$('kpiEngine').className=`value ${running?'pos':'amber'}`;}if($('kpiEngineMeta'))$('kpiEngineMeta').textContent=`ISOLATED · C/W ${v5Brain.ab.accepted}/${v5Brain.ab.brainVetoed} · miss ${v5Brain.ab.missedWins} · saved ${v5Brain.ab.savedLosses}`;};

// Runtime assertion: this layer is intentionally unable to target the main control-center API.
(function v5IsolationAssertion(){
  if(DIP_BRAIN_MAIN_RUNTIME_MUTATION!==false)throw Error('DIP_V5_ISOLATION_BROKEN');
  if(typeof API==='string'&&!API.includes('/brian-dip-trader'))throw Error('DIP_V5_WRONG_BACKEND');
})();

addEventListener('load',()=>{try{v4UiPatch();render();event('INFO',null,null,{metadata:{expert_v4:true,brain_v5:true,info:'BRIAN_DIP_V5_READY',cashbox_id:DIP_BRAIN_CASHBOX,intelligence_version:DIP_BRAIN_V5,main_runtime_mutation:false,main_phase37_mutation:false}});}catch(e){console.warn('dip-v5-init',e);}});
