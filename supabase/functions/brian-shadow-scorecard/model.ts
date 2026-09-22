export type Row = Record<string, any>;
export const number = (v: unknown): number | null => v === null || v === undefined || v === '' || typeof v === 'boolean' ? null : Number.isFinite(Number(v)) ? Number(v) : null;
const time = (v: unknown) => Date.parse(String(v ?? ''));

// A closed trade's ledger P&L already contains costs. Never subtract them twice.
export function summarize(protocol: Row, points: Row[], trades: Row[], truncated: boolean, now = Date.now()) {
  const ordered = [...points].sort((a,b) => time(a.captured_at)-time(b.captured_at));
  const first = ordered[0], last = ordered.at(-1);
  const issues: string[] = [];
  if (!first || !last) issues.push('NO_OBSERVATIONS');
  if (first?.open_positions > 0) issues.push('OPEN_POSITIONS_AT_START');
  if (ordered.some(p => p.session_id !== first.session_id)) issues.push('SESSION_CHANGED');
  if (!first?.policy_version || ordered.some(p => p.policy_version !== first.policy_version)) issues.push('POLICY_CHANGED_OR_UNKNOWN');
  if (ordered.some(p => p.shadow_only !== true || p.live_execution !== false)) issues.push('NON_SHADOW_RECORD');
  if (ordered.some(p => number(p.equity) === null || number(p.equity)! < 0)) issues.push('MISSING_EQUITY');
  const sourceAgeLimit=protocol.engine_id==='treasury'?600000:180000;
  if (ordered.some(p => !Number.isFinite(time(p.source_at)) || time(p.captured_at)-time(p.source_at)>sourceAgeLimit || time(p.source_at)>time(p.captured_at)+30000)) issues.push('STALE_SOURCE');
  if (ordered.some((p,i) => i>0 && time(p.captured_at)-time(ordered[i-1].captured_at)>660000)) issues.push('OBSERVATION_GAP');
  const end = Math.min(now,time(protocol.ends_at));
  if (last && end-time(last.captured_at)>660000) issues.push('COLLECTOR_DELAY');
  if (truncated) issues.push('BOUNDED_REPORT_TRUNCATED');
  const eligible = trades.filter(t => t.policy_version===first?.policy_version && t.shadow_only===true && t.live_execution===false);
  if (eligible.length!==trades.length) issues.push('TRADE_POLICY_OR_MODE_MISMATCH');
  const known = eligible.filter(t => number(t.net)!==null);
  const unknown = trades.length-known.length;
  if (unknown) issues.push('UNRESOLVED_TRADE_COST_OR_ENTRY');
  let gains=0,losses=0,wins=0,lossCount=0,flat=0;
  for(const t of known){const n=number(t.net)!;if(n>0){gains+=n;wins++}else if(n<0){losses-=n;lossCount++}else flat++}
  let peak=number(first?.equity),drawdown=peak!==null&&peak>0?0:null;
  for(const p of ordered){const eq=number(p.equity);if(eq===null||peak===null||peak<=0)continue;peak=Math.max(peak,eq);drawdown=Math.max(drawdown??0,(peak-eq)/peak)}
  const baseline=number(first?.equity),equity=number(last?.equity);
  const continuous=!issues.some(x=>['NO_OBSERVATIONS','SESSION_CHANGED','POLICY_CHANGED_OR_UNKNOWN','NON_SHADOW_RECORD','MISSING_EQUITY'].includes(x));
  const delta=continuous&&baseline!==null&&equity!==null?equity-baseline:null;
  const days=first&&last?Math.max(0,(time(last.captured_at)-time(first.captured_at))/86400000):0;
  const rules=protocol.rules;
  const enough=known.length>=rules.minimum_closed_trades&&days>=rules.minimum_days-300/86400;
  let verdict='COLLECTING';
  if(issues.length)verdict='INCOMPLETE';
  else if(drawdown!==null&&drawdown>=rules.review_drawdown_limit)verdict='RISK_REVIEW';
  else if(now>=time(protocol.ends_at)&&!enough)verdict='INSUFFICIENT_SAMPLE';
  else if(enough)verdict=delta!==null&&delta>0&&gains>losses?'REVIEW_POSITIVE':'REVIEW_NO_EDGE';
  return {engine_id:protocol.engine_id,protocol_id:protocol.protocol_id,starts_at:protocol.starts_at,ends_at:protocol.ends_at,rules,verdict,issues,
    baseline_equity:baseline,latest_equity:equity,net_equity_change:delta,return_pct:delta!==null&&baseline!==null&&baseline>0?delta/baseline:null,
    cash_baseline_return:0,closed_trade_net:unknown?null:gains-losses,known_trade_net:gains-losses,closed_trades:known.length,unknown_trades:unknown,
    wins,losses:lossCount,flat,win_rate:known.length?wins/known.length:null,profit_factor:losses>0?gains/losses:null,
    average_trade_net:known.length?(gains-losses)/known.length:null,sampled_drawdown:continuous?drawdown:null,days_observed:days,
    policy_version:first?.policy_version??null,source_at:last?.source_at??null,captured_at:last?.captured_at??null,open_positions:last?.open_positions??null,
    observations:ordered.map(p=>({at:p.captured_at,equity:number(p.equity),source_at:p.source_at})),
    recent_trades:trades.slice(-20).reverse(),automatic_promotion:false,shadow_only:true,live_execution:false};
}

export function alphaSummary(decisions: Row[], now=Date.now()) {
  const truncated=decisions.length>500, rows=decisions.slice(0,500);
  let pending=0,overdue=0,missingCost=0,netSum=0,count=0,wait=0;
  const versions=new Set(rows.map(d=>d.compiler_version));
  for(const d of rows){
    if(!['OPEN_LONG','OPEN_SHORT'].includes(d.action)){wait++;continue}
    if(number(d.direction_adjusted_return)===null){pending++;if(now-time(d.observed_at)>3900000)overdue++;continue}
    const cost=number(d.estimated_round_trip_cost_bps);
    if(cost===null||cost<0){missingCost++;continue}
    netSum+=number(d.direction_adjusted_return)!*10000-cost;count++;
  }
  return {sampled_decisions:rows.length,truncated,wait_or_veto:wait,pending,overdue,resolved:count,missing_cost:missingCost,
    mean_net_bps:count&&versions.size===1?netSum/count:null,compiler_versions:[...versions],horizon_seconds:3600,
    interpretation:'FIXED_HORIZON_COUNTERFACTUAL_NOT_PORTFOLIO_PROFIT',recent:rows.slice(-10).reverse().map(d=>({decision_id:d.decision_id,asset_id:d.asset_id,action:d.action,reason:d.reason,observed_at:d.observed_at,net_edge_bps:d.net_edge_bps}))};
}
