import { initialTreasuryState, opportunityQuality, planTreasuryCycle, treasuryEquityUsd, type TreasuryOpportunity, type TreasuryPosition } from "./evolution_treasury.ts";

function opp(overrides:Partial<TreasuryOpportunity>={}):TreasuryOpportunity{return{
  assetId:"BTCUSDT",direction:1,observedAt:"2026-09-11T12:59:00Z",referencePrice:100,expectedNetEdgeBps:25,roundTripCostBps:20,
  reliabilityConfidence:.6,matureGroupCount:3,pitClear:true,recommendation:"ALLOW_EDGE",sourceDecisionId:"d-1",...overrides,
};}
function pos(overrides:Partial<TreasuryPosition>={}):TreasuryPosition{return{
  positionId:"p-1",assetId:"BTCUSDT",direction:1,openedAt:"2026-09-11T12:50:00Z",entryPrice:100,capitalUsd:1000,
  entryExpectedNetEdgeBps:20,latestExpectedNetEdgeBps:20,roundTripCostBps:20,sourceDecisionId:"d-old",highWaterPnlBps:0,...overrides,
};}

Deno.test("treasury starts as one 10000 USD cash pool",()=>{const s=initialTreasuryState("2026-09-11T13:00:00Z");if(s.cashUsd!==10000||s.positions.length||treasuryEquityUsd(s,{})!==10000)throw new Error(JSON.stringify(s));});

Deno.test("treasury sizes an ordinary validated edge from measured conviction instead of a fixed ticket",()=>{
  const s=initialTreasuryState("2026-09-11T12:59:00Z");const p=planTreasuryCycle({state:s,opportunities:[opp()],observedAt:"2026-09-11T13:00:00Z",positionIdFor:o=>`p-${o.assetId}`});
  if(p.actions.length!==1||p.actions[0].kind!=="OPEN")throw new Error(JSON.stringify(p.actions));
  const capital=p.actions[0].capitalUsd;
  if(!(capital>100&&capital<5000))throw new Error(`ordinary conviction should size dynamically, got ${capital}`);
  if(capital===3||capital===5||capital===10||capital===20)throw new Error(`fixed legacy ticket leaked into Treasury sizing: ${capital}`);
  if(p.state.positions[0]?.openedAt!=="2026-09-11T13:00:00Z")throw new Error(`open timestamp leaked from signal time: ${p.state.positions[0]?.openedAt}`);
  if(!(p.afterEquityUsd<10000)||p.state.cashUsd<0)throw new Error("entry cost must reduce equity without making cash negative");
});

Deno.test("exceptional validated conviction can use effectively the full 10000 USD shadow cashbox",()=>{
  const s=initialTreasuryState("2026-09-11T12:59:00Z");
  const exceptional=opp({assetId:"EVENTUSDT",expectedNetEdgeBps:80,reliabilityConfidence:.65,matureGroupCount:8,roundTripCostBps:20,sourceDecisionId:"event-1"});
  if(opportunityQuality(exceptional)<.999)throw new Error(`exceptional evidence did not reach full conviction: ${opportunityQuality(exceptional)}`);
  const p=planTreasuryCycle({state:s,opportunities:[exceptional],observedAt:"2026-09-11T13:00:00Z",positionIdFor:o=>`p-${o.assetId}`});
  if(p.actions.length!==1||p.actions[0].kind!=="OPEN")throw new Error(JSON.stringify(p.actions));
  if(p.deploymentPct<.999)throw new Error(`full conviction was arbitrarily capped: ${p.deploymentPct}`);
  if(p.state.cashUsd<0||p.state.cashUsd>1e-6)throw new Error(`entry-cost reserve should leave cash at zero, got ${p.state.cashUsd}`);
  if(!(p.actions[0].capitalUsd>9900&&p.actions[0].capitalUsd<10000))throw new Error(`unexpected full-conviction capital ${p.actions[0].capitalUsd}`);
});

Deno.test("full-conviction adverse mark stays persistable and can still exit on risk",()=>{
  const initial=initialTreasuryState("2026-09-11T12:59:00Z");
  const exceptional=opp({assetId:"EVENTUSDT",expectedNetEdgeBps:80,reliabilityConfidence:.65,matureGroupCount:8,roundTripCostBps:20,sourceDecisionId:"event-1"});
  const opened=planTreasuryCycle({state:initial,opportunities:[exceptional],observedAt:"2026-09-11T13:00:00Z",positionIdFor:o=>`p-${o.assetId}`});
  const marked=planTreasuryCycle({
    state:opened.state,
    opportunities:[opp({assetId:"EVENTUSDT",observedAt:"2026-09-11T13:00:30Z",referencePrice:99.99,expectedNetEdgeBps:80,reliabilityConfidence:.65,matureGroupCount:8,roundTripCostBps:20,sourceDecisionId:"event-2"})],
    observedAt:"2026-09-11T13:01:00Z",
    positionIdFor:o=>`unused-${o.assetId}`,
  });
  if(marked.deploymentPct>1.000001||marked.blockedReasons.some(reason=>reason.includes("deployment exceeds")))throw new Error(JSON.stringify(marked));
  const stopped=planTreasuryCycle({
    state:marked.state,
    opportunities:[opp({assetId:"EVENTUSDT",observedAt:"2026-09-11T13:01:30Z",referencePrice:98,expectedNetEdgeBps:80,reliabilityConfidence:.65,matureGroupCount:8,roundTripCostBps:20,sourceDecisionId:"event-3"})],
    observedAt:"2026-09-11T13:02:00Z",
    positionIdFor:o=>`unused-${o.assetId}`,
  });
  if(stopped.actions[0]?.reason!=="RISK_STOP"||stopped.state.positions.length!==0||stopped.deploymentPct!==0)throw new Error(JSON.stringify(stopped));
});

Deno.test("exceptional new opportunity can close a weaker 6000 USD position and recycle nearly the whole cashbox",()=>{
  const s=initialTreasuryState("2026-09-11T12:50:00Z");
  s.cashUsd=4000;
  s.positions=[pos({assetId:"OLDUSDT",capitalUsd:6000,entryExpectedNetEdgeBps:8,latestExpectedNetEdgeBps:8,sourceDecisionId:"old-entry"})];
  const oldRefresh=opp({assetId:"OLDUSDT",referencePrice:100,expectedNetEdgeBps:8,reliabilityConfidence:.54,matureGroupCount:2,sourceDecisionId:"old-refresh"});
  const exceptional=opp({assetId:"EVENTUSDT",expectedNetEdgeBps:80,reliabilityConfidence:.65,matureGroupCount:8,sourceDecisionId:"event-breakout"});
  const p=planTreasuryCycle({state:s,opportunities:[oldRefresh,exceptional],observedAt:"2026-09-11T13:00:00Z",positionIdFor:o=>`new-${o.assetId}`});
  const exit=p.actions.find(a=>a.kind==="EXIT"&&a.assetId==="OLDUSDT");const open=p.actions.find(a=>a.kind==="OPEN"&&a.assetId==="EVENTUSDT");
  if(!exit||exit.reason!=="OPPORTUNITY_REPLACEMENT"||!open)throw new Error(JSON.stringify(p.actions));
  if(p.state.positions.length!==1||p.state.positions[0].assetId!=="EVENTUSDT")throw new Error(JSON.stringify(p.state.positions));
  if(p.deploymentPct<.99)throw new Error(`strong replacement failed to concentrate capital: ${p.deploymentPct}`);
});

Deno.test("replacement refuses a marginal edge that does not pay old exit plus remaining hold value",()=>{
  const s=initialTreasuryState("2026-09-11T12:50:00Z");s.cashUsd=4000;
  s.positions=[pos({assetId:"OLDUSDT",capitalUsd:6000,entryExpectedNetEdgeBps:10,latestExpectedNetEdgeBps:10,roundTripCostBps:20,sourceDecisionId:"old-entry"})];
  const oldRefresh=opp({assetId:"OLDUSDT",referencePrice:100,expectedNetEdgeBps:10,reliabilityConfidence:.55,matureGroupCount:3,roundTripCostBps:20,sourceDecisionId:"old-refresh"});
  const marginal=opp({assetId:"NEWUSDT",expectedNetEdgeBps:30,reliabilityConfidence:.65,matureGroupCount:8,roundTripCostBps:20,sourceDecisionId:"new-marginal"});
  const p=planTreasuryCycle({state:s,opportunities:[oldRefresh,marginal],observedAt:"2026-09-11T13:00:00Z",positionIdFor:o=>`new-${o.assetId}`});
  if(p.actions.some(a=>a.kind==="EXIT"&&a.assetId==="OLDUSDT"&&a.reason==="OPPORTUNITY_REPLACEMENT"))throw new Error(`marginal switch ignored incremental exit economics: ${JSON.stringify(p.actions)}`);
  if(!p.state.positions.some(row=>row.assetId==="OLDUSDT"))throw new Error("old position was incorrectly recycled");
});

Deno.test("strong edge alone cannot force all-in when reliability and maturity are weak",()=>{
  const s=initialTreasuryState("2026-09-11T12:59:00Z");
  const thin=opp({assetId:"THINUSDT",expectedNetEdgeBps:100,reliabilityConfidence:.51,matureGroupCount:2,sourceDecisionId:"thin-1"});
  const p=planTreasuryCycle({state:s,opportunities:[thin],observedAt:"2026-09-11T13:00:00Z",positionIdFor:o=>`p-${o.assetId}`});
  if(!p.actions.length)throw new Error("validated thin evidence should still receive a bounded position");
  if(p.deploymentPct>=.75)throw new Error(`one strong edge metric overrode weak evidence quality: ${p.deploymentPct}`);
});

Deno.test("treasury refuses non-PIT or non-ALLOW opportunities",()=>{
  const s=initialTreasuryState("2026-09-11T12:59:00Z");const p=planTreasuryCycle({state:s,opportunities:[opp({pitClear:false}),opp({assetId:"ETHUSDT",recommendation:"DOWNGRADE_TO_WAIT"})],observedAt:"2026-09-11T13:00:00Z",positionIdFor:o=>`p-${o.assetId}`});
  if(p.actions.length||p.state.positions.length)throw new Error(JSON.stringify(p.actions));
});

Deno.test("treasury direction flip recycles capital into the stronger opposite edge",()=>{
  const s=initialTreasuryState("2026-09-11T12:50:00Z");s.cashUsd=9000;s.positions=[pos()];const p=planTreasuryCycle({state:s,opportunities:[opp({direction:-1,referencePrice:101,expectedNetEdgeBps:30,sourceDecisionId:"d-short"})],observedAt:"2026-09-11T13:00:00Z",positionIdFor:o=>`new-${o.direction}`});
  if(p.actions.length<2||p.actions[0].kind!=="EXIT"||p.actions[0].reason!=="DIRECTION_FLIP"||p.actions[1].kind!=="OPEN")throw new Error(JSON.stringify(p.actions));
  if(p.state.positions[0]?.direction!==-1)throw new Error("opposite edge was not opened");
  if(p.state.positions[0]?.openedAt!=="2026-09-11T13:00:00Z")throw new Error("flip reopen did not use execution cycle time");
});

Deno.test("treasury exits stale positions instead of holding blind",()=>{
  const s=initialTreasuryState("2026-09-11T12:00:00Z");s.cashUsd=9000;s.positions=[pos({openedAt:"2026-09-11T12:00:00Z"})];const p=planTreasuryCycle({state:s,opportunities:[],observedAt:"2026-09-11T13:00:00Z",positionIdFor:()=>"unused"});
  if(p.actions[0]?.reason!=="STALE_EDGE"||p.state.positions.length!==0)throw new Error(JSON.stringify(p.actions));
});

Deno.test("treasury hard risk stop closes losing shadow capital",()=>{
  const s=initialTreasuryState("2026-09-11T12:50:00Z");s.cashUsd=9000;s.positions=[pos()];const p=planTreasuryCycle({state:s,opportunities:[opp({referencePrice:97,expectedNetEdgeBps:20})],observedAt:"2026-09-11T13:00:00Z",positionIdFor:()=>"unused"});
  if(p.actions[0]?.reason!=="RISK_STOP"||p.state.positions.length!==0)throw new Error(JSON.stringify(p.actions));
});

Deno.test("treasury opportunity replacement removes weakest slot when eight positions are full",()=>{
  const s=initialTreasuryState("2026-09-11T12:50:00Z");s.cashUsd=3600;s.positions=Array.from({length:8},(_,i)=>pos({positionId:`p-${i}`,assetId:`A${i}USDT`,capitalUsd:800,entryExpectedNetEdgeBps:5+i,latestExpectedNetEdgeBps:5+i,sourceDecisionId:`old-${i}`}));
  const existing=s.positions.map((p,i)=>opp({assetId:p.assetId,referencePrice:100,expectedNetEdgeBps:5+i,sourceDecisionId:`refresh-${i}`}));
  const candidate=opp({assetId:"NEWUSDT",expectedNetEdgeBps:40,sourceDecisionId:"new-edge"});const p=planTreasuryCycle({state:s,opportunities:[...existing,candidate],observedAt:"2026-09-11T13:00:00Z",positionIdFor:o=>`new-${o.assetId}`});
  if(!p.actions.some(a=>a.kind==="EXIT"&&a.reason==="OPPORTUNITY_REPLACEMENT")||!p.actions.some(a=>a.kind==="OPEN"&&a.assetId==="NEWUSDT"))throw new Error(JSON.stringify(p.actions));
  if(p.state.positions.length!==8)throw new Error(`unexpected position count ${p.state.positions.length}`);
});
