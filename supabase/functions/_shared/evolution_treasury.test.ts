import { initialTreasuryState, planTreasuryCycle, treasuryEquityUsd, type TreasuryOpportunity, type TreasuryPosition } from "./evolution_treasury.ts";

function opp(overrides:Partial<TreasuryOpportunity>={}):TreasuryOpportunity{return{
  assetId:"BTCUSDT",direction:1,observedAt:"2026-09-11T12:59:00Z",referencePrice:100,expectedNetEdgeBps:25,roundTripCostBps:20,
  reliabilityConfidence:.6,matureGroupCount:3,pitClear:true,recommendation:"ALLOW_EDGE",sourceDecisionId:"d-1",...overrides,
};}
function pos(overrides:Partial<TreasuryPosition>={}):TreasuryPosition{return{
  positionId:"p-1",assetId:"BTCUSDT",direction:1,openedAt:"2026-09-11T12:50:00Z",entryPrice:100,capitalUsd:1000,
  entryExpectedNetEdgeBps:20,latestExpectedNetEdgeBps:20,roundTripCostBps:20,sourceDecisionId:"d-old",highWaterPnlBps:0,...overrides,
};}

Deno.test("treasury starts as one 10000 USD cash pool",()=>{const s=initialTreasuryState("2026-09-11T13:00:00Z");if(s.cashUsd!==10000||s.positions.length||treasuryEquityUsd(s,{})!==10000)throw new Error(JSON.stringify(s));});

Deno.test("treasury allocates only validated positive expected edge and preserves reserve",()=>{
  const s=initialTreasuryState("2026-09-11T12:59:00Z");const p=planTreasuryCycle({state:s,opportunities:[opp()],observedAt:"2026-09-11T13:00:00Z",positionIdFor:o=>`p-${o.assetId}`});
  if(p.actions.length!==1||p.actions[0].kind!=="OPEN")throw new Error(JSON.stringify(p.actions));
  if(p.state.positions.length!==1||p.deploymentPct>.700001||p.cashReservePct<.29)throw new Error(JSON.stringify(p));
  if(p.state.positions[0]?.openedAt!=="2026-09-11T13:00:00Z")throw new Error(`open timestamp leaked from signal time: ${p.state.positions[0]?.openedAt}`);
  if(!(p.afterEquityUsd<10000))throw new Error("entry cost should reduce equity");
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
