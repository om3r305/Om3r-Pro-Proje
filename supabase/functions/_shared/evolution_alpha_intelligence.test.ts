import { boundedProspectiveReliability, estimateExpectedNetEdge, type ExpectedEdgeInput } from "./evolution_alpha_intelligence.ts";

function base():ExpectedEdgeInput{return{
  decisionObservedAt:"2026-09-11T13:00:00Z",direction:1,evidenceScore:.45,roundTripCostBps:20,
  reliability:[
    {group:"price_structure",sampleCount:800,bayesianHitRate:.62,avgSignedBps:42,avgCostAdjustedSignedBps:20,outcomeHorizonSeconds:900,snapshotWindowEnd:"2026-09-11T12:30:00Z",snapshotGeneratedAt:"2026-09-11T12:35:00Z"},
    {group:"derivatives_taker",sampleCount:700,bayesianHitRate:.59,avgSignedBps:36,avgCostAdjustedSignedBps:14,outcomeHorizonSeconds:900,snapshotWindowEnd:"2026-09-11T12:30:00Z",snapshotGeneratedAt:"2026-09-11T12:35:00Z"},
  ],
  freshness:[
    {group:"price_structure",observedAt:"2026-09-11T12:55:00Z",horizon:"FAST_5_30M"},
    {group:"derivatives_taker",observedAt:"2026-09-11T12:54:00Z",horizon:"FAST_5_30M"},
  ],minimumNetMarginBps:2,
};}

Deno.test("bounded prospective reliability shrinks and caps measured scores",()=>{
  const immature=boundedProspectiveReliability(20,.9),mature=boundedProspectiveReliability(2000,.9);
  if(!(immature>.5&&immature<mature))throw new Error(`${immature}/${mature}`);
  if(mature>.65+1e-12)throw new Error(`cap failed ${mature}`);
});

Deno.test("expected edge explicitly subtracts cost uncertainty and decay",()=>{
  const r=estimateExpectedNetEdge(base());
  if(r.expectedGrossMoveBps==null||r.expectedNetEdgeBps==null||r.uncertaintyPenaltyBps==null||r.eventDecayPenaltyBps==null)throw new Error(JSON.stringify(r));
  const expected=r.expectedGrossMoveBps-20-r.uncertaintyPenaltyBps-r.eventDecayPenaltyBps;
  if(Math.abs(expected-r.expectedNetEdgeBps)>1e-9)throw new Error("edge decomposition does not reconcile");
  if(!r.pitClear)throw new Error(JSON.stringify(r.reasons));
});

Deno.test("unknown decision-time cost fails closed instead of becoming zero",()=>{
  const r=estimateExpectedNetEdge({...base(),roundTripCostBps:null});
  if(r.recommendation!=="COST_UNAVAILABLE"||r.eligible||r.expectedNetEdgeBps!==null)throw new Error(JSON.stringify(r));
});

Deno.test("post-decision reliability cannot enter expected-edge estimate",()=>{
  const input=base();
  input.reliability[0]={...input.reliability[0],snapshotGeneratedAt:"2026-09-11T13:01:00Z"};
  const r=estimateExpectedNetEdge(input);
  if(r.pitClear||r.recommendation!=="CONTAMINATED_EVIDENCE"||r.eligible)throw new Error(JSON.stringify(r));
});

Deno.test("future source observation contaminates the edge and blocks action",()=>{
  const input=base();input.freshness[0]={...input.freshness[0],observedAt:"2026-09-11T13:02:00Z"};
  const r=estimateExpectedNetEdge(input);
  if(r.pitClear||r.eligible||r.recommendation!=="CONTAMINATED_EVIDENCE")throw new Error(JSON.stringify(r));
});

Deno.test("at least two mature independent groups are required",()=>{
  const input=base();input.reliability=input.reliability.slice(0,1);
  const r=estimateExpectedNetEdge(input);
  if(r.recommendation!=="INSUFFICIENT_LAGGED_EVIDENCE"||r.eligible)throw new Error(JSON.stringify(r));
});

Deno.test("negative lagged directional edge is downgraded even with strong evidence score",()=>{
  const input=base();input.evidenceScore=.95;input.reliability=input.reliability.map(row=>({...row,avgSignedBps:-8,avgCostAdjustedSignedBps:-28}));
  const r=estimateExpectedNetEdge(input);
  if(r.eligible||r.recommendation!=="DOWNGRADE_TO_WAIT"||Number(r.expectedNetEdgeBps)>=0)throw new Error(JSON.stringify(r));
});
