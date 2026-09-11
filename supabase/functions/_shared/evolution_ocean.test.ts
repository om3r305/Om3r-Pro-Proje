import { activeOceanRun, buildOceanReport, deriveOceanRuns, type OceanCommand, type OceanRunState } from "./evolution_ocean.ts";

function start(overrides:Partial<OceanCommand>={}):OceanCommand{return{commandId:"start-1",runId:"ocean-1",command:"START",requestedAt:"2026-09-11T10:00:00Z",durationHours:24,reason:null,...overrides};}
function run():OceanRunState{return{runId:"ocean-1",startedAt:"2026-09-11T10:00:00.000Z",plannedEndAt:"2026-09-12T10:00:00.000Z",stoppedAt:null,effectiveEndAt:"2026-09-12T10:00:00.000Z",durationHours:24,status:"ENDED",startCommandId:"start-1",stopCommandId:null};}

Deno.test("Ocean run stays active until its planned end",()=>{
  const active=activeOceanRun([start()],"2026-09-11T18:00:00Z");
  if(!active||active.runId!=="ocean-1"||active.status!=="ACTIVE")throw new Error(JSON.stringify(active));
  const ended=deriveOceanRuns([start()],"2026-09-12T10:00:01Z")[0];
  if(ended.status!=="ENDED"||ended.effectiveEndAt!=="2026-09-12T10:00:00.000Z")throw new Error(JSON.stringify(ended));
});

Deno.test("Ocean STOP command ends a run early without mutating its planned end",()=>{
  const commands=[start(),{commandId:"stop-1",runId:"ocean-1",command:"STOP",requestedAt:"2026-09-11T16:30:00Z",durationHours:null,reason:"operator stop"} satisfies OceanCommand];
  const state=deriveOceanRuns(commands,"2026-09-11T17:00:00Z")[0];
  if(state.status!=="ENDED"||state.effectiveEndAt!=="2026-09-11T16:30:00.000Z"||state.plannedEndAt!=="2026-09-12T10:00:00.000Z")throw new Error(JSON.stringify(state));
});

Deno.test("Ocean ignores malformed starts and only accepts 24h or 48h contracts",()=>{
  const malformed=start({durationHours:null});
  if(deriveOceanRuns([malformed],"2026-09-11T12:00:00Z").length!==0)throw new Error("malformed Ocean run accepted");
  const valid=start({durationHours:48});
  const state=deriveOceanRuns([valid],"2026-09-11T12:00:00Z")[0];
  if(state.durationHours!==48||state.plannedEndAt!=="2026-09-13T10:00:00.000Z")throw new Error(JSON.stringify(state));
});

Deno.test("Ocean report reconciles Treasury and prospective evidence counts",()=>{
  const report=buildOceanReport({
    run:run(),
    treasuryStart:{observedAt:"2026-09-11T10:00:00Z",equityUsd:10_000,cashUsd:10_000,deploymentUsd:0,realizedPnlUsd:0,cumulativeCostsUsd:0,openPositions:0},
    treasuryEnd:{observedAt:"2026-09-12T10:00:00Z",equityUsd:10_140,cashUsd:4_000,deploymentUsd:6_100,realizedPnlUsd:125,cumulativeCostsUsd:35,openPositions:6},
    treasuryActions:20,replacements:3,newSources:8,newHypotheses:4,newCodeCandidates:2,experimentResults:12,promotionCandidates:1,rejectedPromotions:2,driftEvents:3,capabilityEvents:5,missedOpportunities:9,alphaOutcomeSamples:100,alphaFavorableAfterCost:42,collectorRuns:500,collectorFailures:5,degradedRuns:10,
  });
  if(report.treasury.netPnlUsd!==140||report.alpha.favorableAfterCostRate!==.42)throw new Error(JSON.stringify(report));
  if(Math.abs(Number(report.health.healthyRunRate)-.97)>1e-9)throw new Error(JSON.stringify(report.health));
  if(!report.shadowOnly||report.liveExecution)throw new Error("Ocean report execution boundary broken");
});
