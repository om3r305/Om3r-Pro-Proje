import { measureActionGateExperiment, measureGateExperiment, measureOutcomeSet, type ProspectiveOutcomePoint } from "./evolution_lab.ts";

function point(i:number,netPositive:boolean):ProspectiveOutcomePoint{
  const grossBps=netPositive?35:5;
  return{
    decisionId:`d-${i}`,
    observedAt:new Date(Date.parse("2026-09-08T00:00:00Z")+i*30*60_000).toISOString(),
    directionAdjustedReturn:grossBps/10_000,
    estimatedRoundTripCostBps:20,
    classification:netPositive?"ACTION_FAVORABLE_AFTER_COST":"ACTION_UNFAVORABLE_AFTER_COST",
  };
}

Deno.test("prospective metric builder subtracts decision-time cost",()=>{
  const rows=Array.from({length:40},(_,i)=>point(i,i%2===0));
  const m=measureOutcomeSet(rows,0);
  if(m.samples!==40||m.grossEdgeBps==null||m.netEdgeBps==null)throw new Error(JSON.stringify(m));
  if(Math.abs((m.grossEdgeBps-m.netEdgeBps)-20)>1e-9)throw new Error("cost was not subtracted exactly");
  if(!m.dataQualityOk)throw new Error("mature valid sample should pass data quality");
});

Deno.test("action-gate challenger only measures explicitly allowed decisions",()=>{
  const outcomes=Array.from({length:80},(_,i)=>point(i,i%4===0));
  const labels=outcomes.map((row,i)=>({decisionId:row.decisionId,challengerAction:i%4===0?"ALLOW_ACTION":"DOWNGRADE_TO_WAIT"}));
  const measured=measureActionGateExperiment(outcomes,labels);
  if(measured.control.samples!==80||measured.challenger.samples!==20)throw new Error(JSON.stringify(measured.lineage));
  if(Number(measured.challenger.netEdgeBps)<=Number(measured.control.netEdgeBps))throw new Error("filtered challenger should improve this fixture");
  if(measured.challenger.complexityDelta!==1)throw new Error("challenger complexity delta missing");
});

Deno.test("generic gate can measure ALLOW_EDGE expected-edge challenger",()=>{
  const outcomes=Array.from({length:80},(_,i)=>point(i,i%5===0));
  const labels=outcomes.map((row,i)=>({decisionId:row.decisionId,challengerAction:i%5===0?"ALLOW_EDGE":"DOWNGRADE_TO_WAIT"}));
  const measured=measureGateExperiment(outcomes,labels,"ALLOW_EDGE",2);
  if(measured.challenger.samples!==16||measured.lineage.allowedLabel!=="ALLOW_EDGE")throw new Error(JSON.stringify(measured.lineage));
  if(Number(measured.challenger.netEdgeBps)<=Number(measured.control.netEdgeBps))throw new Error("expected-edge filter should improve fixture");
  if(measured.challenger.complexityDelta!==2)throw new Error("expected-edge complexity delta missing");
});

Deno.test("bad prospective rows fail data-quality gate instead of becoming zero",()=>{
  const rows=Array.from({length:40},(_,i)=>point(i,true));
  rows[0]={...rows[0],estimatedRoundTripCostBps:Number.NaN};
  const m=measureOutcomeSet(rows,0);
  if(m.dataQualityOk)throw new Error("invalid row should fail data quality");
  if(m.samples!==39)throw new Error(`unexpected valid sample count ${m.samples}`);
});
