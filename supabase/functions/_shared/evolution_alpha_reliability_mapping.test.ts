import { bindLaggedReliabilityToDecision, type DecisionSourceObservation, type ReliabilitySnapshotCandidate } from "./evolution_alpha_reliability_mapping.ts";

const source=(overrides:Partial<DecisionSourceObservation>={}):DecisionSourceObservation=>({
  observationId:"obs-1",independentGroup:"price_structure",sensorFamily:"structure",sensorHorizon:"FAST_5_30M",direction:1,observedAt:"2026-09-11T12:55:00Z",...overrides,
});
const snap=(overrides:Partial<ReliabilitySnapshotCandidate>={}):ReliabilitySnapshotCandidate=>({
  independentGroup:"price_structure",sensorFamily:"structure",sensorHorizon:"FAST_5_30M",sampleCount:500,bayesianHitRate:.6,avgSignedBps:18,avgCostAdjustedSignedBps:-4,outcomeHorizonSeconds:900,snapshotWindowEnd:"2026-09-11T12:00:00Z",snapshotGeneratedAt:"2026-09-11T12:05:00Z",...overrides,
});

Deno.test("reliability binding uses the exact sensor family and horizon that voted",()=>{
  const rows=bindLaggedReliabilityToDecision({
    direction:1,supportGroups:["price_structure"],sourceObservations:[source()],snapshotCandidates:[
      snap({sensorFamily:"different",sampleCount:5000,avgSignedBps:70}),
      snap({sampleCount:500,avgSignedBps:18}),
    ],
  });
  if(rows.length!==1||rows[0].sampleCount!==500||rows[0].avgSignedBps!==18)throw new Error(JSON.stringify(rows));
});

Deno.test("micro reliability is mapped from raw sensor group to ALPHA intrabar_tape",()=>{
  const rows=bindLaggedReliabilityToDecision({
    direction:-1,supportGroups:["intrabar_tape"],sourceObservations:[source({observationId:"micro",independentGroup:"micro_taker_flow",sensorFamily:"taker",sensorHorizon:"MICRO_1_5M",direction:-1})],snapshotCandidates:[
      snap({independentGroup:"micro_taker_flow",sensorFamily:"taker",sensorHorizon:"MICRO_1_5M",sampleCount:900,avgSignedBps:24}),
    ],
  });
  if(rows.length!==1||rows[0].group!=="intrabar_tape"||rows[0].avgSignedBps!==24)throw new Error(JSON.stringify(rows));
});

Deno.test("conflicting source observations cannot lend reliability to the chosen direction",()=>{
  const rows=bindLaggedReliabilityToDecision({
    direction:1,supportGroups:["price_structure"],sourceObservations:[source({direction:-1})],snapshotCandidates:[snap()],
  });
  if(rows.length!==0)throw new Error(JSON.stringify(rows));
});

Deno.test("avg signed bps stays sensor-aligned and is not flipped again for shorts",()=>{
  const rows=bindLaggedReliabilityToDecision({
    direction:-1,supportGroups:["price_structure"],sourceObservations:[source({direction:-1})],snapshotCandidates:[snap({avgSignedBps:21})],
  });
  if(rows.length!==1||rows[0].avgSignedBps!==21)throw new Error(JSON.stringify(rows));
});
