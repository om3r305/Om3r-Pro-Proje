import assert from "node:assert/strict";
import { assertReleaseSealed } from "./worker.ts";
import { closePosition } from "./execution.ts";
import type { Runtime } from "../_shared/dip_v84_contract.ts";

Deno.test("sealed source still fails closed on release-registry mismatch",async()=>{
  const chain:any={
    select:()=>chain,eq:()=>chain,maybeSingle:async()=>({error:null,data:{status:"SEALED",logic_hash:"wrong",strategy_manifest_hash:"wrong",calibration_family_id:"wrong",db_contract_version:"wrong"}}),
  };
  const db:any={from:()=>chain};
  await assert.rejects(()=>assertReleaseSealed(db),/V84_RELEASE_REGISTRY_MISMATCH/);
});

Deno.test("ordered trade trigger cannot replace the frozen stop barrier",()=>{
  const rt:Runtime={
    start:500,cash:459.96,realized:0,trades:0,wins:0,losses:0,lastOccurrence:"occ",lastSnapshotHour:null,marketCursor:0,lastClosedAt:null,latestThesis:null,
    pos:{
      side:"LONG",position_id:"p",thesis_id:"occ",episode_id:"ep",setup:"SWEEP_RECLAIM",regime:"RANGE",
      entry:100,qty:1,notional:100,target:105,stop:99,stop_source:"SWEEP_TRIGGER_PIVOT",stop_structure_id:"1m:1:L",
      opened_at:new Date(0).toISOString(),due_at:new Date(90*60_000).toISOString(),fees_open:.04,fee_bps:4,slippage_bps:1,
      expected_exit_spread_bps:2,expected_exit_slippage_bps:1,venue:"SHADOW_PERP",policy_version:"test",release_id:"test",calibration_family_id:"test",
      checked_until:0,market_price:100,leverage:1,margin:40,actual_fraction:.2,gross_fraction:.2,
    },
  };
  const event=closePosition(rt,{reason:"INVALIDATION_FIRST",hit:false,price:95,eventAt:10,eventRangeStart:10,checkedUntil:20},20) as Record<string,any>;
  assert.equal(event.metadata.path_trigger_price,95);
  assert.equal(event.metadata.execution_barrier_price,99);
  const expectedExit=99*(1-(2/2+1)/10000);
  assert.ok(Math.abs(Number(event.exit_price)-expectedExit)<1e-12);
});
