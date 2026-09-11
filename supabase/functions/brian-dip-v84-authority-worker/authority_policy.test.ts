import { assert, assertEquals, assertThrows } from "jsr:@std/assert@1";
import { LOGIC_HASH, STRATEGY_MANIFEST, validateSession, RELEASE_ID, STRATEGY_MANIFEST_HASH, CALIBRATION_FAMILY_ID, DB_CONTRACT_VERSION, ENGINE_VERSION, POLICY_VERSION, DECISION_REVISION, METRIC_VERSION } from "../_shared/dip_v84_authority_contract.ts";

Deno.test("V8.4.1 manifest names Brian as sole strategic authority",()=>{
  assertEquals(STRATEGY_MANIFEST.decision_authority,"BRIAN_SINGLE_STRATEGY_AUTHORITY");
  assertEquals(STRATEGY_MANIFEST.target.policy,"BRIAN_STRUCTURAL_LADDER");
  assertEquals(STRATEGY_MANIFEST.target.l1_mandatory,false);
  assertEquals(STRATEGY_MANIFEST.risk.fixed_notional_fraction_cap,false);
  assertEquals(STRATEGY_MANIFEST.live_execution,false);
  assertEquals(STRATEGY_MANIFEST.browser_execution,false);
});

Deno.test("authority session remains hard-locked to SHADOW",()=>{
  const good={release_id:RELEASE_ID,logic_hash:LOGIC_HASH,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,calibration_family_id:CALIBRATION_FAMILY_ID,db_contract_version:DB_CONTRACT_VERSION,engine_version:ENGINE_VERSION,policy_version:POLICY_VERSION,decision_revision:DECISION_REVISION,metric_version:METRIC_VERSION,symbols:["ETHUSDT"],shadow_only:true,live_execution:false,browser_execution:false,server_authoritative:true,allow_shadow_short:true,max_shadow_leverage:1,execution_mode:"SHADOW_PAPER",decision_authority:"BRIAN"};
  validateSession(good);
  assertThrows(()=>validateSession({...good,live_execution:true}),Error,"V841_INVALID_SHADOW_CONTRACT");
  assertThrows(()=>validateSession({...good,decision_authority:"GATE_CHAIN"}),Error,"V841_AUTHORITY_CONTRACT_MISMATCH");
  assertThrows(()=>validateSession({...good,max_shadow_leverage:2}),Error,"V841_INVALID_RISK_CONTRACT");
});

Deno.test("strategic legacy vetoes are soft evidence, not canEnter veto chain",async()=>{
  const src=await Deno.readTextFile(new URL("./authority.ts",import.meta.url));
  for(const name of ["TARGET_BELOW_COST","ECONOMIC_RR_TOO_LOW","WAIT_RETEST","ENTRY_TOO_LATE","DIRECTION_REFEREE_REJECT","COUNTER_STRUCTURE","OPPOSING_MULTI_FLOW","RAW_CONVICTION_LOW"])assert(src.includes(`\"${name}\"`),`missing soft evidence ${name}`);
  assert(src.includes("soft_evidence:softEvidence"));
  assert(src.includes("hard.length===0"));
  assert(!src.includes("veto.length===0"));
  assert(src.includes('policy:"BRIAN_STRUCTURAL_LADDER"'));
});
