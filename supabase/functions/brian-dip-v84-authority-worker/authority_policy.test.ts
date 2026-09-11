import { assert, assertEquals, assertThrows } from "jsr:@std/assert@1";
import { LOGIC_HASH, STRATEGY_MANIFEST, validateSession, RELEASE_ID, STRATEGY_MANIFEST_HASH, CALIBRATION_FAMILY_ID, DB_CONTRACT_VERSION, ENGINE_VERSION, POLICY_VERSION, DECISION_REVISION, METRIC_VERSION } from "../_shared/dip_v84_authority_contract.ts";

Deno.test("V8.4.2 manifest is Brian long-only stateful authority",()=>{
  assertEquals(STRATEGY_MANIFEST.decision_authority,"BRIAN_LONG_ONLY_STATEFUL");
  assertEquals(STRATEGY_MANIFEST.target.policy,"BRIAN_DYNAMIC_ADVISORY_TARGET");
  assertEquals(STRATEGY_MANIFEST.target.hard_take_profit,false);
  assertEquals(STRATEGY_MANIFEST.target.trailing_thesis,true);
  assertEquals(STRATEGY_MANIFEST.short_entries,false);
  assertEquals(STRATEGY_MANIFEST.entry.direction_and_quality_separate,true);
  assertEquals(STRATEGY_MANIFEST.risk.fixed_notional_fraction_cap,false);
  assertEquals(STRATEGY_MANIFEST.live_execution,false);
  assertEquals(STRATEGY_MANIFEST.browser_execution,false);
});

Deno.test("V8.4.2 session remains SHADOW and short-disabled",()=>{
  const good={release_id:RELEASE_ID,logic_hash:LOGIC_HASH,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,calibration_family_id:CALIBRATION_FAMILY_ID,db_contract_version:DB_CONTRACT_VERSION,engine_version:ENGINE_VERSION,policy_version:POLICY_VERSION,decision_revision:DECISION_REVISION,metric_version:METRIC_VERSION,symbols:["ETHUSDT"],shadow_only:true,live_execution:false,browser_execution:false,server_authoritative:true,allow_shadow_short:false,max_shadow_leverage:1,execution_mode:"SHADOW_PAPER",decision_authority:"BRIAN"};
  validateSession(good);
  assertThrows(()=>validateSession({...good,live_execution:true}),Error,"V842_INVALID_SHADOW_CONTRACT");
  assertThrows(()=>validateSession({...good,decision_authority:"GATE_CHAIN"}),Error,"V842_AUTHORITY_CONTRACT_MISMATCH");
  assertThrows(()=>validateSession({...good,max_shadow_leverage:2}),Error,"V842_INVALID_LONG_ONLY_CONTRACT");
  assertThrows(()=>validateSession({...good,allow_shadow_short:true}),Error,"V842_INVALID_LONG_ONLY_CONTRACT");
});

Deno.test("Brian separates direction from entry quality and blocks chase internally",async()=>{
  const src=await Deno.readTextFile(new URL("./authority.ts",import.meta.url));
  assert(src.includes("authority_entry_quality"));
  assert(src.includes("BRIAN_CHASE_FOMO_WAIT"));
  assert(src.includes("LONG_ONLY_BEARISH_WAIT"));
  assert(src.includes('policy:"BRIAN_DYNAMIC_ADVISORY_TARGET"'));
  assert(src.includes("hard_take_profit:false"));
  assert(!src.includes('authority_action:action==="SHORT"'));
});

Deno.test("worker never opens SHORT and target is advisory",async()=>{
  const src=await Deno.readTextFile(new URL("./worker.ts",import.meta.url));
  assert(src.includes('side:"LONG"'));
  assert(!src.includes('event_kind:"SHORT_OPEN"'));
  assert(src.includes("Advisory target is intentionally NOT an automatic exit"));
  assert(src.includes("BRIAN_WAIT_REBASE_AFTER_EXIT"));
});