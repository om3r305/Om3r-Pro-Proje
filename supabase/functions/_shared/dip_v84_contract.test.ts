import assert from "node:assert/strict";
import {
  CALIBRATION_FAMILY_ID, calibrationNoEdge, canonicalJson, executionCalibration, executionPermission,
  manifestHash, RELEASE_ID, STRATEGY_MANIFEST, STRATEGY_MANIFEST_HASH, unavailableCalibration, validateSession,
  DB_CONTRACT_VERSION, DECISION_REVISION, ENGINE_VERSION, LOGIC_HASH, METRIC_VERSION, POLICY_VERSION,
} from "./dip_v84_contract.ts";

Deno.test("strategy manifest uses deep canonical hashing",async()=>{
  assert.equal(await manifestHash(STRATEGY_MANIFEST),STRATEGY_MANIFEST_HASH);
  assert.equal(canonicalJson({b:{z:1,a:2},a:3}),'{"a":3,"b":{"a":2,"z":1}}');
  assert.notEqual(await manifestHash({a:{x:1}}),await manifestHash({a:{x:2}}));
});

Deno.test("calibration has three distinct states and cold is not no-edge",()=>{
  const unavailable=unavailableCalibration("RPC_FAILURE");
  assert.equal(unavailable.state,"UNAVAILABLE");
  assert.equal(executionPermission(unavailable).entryAllowed,false);
  const cold=executionCalibration({wins:0,losses:0,episodes:0,days:0,ambiguousLosses:0});
  assert.equal(cold.state,"COLD_NEW_FAMILY");
  assert.equal(cold.p,null);assert.equal(cold.lower,null);assert.equal(calibrationNoEdge(cold,2),null);
  assert.equal(executionPermission(cold).entryAllowed,true);assert.equal(executionPermission(cold).leverage,1);assert.equal(executionPermission(cold).maxNotionalFraction,.08);
  const warm=executionCalibration({wins:30,losses:10,episodes:40,days:20,ambiguousLosses:2});
  assert.equal(warm.state,"WARM");assert.equal(warm.samples,40);assert.ok(warm.p!==null&&warm.lower!==null&&warm.upper!==null);
  assert.equal(executionPermission(warm).leverage,1);assert.equal(executionPermission(warm).maxNotionalFraction,.08);
});

Deno.test("ambiguous execution losses are counted as losses, not null samples",()=>{
  const cal=executionCalibration({wins:20,losses:20,episodes:40,days:10,ambiguousLosses:7});
  assert.equal(cal.samples,40);assert.equal(cal.losses,20);assert.equal(cal.ambiguousLosses,7);assert.equal(cal.p,.5);
});

Deno.test("V8.4 session contract is strict shadow-only and isolated",()=>{
  const cfg={release_id:RELEASE_ID,logic_hash:LOGIC_HASH,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,calibration_family_id:CALIBRATION_FAMILY_ID,db_contract_version:DB_CONTRACT_VERSION,engine_version:ENGINE_VERSION,policy_version:POLICY_VERSION,decision_revision:DECISION_REVISION,metric_version:METRIC_VERSION,symbols:["ETHUSDT"],shadow_only:true,live_execution:false,browser_execution:false,server_authoritative:true,allow_shadow_short:true,max_shadow_leverage:1,execution_mode:"SHADOW_PAPER"};
  validateSession(cfg);
  assert.throws(()=>validateSession({...cfg,live_execution:true}),/SHADOW/);
  assert.throws(()=>validateSession({...cfg,max_shadow_leverage:2}),/RISK/);
  assert.throws(()=>validateSession({...cfg,calibration_family_id:"legacy"}),/RELEASE/);
  assert.throws(()=>validateSession({...cfg,logic_hash:"wrong"}),/RELEASE/);
});
