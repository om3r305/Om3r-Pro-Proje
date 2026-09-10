import assert from "node:assert/strict";
import { buildCostContract, executionEconomics, fillForwardCostBps } from "./cost.ts";

Deno.test("fill-forward cost never recharges opening spread or opening slippage",()=>{
  const c=buildCostContract({feeOpenBps:10,feeCloseBps:10,openingSlippageBps:7,expectedExitSpreadBps:2,expectedExitSlippageBps:3,expectedFundingBps:1});
  assert.equal(fillForwardCostBps(c),25); // 10 open fee + 10 close fee + 1 exit half-spread + 3 exit slip + 1 funding
});

Deno.test("LONG and SHORT target/stop economics are directionally symmetric",()=>{
  const c=buildCostContract({feeOpenBps:5,openingSlippageBps:1,expectedExitSpreadBps:2,expectedExitSlippageBps:1,expectedFundingBps:0});
  const long=executionEconomics("UP",100,102,99,c);
  const short=executionEconomics("DOWN",100,98,101,c);
  assert.ok(Math.abs(long.target_net_reward_bps-short.target_net_reward_bps)<1e-9);
  assert.ok(Math.abs(long.stop_net_loss_bps-short.stop_net_loss_bps)<1e-9);
  assert.ok(Math.abs(long.economic_rr-short.economic_rr)<1e-9);
});

Deno.test("exit spread is applied exactly once to barrier prices",()=>{
  const c=buildCostContract({feeOpenBps:0,feeCloseBps:0,openingSlippageBps:9,expectedExitSpreadBps:4,expectedExitSlippageBps:0,expectedFundingBps:0});
  const e=executionEconomics("UP",100,101,99,c);
  assert.ok(Math.abs(e.target_exit_price-100.9798)<1e-10);
  assert.ok(Math.abs(e.stop_exit_price-98.9802)<1e-10);
  // Opening slippage value is intentionally absent from fill-forward PnL because entry is already the actual simulated fill.
  assert.ok(e.target_net_reward_bps>97&&e.target_net_reward_bps<99);
});
