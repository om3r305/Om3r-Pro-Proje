import { COST_MODEL_VERSION, type Direction } from "../_shared/dip_v84_contract.ts";

export type CostContract = {
  model:typeof COST_MODEL_VERSION;
  feeOpenBps:number;
  feeCloseBps:number;
  openingSlippageBps:number;
  expectedExitSpreadBps:number;
  expectedExitSlippageBps:number;
  expectedFundingBps:number;
};

export type Economics = {
  model:typeof COST_MODEL_VERSION;
  gross_reward_bps:number;
  gross_stop_bps:number;
  fill_forward_cost_bps:number;
  target_net_reward_bps:number;
  stop_net_loss_bps:number;
  economic_rr:number;
  target_exit_price:number;
  stop_exit_price:number;
};

function finiteNonNegative(name:string,value:number):number {
  if(!Number.isFinite(value)||value<0)throw Error(`V84_INVALID_${name}`);
  return value;
}

export function buildCostContract(input:{
  feeOpenBps:number;
  feeCloseBps?:number;
  openingSlippageBps:number;
  expectedExitSpreadBps:number;
  expectedExitSlippageBps?:number;
  expectedFundingBps:number;
}):CostContract {
  const feeOpenBps=finiteNonNegative("FEE_OPEN_BPS",input.feeOpenBps);
  const feeCloseBps=finiteNonNegative("FEE_CLOSE_BPS",input.feeCloseBps??input.feeOpenBps);
  const openingSlippageBps=finiteNonNegative("OPENING_SLIPPAGE_BPS",input.openingSlippageBps);
  const expectedExitSpreadBps=finiteNonNegative("EXIT_SPREAD_BPS",input.expectedExitSpreadBps);
  const expectedExitSlippageBps=finiteNonNegative("EXIT_SLIPPAGE_BPS",input.expectedExitSlippageBps??input.openingSlippageBps);
  const expectedFundingBps=finiteNonNegative("EXPECTED_FUNDING_BPS",input.expectedFundingBps);
  return{model:COST_MODEL_VERSION,feeOpenBps,feeCloseBps,openingSlippageBps,expectedExitSpreadBps,expectedExitSlippageBps,expectedFundingBps};
}

// Entry is already an executable simulated fill: LONG ask+open-slip, SHORT bid-open-slip.
// Opening spread and opening slippage are therefore NOT charged again here.
export function fillForwardCostBps(cost:CostContract):number {
  return cost.feeOpenBps+cost.feeCloseBps+cost.expectedExitSpreadBps/2+cost.expectedExitSlippageBps+cost.expectedFundingBps;
}

// For structural/reference comparisons only. Never use this to authorize a fill.
export function referenceRoundtripCostBps(cost:CostContract,currentSpreadBps:number):number {
  finiteNonNegative("CURRENT_SPREAD_BPS",currentSpreadBps);
  return cost.feeOpenBps+cost.feeCloseBps+currentSpreadBps+cost.openingSlippageBps+cost.expectedExitSlippageBps+cost.expectedFundingBps;
}

// Barrier is a structural/reference price. Convert to the executable side exactly once,
// then apply exit slippage exactly once. If a future caller already has a bid/ask, it
// must call economicsFromExecutableExit instead of this conversion path.
export function executableExitFromBarrier(direction:Direction,barrier:number,cost:CostContract):number {
  if(!(barrier>0)&&Number.isFinite(barrier))throw Error("V84_INVALID_EXIT_BARRIER");
  if(!(barrier>0)||!Number.isFinite(barrier))throw Error("V84_INVALID_EXIT_BARRIER");
  const friction=(cost.expectedExitSpreadBps/2+cost.expectedExitSlippageBps)/10000;
  return barrier*(direction==="UP"?1-friction:1+friction);
}

export function economicsFromExecutableExit(direction:Direction,entry:number,targetExecutable:number,stopExecutable:number,cost:CostContract):Economics {
  if(![entry,targetExecutable,stopExecutable].every(x=>Number.isFinite(x)&&x>0))throw Error("V84_INVALID_EXECUTION_PRICE");
  const sign=direction==="UP"?1:-1;
  const targetGross=sign*(targetExecutable-entry);
  const stopGross=sign*(stopExecutable-entry);
  const targetFees=entry*cost.feeOpenBps/10000+targetExecutable*cost.feeCloseBps/10000+entry*cost.expectedFundingBps/10000;
  const stopFees=entry*cost.feeOpenBps/10000+stopExecutable*cost.feeCloseBps/10000+entry*cost.expectedFundingBps/10000;
  const targetNet=targetGross-targetFees;
  const stopNet=-(stopGross-stopFees);
  const targetNetBps=targetNet/entry*10000;
  const stopNetBps=stopNet/entry*10000;
  return{
    model:COST_MODEL_VERSION,
    gross_reward_bps:Math.max(0,targetGross/entry*10000),
    gross_stop_bps:Math.max(0,-stopGross/entry*10000),
    fill_forward_cost_bps:fillForwardCostBps(cost),
    target_net_reward_bps:Math.max(0,targetNetBps),
    stop_net_loss_bps:Math.max(0,stopNetBps),
    economic_rr:stopNet>0?Math.max(0,targetNet)/stopNet:0,
    target_exit_price:targetExecutable,
    stop_exit_price:stopExecutable,
  };
}

export function executionEconomics(direction:Direction,entry:number,targetBarrier:number,stopBarrier:number,cost:CostContract):Economics {
  const targetExit=executableExitFromBarrier(direction,targetBarrier,cost);
  const stopExit=executableExitFromBarrier(direction,stopBarrier,cost);
  return economicsFromExecutableExit(direction,entry,targetExit,stopExit,cost);
}
