import {
  ACCOUNT_RISK_FRACTION,
  COLD_MAX_NOTIONAL_FRACTION,
  type Direction,
  type ExecutionCalibration,
  executionPermission,
} from "../_shared/dip_v84_contract.ts";
import type { Economics } from "./cost.ts";

export type RiskRules={minNotional:number;minQty:number;maxQty:number;stepSize:number};

export function sizePosition(input:{
  cash:number;
  tradeNotional:number;
  direction:Direction;
  entry:number;
  stop:number;
  feeOpenBps:number;
  economics:Economics;
  calibration:ExecutionCalibration;
  rules:RiskRules;
  rawConviction:number;
}){
  const {cash,tradeNotional,direction,entry,stop,feeOpenBps,economics,calibration,rules,rawConviction}=input;
  const permission=executionPermission(calibration);
  if(!permission.entryAllowed)return null;
  if(!(cash>0&&tradeNotional>0&&entry>0&&stop>0&&feeOpenBps>=0)||!Number.isFinite(rawConviction))return null;
  if(direction==="UP"?stop>=entry:stop<=entry)return null;
  if(!(economics.stop_net_loss_bps>0))return null;
  // Package 1 evidence release is always 1x and never exceeds the cold 8% gross-notional ceiling.
  const leverage:1=1;
  const maxNotional=Math.min(tradeNotional,cash*COLD_MAX_NOTIONAL_FRACTION,cash*permission.maxNotionalFraction);
  const maxMargin=Math.min(cash,maxNotional);
  const riskBudget=cash*ACCOUNT_RISK_FRACTION;
  const lossPerUnit=entry*economics.stop_net_loss_bps/10000;
  const byRisk=riskBudget/lossPerUnit;
  const byExposure=maxNotional/entry;
  const byCash=cash/(entry+entry*feeOpenBps/10000);
  const budgetQty=Math.min(byRisk,byExposure,byCash,rules.maxQty);
  const qty=Math.floor((budgetQty+Number.EPSILON)/rules.stepSize)*rules.stepSize;
  const notional=qty*entry,margin=notional,fees_open=notional*feeOpenBps/10000,worstLoss=qty*lossPerUnit;
  if(!Number.isFinite(qty)||qty<rules.minQty||notional<rules.minNotional||margin+fees_open>cash+1e-8||margin>maxMargin+1e-8||worstLoss>riskBudget+1e-8)return null;
  return{
    qty,notional,margin,leverage,
    actual_fraction:margin/cash,
    gross_fraction:notional/cash,
    fees_open,worst_loss:worstLoss,risk_fraction:worstLoss/cash,
    allocation:1,
    risk_policy:"V84_PACKAGE1_EVIDENCE_COLD_ONLY",
    max_notional_fraction:COLD_MAX_NOTIONAL_FRACTION,
  };
}
