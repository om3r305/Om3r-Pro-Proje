from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import argparse, calendar, glob, json, subprocess, time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .data import canonical_hash
from .evaluation import evaluate_predictions
from .learning import GradientBoostingBaseline, LogisticRegressionBaseline, ProbabilityPrediction, metadata_for
from .meta_trader import MetaTrader
from .phase24_experiment import _ema, _roll
from .policy import PolicyThresholds, decide
from .portfolio import DEVELOPMENT_CUTOFF, PortfolioBar, PortfolioConfig, PortfolioResult, StatefulPortfolioSimulator, simulate_portfolio
from .robustness import EvidencePolicy, assert_development_only, purged_temporal_yearly_splits, development_candidate
from .risk_governor import RiskConfig, RiskGovernor
from .samples import SupervisedSample
from .specialists import run_specialists
from .types import MarketSnapshot

SCHEMA="brian.phase25-development.v1";SEED=20260902;HORIZON=6
START=datetime(2020,1,1,tzinfo=timezone.utc).timestamp();END=DEVELOPMENT_CUTOFF
FOLDS=(("2020-01-01","2023-01-01","2023-07-01","2024-01-01"),
       ("2020-01-01","2024-01-01","2024-07-01","2025-01-01"),
       ("2020-01-01","2025-01-01","2025-07-01","2026-01-01"))
COSTS={"LOW":dict(fee_bps=5.,assumed_spread_bps=1.,slippage_bps=.5),
       "BASE":dict(fee_bps=10.,assumed_spread_bps=2.,slippage_bps=1.),
       "STRESS":dict(fee_bps=15.,assumed_spread_bps=5.,slippage_bps=3.)}
FEATURE_GROUPS={
 "price_returns":("return_1","return_5"),"trend_ema":("ema_fast_ratio","ema_slow_ratio","ema_slope"),
 "momentum_rsi":("return_5","rsi"),"volatility":("atr_pct","zscore","bb_position"),
 "volume_breakout_acceleration":("volume_z","breakout","acceleration"),"regime":("regime_code","high_volatility"),
 "context_15m":("m15_return","m15_ema_ratio"),"context_1h":("h1_return","h1_ema_ratio")}
FEATURE_GROUPS["combined"]=tuple(dict.fromkeys(x for group in FEATURE_GROUPS.values() for x in group))
BUCKETS=((.50,.55),(.55,.60),(.60,.65),(.65,.70),(.70,.75),(.75,.80),(.80,1.01))

def ts(value:str)->float:return datetime.fromisoformat(value+"T00:00:00+00:00").timestamp()

def _load(root:Path,timeframe:str):
    pattern=root/"parquet"/"exchange=binance"/"market=spot"/"symbol=BTCUSDT"/f"timeframe={timeframe}"/"year=*"/"month=*"/"part.parquet"
    files=[]
    for name in sorted(glob.glob(str(pattern))):
        year=int(next(part.split("=")[1] for part in Path(name).parts if part.startswith("year=")))
        if year<=2025:files.append(name)
    if not files:raise FileNotFoundError(f"no development {timeframe} partitions")
    columns=["close_timestamp","open","high","low","close","volume"]
    table=pa.concat_tables([pq.ParquetFile(name).read(columns=columns) for name in files]);out={n:np.asarray(table[n].to_numpy(),dtype=float) for n in columns}
    if out["close_timestamp"].max()>=END:raise ValueError("2026 data is INVALID_CONTAMINATED and forbidden")
    return out

def build_features(root:Path):
    d=_load(root,"5m");t,o,h,l,c,v=(d[n] for n in ("close_timestamp","open","high","low","close","volume"))
    e9,e21=_ema(c,9),_ema(c,21);mean,sd=_roll(c,20),_roll(c,20,"std");ret=np.r_[np.nan,c[1:]/c[:-1]-1];ret5=np.r_[np.full(5,np.nan),c[5:]/c[:-5]-1]
    clean=np.nan_to_num(ret,nan=0);gain=_roll(np.maximum(clean,0),14);loss=_roll(np.maximum(-clean,0),14);rs=np.divide(gain,loss,out=np.zeros_like(gain),where=loss!=0);rsi=np.where(loss==0,np.where(gain>0,100.,50.),100-100/(1+rs))
    tr=np.maximum(h-l,np.maximum(np.abs(h-np.r_[c[0],c[:-1]]),np.abs(l-np.r_[c[0],c[:-1]])));atr=_roll(tr,14)/c
    recent=np.full(len(c),np.nan)
    for i in range(20,len(c)):recent[i]=np.max(h[i-20:i])
    f={"return_1":ret,"return_5":ret5,"ema_fast_ratio":e9/c-1,"ema_slow_ratio":e21/c-1,"ema_slope":np.r_[np.full(3,np.nan),e9[3:]/e9[:-3]-1],"rsi":rsi/100,"zscore":(c-mean)/sd,"bb_position":(c-(mean-2*sd))/(4*sd),"atr_pct":atr,"volume_z":(v-_roll(v,20))/_roll(v,20,"std"),"acceleration":ret-np.r_[np.nan,ret[:-1]],"breakout":(c-recent)/np.maximum(_roll(tr,14),1e-12)}
    for timeframe,prefix in (("15m","m15"),("1h","h1")):
        q=_load(root,timeframe);qc=q["close"];qr=np.r_[np.nan,qc[1:]/qc[:-1]-1];qe=_ema(qc,9);ix=np.searchsorted(q["close_timestamp"],t,side="right")-1;valid=ix>=0;a=np.full(len(t),np.nan);b=a.copy();a[valid]=qr[ix[valid]];b[valid]=qe[ix[valid]]/qc[ix[valid]]-1;f[prefix+"_return"]=a;f[prefix+"_ema_ratio"]=b
    f["regime_code"]=np.where((e9>e21)&(f["ema_slope"]>0),1,np.where((e9<e21)&(f["ema_slope"]<0),-1,0)).astype(float)
    past_atr=_roll(np.nan_to_num(atr,nan=0),288);f["high_volatility"]=(atr>past_atr).astype(float)
    assert_development_only(t,"feature construction")
    if t.min()<START:raise ValueError("development range violation")
    return t,o,h,l,c,v,f

def _samples(indices,labels,future,features,times,dataset_id,names):
    return tuple(SupervisedSample(float(times[i]),"BTCUSDT",tuple((n,float(features[n][i])) for n in names),int(labels[i]),float(future[i]),float(times[i+HORIZON]),dataset_id) for i in indices)

def _cap(indices,n):return indices if len(indices)<=n else indices[np.linspace(0,len(indices)-1,n,dtype=int)]

def _bars(indices,t,o,h,l,c):return tuple(PortfolioBar(float(t[i]),float(o[i]),float(h[i]),float(l[i]),float(c[i])) for i in indices)

def _config(cost_name):return PortfolioConfig(starting_equity=10_000,sizing_mode="equity_fraction",equity_fraction=.10,max_position_notional=2_000,max_equity_fraction=.20,stop_loss_pct=1,take_profit_pct=2,max_holding_bars=12,cooldown_bars=1,reversal_enabled=True,**COSTS[cost_name])

def _summary(result:PortfolioResult):
    pnls=[trade.net_pnl for trade in result.trades];wins=[value for value in pnls if value>0];losses=[value for value in pnls if value<0];gross_win=sum(wins);gross_loss=abs(sum(losses))
    values=asdict(result);values.pop("trades");values.pop("equity_curve");values.update({"wins":len(wins),"losses":len(losses),"win_rate":len(wins)/len(pnls) if pnls else 0.0,"expectancy":sum(pnls)/len(pnls) if pnls else 0.0,"profit_factor":gross_win/gross_loss if gross_loss>1e-12 else None,"profit_factor_is_infinite":gross_win>0 and gross_loss<=1e-12,"average_win":sum(wins)/len(wins) if wins else 0.0,"average_loss":sum(losses)/len(losses) if losses else 0.0});values["max_drawdown_pct"]=100*result.max_drawdown/result.starting_equity
    values["signal_to_entry_ratio"]=result.signals/result.entries if result.entries else None;return values

def _portfolio(indices,actions,t,o,h,l,c,cost="BASE"):
    assert_development_only((t[i] for i in indices),"portfolio replay")
    return simulate_portfolio(_bars(indices,t,o,h,l,c),tuple(actions),_config(cost))

def _portfolio_excluding_months(indices,actions,t,o,h,l,c,excluded_months,cost="BASE"):
    simulator=StatefulPortfolioSimulator(_config(cost));last_bar=None;inside_exclusion=False
    for i,action in zip(indices,actions):
        month=datetime.fromtimestamp(float(t[i]),timezone.utc).strftime("%Y-%m")
        if month in excluded_months:
            if not inside_exclusion and last_bar is not None:simulator.break_segment(last_bar)
            inside_exclusion=True;continue
        current=PortfolioBar(float(t[i]),float(o[i]),float(h[i]),float(l[i]),float(c[i]));simulator.step(current,action);last_bar=current;inside_exclusion=False
    if last_bar is None:raise ValueError("quality exclusion removed all observations")
    return simulator.finish(last_bar)

class _StaticMemory:
    def specialist_weight(self,name,default=1.0):return default

def static_brian_actions(indices,t,c,f):
    meta=MetaTrader(_StaticMemory());risk=RiskGovernor(RiskConfig(max_open_positions=1));actions=[];conf=[]
    regimes={-1:"DOWNTREND",0:"SIDEWAYS",1:"UPTREND"}
    for i in indices:
        features={"ema_fast":c[i]*(1+f["ema_fast_ratio"][i]),"ema_slow":c[i]*(1+f["ema_slow_ratio"][i]),"ema_slope_pct":f["ema_slope"][i]*100,"rsi":f["rsi"][i]*100,"return_5":f["return_5"][i]*100,"zscore":f["zscore"][i],"bb_position":f["bb_position"][i],"atr_pct":f["atr_pct"][i]*100,"breakout_score":f["breakout"][i],"volume_z":f["volume_z"][i],"acceleration":f["acceleration"][i]*100}
        snap=MarketSnapshot("BTCUSDT",float(c[i]),float(t[i]),"5m",regimes[int(f["regime_code"][i])],features,{"historical_mode":"STATIC_BRIAN_META","order_book":"unavailable"});decision=meta.decide(snap,run_specialists(snap));allowed,_,_=risk.review(decision,snap,{"open_positions":0,"drawdown_pct":0,"daily_pnl_pct":0});actions.append(decision.action if allowed else "WAIT");conf.append(decision.confidence)
    return actions,conf

def threshold_curve(probabilities,labels,indices,t,o,h,l,c):
    rows=[]
    for threshold in (.50,.55,.60,.65,.70):
        policy=PolicyThresholds(threshold,threshold,.10);actions=[decide(p,policy) for p in probabilities];result=_portfolio(indices,actions,t,o,h,l,c);coverage=sum(a!="WAIT" for a in actions)/len(actions)
        rows.append({"threshold":threshold,"coverage":coverage,"acted_hit_rate":asdict(evaluate_predictions(probabilities,labels,policy))["acted_hit_rate"],"portfolio":_summary(result)})
    return rows

def select_validation_threshold(curve):
    eligible=[r for r in curve if r["coverage"]>=.02 and r["portfolio"]["entries"]>=40]
    pool=eligible or curve
    return max(pool,key=lambda r:(r["portfolio"]["net_pnl"]-r["portfolio"]["max_drawdown"],r["coverage"]))["threshold"]

def calibration_buckets(probabilities,labels,actions,result,indices,times):
    entry_pnl={trade.entry_timestamp:trade.net_pnl for trade in result.trades};rows=[]
    for side in ("BUY","SELL"):
        for lo,hi in BUCKETS:
            selected=[]
            for i,(p,a) in enumerate(zip(probabilities,actions)):
                confidence=p.up if side=="BUY" else p.down
                if a==side and lo<=confidence<hi:selected.append((i,confidence))
            correct=sum((side=="BUY" and labels[i]==1) or (side=="SELL" and labels[i]==-1) for i,_ in selected);entry_values=[entry_pnl[float(times[indices[i]])] for i,_ in selected if float(times[indices[i]]) in entry_pnl]
            rows.append({"side":side,"bucket":f"{lo:.2f}-{min(hi,1):.2f}","predictions":len(selected),"empirical_accuracy":correct/len(selected) if selected else None,"mean_confidence":sum(x[1] for x in selected)/len(selected) if selected else None,"confidence_minus_accuracy":(sum(x[1] for x in selected)/len(selected)-correct/len(selected)) if selected else None,"entries":len(entry_values),"entry_expectancy":sum(entry_values)/len(entry_values) if entry_values else None,"status":"SUFFICIENT" if len(selected)>=100 and len(entry_values)>=40 else "INSUFFICIENT_SAMPLE"})
    return rows

def run(root:Path,dataset_id:str):
    started=time.time();t,o,h,l,c,v,f=build_features(root);valid=np.all(np.column_stack([np.isfinite(f[n]) for n in FEATURE_GROUPS["combined"]]),axis=1);future=np.full(len(c),np.nan);future[:-HORIZON]=(c[HORIZON:]/c[:-HORIZON]-1)*100
    quality_manifest=json.loads((root/"dataset_manifests"/f"{dataset_id}.json").read_text());affected=[b for b in quality_manifest["monthly_builds"] if b["anomaly_classification"]!="NONE"]
    policy=EvidencePolicy();groups=[datetime.fromtimestamp(float(x),timezone.utc).year-2020 for x in t if x<END];yearly_robustness=purged_temporal_yearly_splits(t.tolist(),groups,test_group_count=1,purge_seconds=3600,embargo_seconds=3600)
    prereg={"schema":SCHEMA,"dataset_id":dataset_id,"range":[START,END],"holdout":{"status":"INVALID_CONTAMINATED","evaluation_allowed":False},"folds":FOLDS,"horizon_minutes":30,"feature_groups":FEATURE_GROUPS,"costs":COSTS,"evidence_policy":asdict(policy),"threshold_candidates":[.50,.55,.60,.65,.70],"robustness_method":"PURGED_TEMPORAL_YEARLY_ROBUSTNESS","purged_temporal_yearly_splits":[{"split_id":s.split_id,"train_groups":s.train_groups,"test_groups":s.test_groups,"purge_seconds":s.purge_seconds,"embargo_seconds":s.embargo_seconds} for s in yearly_robustness],"declaration":"NO PRISTINE FINAL HOLDOUT EVALUATED"}
    prereg_id=canonical_hash(prereg);directory=root/"phase25";directory.mkdir(parents=True,exist_ok=True);(directory/f"preregistered-{prereg_id}.json").write_text(json.dumps({"preregistration_id":prereg_id,**prereg},sort_keys=True,separators=(",",":"))+"\n")
    results=[];all_predictions={};all_labels={};all_actions={};all_portfolios={};all_indices={};attempted=[]
    for fold,(start,tr_end,val_end,test_end) in enumerate(FOLDS,1):
        train=np.where(valid&(t>=ts(start))&(t<ts(tr_end)-3600)&(np.arange(len(t))+HORIZON<len(t)))[0];val=np.where(valid&(t>=ts(tr_end)+3600)&(t<ts(val_end)-3600)&(np.arange(len(t))+HORIZON<len(t)))[0];test=np.where(valid&(t>=ts(val_end)+3600)&(t<ts(test_end))&(np.arange(len(t))+HORIZON<len(t)))[0]
        if max(t[test],default=0)>=END:raise ValueError("2026 development test forbidden")
        neutral=float(np.quantile(np.abs(future[train]),.33));labels=np.where(future>neutral,1,np.where(future<-neutral,-1,0));trainfit=_cap(train,60000);valfit=_cap(val,30000)
        for cls,name,hp in ((LogisticRegressionBaseline,"logistic_regression",{}),(GradientBoostingBaseline,"gradient_boosting",{"n_estimators":40,"max_depth":2,"learning_rate":.05,"subsample":.8})):
            attempted.append({"fold":fold,"model":name,"hyperparameters":hp,"status":"attempted"});model=cls(metadata_for(name,dataset_id,"7a3069b",SCHEMA,fold,hp),random_state=SEED);fit0=time.time();names=FEATURE_GROUPS["combined"];model.fit(_samples(trainfit,labels,future,f,t,dataset_id,names));model.calibrate(_samples(valfit,labels,future,f,t,dataset_id,names));vp=model.predict_probability(_samples(valfit,labels,future,f,t,dataset_id,names));curve=threshold_curve(vp,labels[valfit].tolist(),valfit,t,o,h,l,c);selected=select_validation_threshold(curve);thresholds=PolicyThresholds(selected,selected,.10);tp=model.predict_probability(_samples(test,labels,future,f,t,dataset_id,names));actions=[decide(p,thresholds) for p in tp];base=_portfolio(test,actions,t,o,h,l,c);cost={k:_summary(_portfolio(test,actions,t,o,h,l,c,k)) for k in COSTS}
            key=f"{name}:fold{fold}";all_predictions[key]=tp;all_labels[key]=labels[test].tolist();all_actions[key]=actions;all_portfolios[key]=base;all_indices[key]=test
            results.append({"fold":fold,"model":name,"neutral_threshold_pct":neutral,"selected_threshold":selected,"threshold_curve":curve,"prediction":asdict(evaluate_predictions(tp,labels[test].tolist(),thresholds)),"portfolio":_summary(base),"cost_sensitivity":cost,"training_seconds":time.time()-fit0})
        baseline_actions={"wait":["WAIT"]*len(test),"always_long":["BUY"]*len(test),"legacy":np.where(f["ema_fast_ratio"][test]>f["ema_slow_ratio"][test],"BUY","SELL").tolist()};baseline_actions["STATIC_BRIAN_META"],_=static_brian_actions(test,t,c,f)
        for name,actions in baseline_actions.items():results.append({"fold":fold,"model":name,"portfolio":_summary(_portfolio(test,actions,t,o,h,l,c)),"cost_sensitivity":{k:_summary(_portfolio(test,actions,t,o,h,l,c,k)) for k in COSTS}})
    # Development-only ablations use the same locked folds and logistic configuration; no candidate is selected from test values.
    ablations=[]
    for group,names in FEATURE_GROUPS.items():
        fold_metrics=[]
        for fold,(start,tr_end,val_end,test_end) in enumerate(FOLDS,1):
            attempted.append({"fold":fold,"model":"logistic_regression","hyperparameters":{"ablation":group},"status":"attempted"});group_valid=np.all(np.column_stack([np.isfinite(f[n]) for n in names]),axis=1);target_available=np.arange(len(t))+HORIZON<len(t)
            train=np.where(group_valid&target_available&(t>=ts(start))&(t<ts(tr_end)-3600))[0];val=np.where(group_valid&target_available&(t>=ts(tr_end)+3600)&(t<ts(val_end)-3600))[0];test=np.where(group_valid&target_available&(t>=ts(val_end)+3600)&(t<ts(test_end)))[0]
            neutral=float(np.quantile(np.abs(future[train][np.isfinite(future[train])]),.33));labels=np.where(future>neutral,1,np.where(future<-neutral,-1,0));model=LogisticRegressionBaseline(metadata_for("logistic_regression",dataset_id,"7a3069b",SCHEMA,fold,{"ablation":group}),random_state=SEED);model.fit(_samples(_cap(train,60000),labels,future,f,t,dataset_id,names));model.calibrate(_samples(_cap(val,30000),labels,future,f,t,dataset_id,names));vp=model.predict_probability(_samples(_cap(val,30000),labels,future,f,t,dataset_id,names));curve=threshold_curve(vp,labels[_cap(val,30000)].tolist(),_cap(val,30000),t,o,h,l,c);threshold=select_validation_threshold(curve);tp=model.predict_probability(_samples(test,labels,future,f,t,dataset_id,names));actions=[decide(p,PolicyThresholds(threshold,threshold,.1)) for p in tp];port=_portfolio(test,actions,t,o,h,l,c);pred=evaluate_predictions(tp,labels[test].tolist(),PolicyThresholds(threshold,threshold,.1));fold_metrics.append({"fold":fold,"balanced_accuracy":pred.balanced_accuracy,"calibration_error":pred.calibration_error,"coverage":pred.coverage,"expectancy":port.net_pnl/port.entries if port.entries else 0,"max_drawdown":port.max_drawdown,"entries":port.entries})
        ablations.append({"feature_group":group,"folds":fold_metrics})
    combined_by_fold={row["fold"]:row for item in ablations if item["feature_group"]=="combined" for row in item["folds"]}
    for item in ablations:
        for row in item["folds"]:
            base=combined_by_fold[row["fold"]];row["predictive_delta_vs_combined"]=row["balanced_accuracy"]-base["balanced_accuracy"];row["calibration_delta_vs_combined"]=row["calibration_error"]-base["calibration_error"];row["coverage_delta_vs_combined"]=row["coverage"]-base["coverage"];row["expectancy_delta_vs_combined"]=row["expectancy"]-base["expectancy"];row["drawdown_delta_vs_combined"]=row["max_drawdown"]-base["max_drawdown"]
    calibration=[]
    for key,probs in all_predictions.items():calibration.extend({"model_fold":key,**row} for row in calibration_buckets(probs,all_labels[key],all_actions[key],all_portfolios[key],all_indices[key],t))
    regime=[]
    for key,port in all_portfolios.items():
        indices=all_indices[key];actions=all_actions[key];lookup={float(t[i]):int(f["regime_code"][i]) for i in indices};by={-1:[],0:[],1:[]}
        for trade in port.trades:by[lookup.get(trade.entry_timestamp,0)].append(trade.net_pnl)
        for code,pnls in by.items():
            positions=[j for j,i in enumerate(indices) if int(f["regime_code"][i])==code];signals=sum(actions[j]!="WAIT" for j in positions);waits=len(positions)-signals;gross_win=sum(max(0,x) for x in pnls);gross_loss=abs(sum(min(0,x) for x in pnls))
            regime.append({"model_fold":key,"regime":{-1:"downtrend",0:"sideways",1:"uptrend"}[code],"observations":len(positions),"signals":signals,"entries":len(pnls),"expectancy":sum(pnls)/len(pnls) if pnls else 0,"profit_factor":gross_win/max(gross_loss,1e-12),"drawdown_contribution":abs(sum(min(0,x) for x in pnls)),"wait_rate":waits/len(positions) if positions else 0,"status":"SUFFICIENT" if len(pnls)>=40 else "INSUFFICIENT_SAMPLE"})
        for high_vol,label in ((0,"lower_volatility"),(1,"high_volatility")):
            positions=[j for j,i in enumerate(indices) if int(f["high_volatility"][i])==high_vol];signals=sum(actions[j]!="WAIT" for j in positions);waits=len(positions)-signals;entry_times={float(t[indices[j]]) for j in positions};pnls=[trade.net_pnl for trade in port.trades if trade.entry_timestamp in entry_times];gross_win=sum(max(0,x) for x in pnls);gross_loss=abs(sum(min(0,x) for x in pnls))
            regime.append({"model_fold":key,"regime":label,"observations":len(positions),"signals":signals,"entries":len(pnls),"expectancy":sum(pnls)/len(pnls) if pnls else 0,"profit_factor":gross_win/max(gross_loss,1e-12),"drawdown_contribution":abs(sum(min(0,x) for x in pnls)),"wait_rate":waits/len(positions) if positions else 0,"status":"SUFFICIENT" if len(pnls)>=40 else "INSUFFICIENT_SAMPLE"})
    equity_artifacts={}
    curve_dir=directory/"equity_curves";curve_dir.mkdir(parents=True,exist_ok=True)
    for key,portfolio in all_portfolios.items():
        path=curve_dir/f"{key.replace(':','-')}.parquet";pq.write_table(pa.table({"timestamp":[x.timestamp for x in portfolio.equity_curve],"cash":[x.cash for x in portfolio.equity_curve],"equity":[x.equity for x in portfolio.equity_curve],"realized_pnl":[x.realized_pnl for x in portfolio.equity_curve],"unrealized_pnl":[x.unrealized_pnl for x in portfolio.equity_curve],"drawdown":[x.drawdown for x in portfolio.equity_curve],"state":[x.state for x in portfolio.equity_curve]}),path,compression="zstd");equity_artifacts[key]={"path":str(path),"sha256":__import__("hashlib").sha256(path.read_bytes()).hexdigest(),"rows":len(portfolio.equity_curve)}
    excluded_months={b["period"] for b in affected};month_keys=np.asarray([datetime.fromtimestamp(float(x),timezone.utc).strftime("%Y-%m") for x in t]);clean_month=~np.isin(month_keys,list(excluded_months));quality_sensitivity=[]
    for model_name,model_class,hp in (("logistic_regression",LogisticRegressionBaseline,{}),("gradient_boosting",GradientBoostingBaseline,{"n_estimators":40,"max_depth":2,"learning_rate":.05,"subsample":.8})):
        for fold,(start,tr_end,val_end,test_end) in enumerate(FOLDS,1):
            train=np.where(valid&clean_month&(t>=ts(start))&(t<ts(tr_end)-3600)&(np.arange(len(t))+HORIZON<len(t)))[0];val=np.where(valid&clean_month&(t>=ts(tr_end)+3600)&(t<ts(val_end)-3600)&(np.arange(len(t))+HORIZON<len(t)))[0];test=np.where(valid&clean_month&(t>=ts(val_end)+3600)&(t<ts(test_end))&(np.arange(len(t))+HORIZON<len(t)))[0];neutral=float(np.quantile(np.abs(future[train]),.33));labels=np.where(future>neutral,1,np.where(future<-neutral,-1,0));attempted.append({"fold":fold,"model":model_name,"hyperparameters":hp,"quality_exclusion":sorted(excluded_months),"status":"attempted"});model=model_class(metadata_for(model_name,dataset_id,"7a3069b",SCHEMA,200+fold,{**hp,"quality_exclusion":True}),random_state=SEED);model.fit(_samples(_cap(train,60000),labels,future,f,t,dataset_id,FEATURE_GROUPS["combined"]));model.calibrate(_samples(_cap(val,30000),labels,future,f,t,dataset_id,FEATURE_GROUPS["combined"]));vp=model.predict_probability(_samples(_cap(val,30000),labels,future,f,t,dataset_id,FEATURE_GROUPS["combined"]));curve=threshold_curve(vp,labels[_cap(val,30000)].tolist(),_cap(val,30000),t,o,h,l,c);threshold=select_validation_threshold(curve);tp=model.predict_probability(_samples(test,labels,future,f,t,dataset_id,FEATURE_GROUPS["combined"]));actions=[decide(p,PolicyThresholds(threshold,threshold,.1)) for p in tp];excluded=_portfolio_excluding_months(test,actions,t,o,h,l,c,excluded_months);full=all_portfolios[f"{model_name}:fold{fold}"]
            quality_sensitivity.append({"model_fold":f"{model_name}:fold{fold}","full":_summary(full),"exclude_affected_months_retrained":_summary(excluded),"selected_threshold":threshold,"net_pnl_delta":excluded.net_pnl-full.net_pnl,"entry_delta":excluded.entries-full.entries})
    yearly_robustness_metrics=[]
    combined=FEATURE_GROUPS["combined"]
    for split in yearly_robustness:
        train=np.asarray([i for i in split.train_indices if valid[i] and i+HORIZON<len(t)],dtype=int);test=np.asarray([i for i in split.test_indices if valid[i] and i+HORIZON<len(t)],dtype=int)
        ordered=train[np.argsort(t[train])];cut=max(1,int(len(ordered)*.8));fit=_cap(ordered[:cut],60000);validation=_cap(ordered[cut:],30000);neutral=float(np.quantile(np.abs(future[fit]),.33));labels=np.where(future>neutral,1,np.where(future<-neutral,-1,0));model=LogisticRegressionBaseline(metadata_for("logistic_regression",dataset_id,"7a3069b",SCHEMA,100+split.test_groups[0],{"yearly_robustness_split_id":split.split_id}),random_state=SEED);model.fit(_samples(fit,labels,future,f,t,dataset_id,combined));model.calibrate(_samples(validation,labels,future,f,t,dataset_id,combined));vp=model.predict_probability(_samples(validation,labels,future,f,t,dataset_id,combined));curve=threshold_curve(vp,labels[validation].tolist(),validation,t,o,h,l,c);threshold=select_validation_threshold(curve);tp=model.predict_probability(_samples(test,labels,future,f,t,dataset_id,combined));actions=[decide(p,PolicyThresholds(threshold,threshold,.1)) for p in tp];portfolio=_portfolio(test,actions,t,o,h,l,c);prediction=evaluate_predictions(tp,labels[test].tolist(),PolicyThresholds(threshold,threshold,.1));yearly_robustness_metrics.append({"split_id":split.split_id,"test_groups":split.test_groups,"threshold":threshold,"balanced_accuracy":prediction.balanced_accuracy,"calibration_error":prediction.calibration_error,"coverage":prediction.coverage,"portfolio":_summary(portfolio)})
    candidate={}
    for model in ("logistic_regression","gradient_boosting"):
        rows=[r for r in results if r["model"]==model];fold_metrics=[]
        for row in rows:
            portfolio=all_portfolios[f"{model}:fold{row['fold']}"];gross_win=sum(max(0,trade.net_pnl) for trade in portfolio.trades);gross_loss=abs(sum(min(0,trade.net_pnl) for trade in portfolio.trades))
            fold_metrics.append({"trades":row["portfolio"]["entries"],"expectancy":row["portfolio"]["net_pnl"]/row["portfolio"]["entries"] if row["portfolio"]["entries"] else 0,"profit_factor":gross_win/max(1e-12,gross_loss),"max_drawdown_pct":row["portfolio"]["max_drawdown_pct"]})
        coverage=sum(row["prediction"]["coverage"] for row in rows)/len(rows);stress=sum(row["cost_sensitivity"]["STRESS"]["net_pnl"] for row in rows)
        samples=sum(len(all_labels[f"{model}:fold{row['fold']}"]) for row in rows);candidate[model]=development_candidate(fold_metrics,coverage=coverage,calibration_samples=samples,stress_net_pnl=stress,policy=policy)
    catalog_by_dataset={}
    for path in (root/"catalog").glob("*.json"):
        item=json.loads(path.read_text());catalog_by_dataset[item["dataset_id"]]=item
    suppressed={"5m":0,"15m":0,"1h":0}
    for build in affected:
        partitions=dict(build["partitions"])
        year,month=map(int,build["period"].split("-"));expected_minutes=calendar.monthrange(year,month)[1]*24*60
        for timeframe,size in (("5m",5),("15m",15),("1h",60)):
            actual=catalog_by_dataset[partitions[timeframe]]["rows"];suppressed[timeframe]+=max(0,expected_minutes//size-actual)
    quality={"dataset_quality":quality_manifest["quality_status"],"known_missing_minutes":sum(b["missing_intervals"] for b in affected),"affected_months":[b["period"] for b in affected],"source_gap_evidence_ids":[b["anomaly_evidence_id"] for b in affected],"derived_bars_suppressed":suppressed,"derived_suppression_policy":"incomplete constituent groups omitted","exclusion_policy":"remove affected months from train/validation/test, refit preprocessing/model/calibration/thresholds, force flat at excluded boundaries","sensitivity":quality_sensitivity}
    manifest={"schema_version":SCHEMA,"experiment_id":canonical_hash({"preregistration":prereg_id,"code":"7a3069b","created":int(started)}),"preregistration_id":prereg_id,"source_dataset_id":dataset_id,"code_base_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"date_range":{"start":START,"end_exclusive":END,"max_observed_timestamp":float(t.max())},"portfolio_simulator_version":"brian.stateful-portfolio.v1","account_rules":asdict(_config("BASE")),"costs":COSTS,"folds":FOLDS,"robustness_method":"PURGED_TEMPORAL_YEARLY_ROBUSTNESS","purged_temporal_yearly_splits":[{"split_id":s.split_id,"train_groups":s.train_groups,"test_groups":s.test_groups,"train_count":len(s.train_indices),"test_count":len(s.test_indices)} for s in yearly_robustness],"purged_temporal_yearly_robustness_metrics":yearly_robustness_metrics,"evidence_policy":asdict(policy),"attempted_models":attempted,"results":results,"calibration_buckets":calibration,"feature_ablations":ablations,"regime_metrics":regime,"equity_curve_artifacts":equity_artifacts,"quality":quality,"candidate_decisions":candidate,"brian_baseline":"STATIC_BRIAN_META: actual specialists + MetaTrader + confidence/volatility risk review with immutable default weights; portfolio state enforces position risk; no adaptive memory or future reliability","holdout":{"status":"INVALID_CONTAMINATED","evaluation_allowed":False,"results":None},"declaration":"NO PRISTINE FINAL HOLDOUT EVALUATED","runtime_seconds":time.time()-started}
    target=directory/f"{manifest['experiment_id']}.json";target.write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n");print(json.dumps({"experiment_id":manifest["experiment_id"],"manifest":str(target),"max_timestamp":float(t.max()),"runtime_seconds":manifest["runtime_seconds"]}));return manifest

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--root",default="research_data");p.add_argument("--dataset-id",required=True);a=p.parse_args(argv);run(Path(a.root),a.dataset_id)
if __name__=="__main__":main()
