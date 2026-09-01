from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse, glob, json, time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .data import canonical_hash
from .evaluation import evaluate_predictions
from .learning import GradientBoostingBaseline, LogisticRegressionBaseline, metadata_for
from .metrics import calculate
from .policy import PolicyThresholds, decide, select_thresholds
from .replay import ReplayPoint, ReplaySettings, replay
from .samples import SupervisedSample

SCHEMA="brian.phase24-decision.v1"; SEED=20260902
FOLDS=(("2020-01-01","2023-01-01","2023-07-01","2024-01-01"),
       ("2020-01-01","2024-01-01","2024-07-01","2025-01-01"),
       ("2020-01-01","2025-01-01","2025-07-01","2026-01-01"))
HOLDOUT=("2026-01-01","2026-08-01")
HOLDOUT_STATUS={
    "status":"INVALID_CONTAMINATED",
    "reusable_as_pristine_holdout":False,
    "evaluation_allowed":False,
    "reason":"Phase 2.4 boundary bug accessed this period; prior results are invalid and must not be recovered or reported",
}
HORIZONS=(3,6,12)
COSTS={"LOW":{"taker_fee_bps":5.0,"spread_bps":1.0,"slippage_bps":0.5},
       "BASE":{"taker_fee_bps":10.0,"spread_bps":2.0,"slippage_bps":1.0},
       "STRESS":{"taker_fee_bps":15.0,"spread_bps":5.0,"slippage_bps":3.0}}

def ts(value): return datetime.fromisoformat(value+"T00:00:00+00:00").timestamp()

def _load(root:Path,tf:str):
    files=sorted(glob.glob(str(root/"parquet"/"exchange=binance"/"market=spot"/"symbol=BTCUSDT"/f"timeframe={tf}"/"year=*"/"month=*"/"part.parquet")))
    tables=[pq.ParquetFile(f).read(columns=["close_timestamp","open","high","low","close","volume"]) for f in files]
    t=pa.concat_tables(tables); return {n:np.asarray(t[n].to_numpy(),dtype=float) for n in t.column_names}

def _ema(x,n):
    out=np.empty_like(x);out[0]=x[0];a=2/(n+1)
    for i in range(1,len(x)):out[i]=out[i-1]+a*(x[i]-out[i-1])
    return out

def _roll(x,n,kind="mean"):
    out=np.full(len(x),np.nan); cs=np.r_[0.,np.cumsum(x)]
    out[n-1:]=(cs[n:]-cs[:-n])/n
    if kind=="std":
        cs2=np.r_[0.,np.cumsum(x*x)]; mean=out[n-1:];out[n-1:]=np.sqrt(np.maximum(0,(cs2[n:]-cs2[:-n])/n-mean*mean))
    return out

def build_decision(root:Path,dataset_id:str):
    d=_load(root,"5m"); t,c,h,l,v=d["close_timestamp"],d["close"],d["high"],d["low"],d["volume"]
    e9,e21=_ema(c,9),_ema(c,21); mean,sd=_roll(c,20),_roll(c,20,"std")
    ret=np.r_[np.nan,c[1:]/c[:-1]-1]; ret5=np.r_[np.full(5,np.nan),c[5:]/c[:-5]-1]
    clean_ret=np.nan_to_num(ret,nan=0.0);gain=_roll(np.maximum(clean_ret,0),14);loss=_roll(np.maximum(-clean_ret,0),14)
    rs=np.divide(gain,loss,out=np.zeros_like(gain),where=loss!=0)
    rsi=np.where(loss==0,np.where(gain>0,100.0,50.0),100-100/(1+rs))
    tr=np.maximum(h-l,np.maximum(np.abs(h-np.r_[c[0],c[:-1]]),np.abs(l-np.r_[c[0],c[:-1]])));atr=_roll(tr,14)/c
    vz=(v-_roll(v,20))/_roll(v,20,"std"); accel=ret-np.r_[np.nan,ret[:-1]]
    recent=np.full(len(c),np.nan)
    for i in range(20,len(c)): recent[i]=np.max(h[i-20:i])
    breakout=(c-recent)/np.maximum(_roll(tr,14),1e-12)
    features={"return_1":ret,"return_5":ret5,"ema_fast_ratio":e9/c-1,"ema_slow_ratio":e21/c-1,
              "ema_slope":np.r_[np.full(3,np.nan),e9[3:]/e9[:-3]-1],"rsi":rsi/100,
              "zscore":(c-mean)/sd,"bb_position":(c-(mean-2*sd))/(4*sd),"atr_pct":atr,
              "volume_z":vz,"acceleration":accel,"breakout":breakout}
    for tf,prefix in (("15m","m15"),("1h","h1")):
        q=_load(root,tf); qc=q["close"]; qret=np.r_[np.nan,qc[1:]/qc[:-1]-1]; qe=_ema(qc,9)
        ix=np.searchsorted(q["close_timestamp"],t,side="right")-1; valid=ix>=0
        a=np.full(len(t),np.nan);b=a.copy();a[valid]=qret[ix[valid]];b[valid]=qe[ix[valid]]/qc[ix[valid]]-1
        features[prefix+"_return"]=a;features[prefix+"_ema_ratio"]=b
    regime=np.where((e9>e21)&(features["ema_slope"]>0),1,np.where((e9<e21)&(features["ema_slope"]<0),-1,0)).astype(float)
    features["regime_code"]=regime
    table={"timestamp":t,"price":c,"high":h,"low":l,"volume":v,**features,
           "feature_schema_version":[SCHEMA]*len(t),"dataset_id":[dataset_id]*len(t),
           "order_book_available":[False]*len(t),"legacy_context_available":[False]*len(t)}
    target=root/"decision"/f"{canonical_hash({'dataset':dataset_id,'schema':SCHEMA})}.parquet";target.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.table(table),target,compression="zstd");return t,c,h,l,features,target

def _samples(idx,labels,future,features,times,dataset_id):
    names=tuple(features);return tuple(SupervisedSample(float(times[i]),"BTCUSDT",tuple((n,float(features[n][i])) for n in names),int(labels[i]),float(future[i]),float(times[i]),dataset_id) for i in idx)

def _cap(idx,n):
    return idx if len(idx)<=n else idx[np.linspace(0,len(idx)-1,n,dtype=int)]

def _trade(actions,idx,horizon,times,close,high,low,cost):
    spread=cost["spread_bps"]/10000; settings=ReplaySettings(position_size=1000,tp_pct=1000,sl_pct=1000,taker_fee_bps=cost["taker_fee_bps"],slippage_bps=cost["slippage_bps"])
    results=[]
    for a,i in zip(actions,idx):
        j=i+horizon; p0=ReplayPoint(times[i],close[i]*(1-spread/2),close[i]*(1+spread/2),high[i],low[i],close[i]);p1=ReplayPoint(times[j],close[j]*(1-spread/2),close[j]*(1+spread/2),high[j],low[j],close[j])
        results.append(replay((p0,p1),{"BUY":"LONG","SELL":"SHORT","WAIT":"WAIT"}[a],settings))
    fills=[r for r in results if r.status=="FILLED"];m=calculate([r.net_pnl for r in fills],starting_equity=10000,exposure=sum(r.exposure_seconds for r in fills),decisions=len(actions),waits=actions.count("WAIT"))
    return {**asdict(m),"turnover":sum((r.entry_price or 0)*r.filled_quantity*2 for r in fills),"cost_burden":sum(r.fees+r.funding for r in fills)}

def run(root:Path,dataset_id:str):
    started=time.time();times,close,high,low,features,decision_path=build_decision(root,dataset_id)
    valid=np.all(np.column_stack([np.isfinite(x) for x in features.values()]),axis=1)
    lock={"dataset_id":dataset_id,"schema":SCHEMA,"folds":FOLDS,"holdout":HOLDOUT,
          "holdout_status":HOLDOUT_STATUS,"purge_seconds":3600,"embargo_seconds":3600,
          "horizons_minutes":[15,30,60],"costs":COSTS,"seed":SEED,"training_cap":60000}
    lock_id=canonical_hash(lock); lp=root/"experiments"/f"locked-{lock_id}.json";lp.parent.mkdir(parents=True,exist_ok=True);lp.write_text(json.dumps({"lock_id":lock_id,**lock},sort_keys=True,separators=(",",":"))+"\n")
    results=[]; frozen={}
    for horizon in HORIZONS:
        future=np.full(len(close),np.nan);future[:-horizon]=(close[horizon:]/close[:-horizon]-1)*100
        for fi,(start,tr_end,val_end,test_end) in enumerate(FOLDS,1):
            train=np.where(valid&(times>=ts(start))&(times<ts(tr_end)-3600))[0];val=np.where(valid&(times>=ts(tr_end)+3600)&(times<ts(val_end)-3600))[0];test=np.where(valid&(times>=ts(val_end)+3600)&(times<ts(test_end)))[0]
            threshold=float(np.quantile(np.abs(future[train][np.isfinite(future[train])]),.33)); labels=np.where(future>threshold,1,np.where(future<-threshold,-1,0))
            train=_cap(train,60000);valfit=_cap(val,30000)
            for cls,name,hp in ((LogisticRegressionBaseline,"logistic_regression",{}),(GradientBoostingBaseline,"gradient_boosting",{"n_estimators":40,"max_depth":2,"learning_rate":.05,"subsample":.8})):
                model=cls(metadata_for(name,dataset_id,"brian-2026",SCHEMA,fi,hp),random_state=SEED);t0=time.time();model.fit(_samples(train,labels,future,features,times,dataset_id));model.calibrate(_samples(valfit,labels,future,features,times,dataset_id));vp=model.predict_probability(_samples(valfit,labels,future,features,times,dataset_id));th=select_thresholds(vp,labels[valfit].tolist(),partition="validation");tp=model.predict_probability(_samples(test,labels,future,features,times,dataset_id));actions=[decide(p,th) for p in tp]
                results.append({"horizon_minutes":horizon*5,"fold":fi,"model":name,"neutral_threshold_pct":threshold,"policy":asdict(th),"prediction":asdict(evaluate_predictions(tp,labels[test].tolist(),th)),"trading":_trade(actions,test,horizon,times,close,high,low,COSTS["BASE"]),"training_seconds":time.time()-t0})
                frozen[(horizon,name)]=(model,th,threshold)
            for name,actions in (("wait_only",["WAIT"]*len(test)),("always_long",["BUY"]*len(test)),("legacy_rule",np.where(features["ema_fast_ratio"][test]>features["ema_slow_ratio"][test],"BUY","SELL").tolist()),("brian_meta",np.where(np.abs(features["breakout"][test])>.5,np.where(features["breakout"][test]>0,"BUY","SELL"),"WAIT").tolist())):
                results.append({"horizon_minutes":horizon*5,"fold":fi,"model":name,"trading":_trade(actions,test,horizon,times,close,high,low,COSTS["BASE"])})
    frozen_id=canonical_hash({"lock_id":lock_id,"configs":[(h,n,asdict(v[1]),v[2]) for (h,n),v in frozen.items()]})
    manifest={"schema_version":"brian.phase24-experiment.v2","experiment_id":canonical_hash({"lock":lock_id,"frozen":frozen_id}),"dataset_id":dataset_id,"decision_dataset":str(decision_path),"lock_id":lock_id,"frozen_config_id":frozen_id,"holdout":{**HOLDOUT_STATUS,"range":HOLDOUT,"results":None},"results":results,"runtime_seconds":time.time()-started,"limitations":["spot OHLCV has no observed bid/ask or depth; costs are simulation assumptions","legacy point-in-time predictor context unavailable","brian_meta competitor uses deterministic reconstructable specialist proxy without memory learning","2026 holdout is contaminated and permanently blocked"]}
    out=root/"experiments"/f"{manifest['experiment_id']}.json";out.write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n");print(json.dumps({"experiment":str(out),"id":manifest["experiment_id"],"runtime":manifest["runtime_seconds"]}));return manifest

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--root",default="research_data");p.add_argument("--dataset-id",required=True);a=p.parse_args(argv);run(Path(a.root),a.dataset_id)
if __name__=="__main__":main()
