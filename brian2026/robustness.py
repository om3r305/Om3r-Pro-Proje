from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Sequence

from .data import canonical_hash
from .portfolio import DEVELOPMENT_CUTOFF


def assert_development_only(timestamps:Sequence[float],context:str)->None:
    if any(float(timestamp)>=DEVELOPMENT_CUTOFF for timestamp in timestamps):
        raise ValueError(f"2026 data is INVALID_CONTAMINATED and forbidden for {context}")


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    split_id: str
    train_groups: tuple[int,...]
    test_groups: tuple[int,...]
    train_indices: tuple[int,...]
    test_indices: tuple[int,...]
    purge_seconds: float
    embargo_seconds: float


def purged_temporal_yearly_splits(timestamps:Sequence[float],groups:Sequence[int],*,test_group_count:int=1,purge_seconds:float=3600,embargo_seconds:float=3600)->tuple[TemporalSplit,...]:
    """Build deterministic hold-one-year-out splits with purge and embargo.

    The configured Phase 2.5 use holds out one year per split. Its reported
    method label describes that configured evaluation rather than claiming a
    general combinatorial validation protocol.
    """
    if len(timestamps)!=len(groups) or not timestamps:raise ValueError("timestamps/groups must align")
    assert_development_only(timestamps,"purged temporal yearly robustness")
    unique=tuple(sorted(set(int(g) for g in groups)))
    if not 0<test_group_count<len(unique):raise ValueError("invalid yearly robustness group count")
    out=[]
    for selected in combinations(unique,test_group_count):
        test=tuple(i for i,g in enumerate(groups) if g in selected);lo=min(timestamps[i] for i in test);hi=max(timestamps[i] for i in test)
        train=tuple(i for i,g in enumerate(groups) if g not in selected and not (lo-purge_seconds<=timestamps[i]<=hi+embargo_seconds))
        payload={"test_groups":selected,"train_indices":train,"test_indices":test,"purge_seconds":purge_seconds,"embargo_seconds":embargo_seconds}
        out.append(TemporalSplit(canonical_hash(payload),tuple(g for g in unique if g not in selected),selected,train,test,purge_seconds,embargo_seconds))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    min_total_trades:int=200
    min_trades_per_fold:int=40
    min_coverage:float=0.02
    max_wait_rate:float=0.98
    min_positive_expectancy_folds:int=2
    min_profit_factor:float=1.10
    max_drawdown_pct:float=20.0
    min_calibration_samples:int=200
    require_stress_positive:bool=True


def development_candidate(folds:Sequence[dict],*,coverage:float,calibration_samples:int,stress_net_pnl:float,policy:EvidencePolicy=EvidencePolicy())->dict:
    reasons=[];trades=sum(int(f.get("trades",0)) for f in folds);positive=sum(float(f.get("expectancy",0))>0 for f in folds)
    if trades<policy.min_total_trades:reasons.append("minimum total trades not met")
    if any(int(f.get("trades",0))<policy.min_trades_per_fold for f in folds):reasons.append("minimum trades per fold not met")
    if coverage<policy.min_coverage or 1-coverage>policy.max_wait_rate:reasons.append("insufficient coverage")
    if positive<policy.min_positive_expectancy_folds:reasons.append("insufficient positive-expectancy folds")
    if any(float(f.get("profit_factor",0))<policy.min_profit_factor for f in folds):reasons.append("profit factor gate failed")
    if any(float(f.get("max_drawdown_pct",float("inf")))>policy.max_drawdown_pct for f in folds):reasons.append("drawdown gate failed")
    if calibration_samples<policy.min_calibration_samples:reasons.append("insufficient calibration samples")
    if policy.require_stress_positive and stress_net_pnl<=0:reasons.append("cost-stress survivability failed")
    return {"status":"DEVELOPMENT_CANDIDATE" if not reasons else "INSUFFICIENT_EVIDENCE","reasons":reasons,"policy":asdict(policy),"final_champion":False,"shadow_only":True}
