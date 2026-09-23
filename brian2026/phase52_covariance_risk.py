from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence
import math

import numpy as np

from .phase44_portfolio_brain import PortfolioBookPlan

PHASE52_SCHEMA_VERSION = "brian.phase52-covariance-risk.v1"
ShrinkAlpha = float | Literal["lw"]


@dataclass(frozen=True, slots=True)
class CovarianceRiskConfig:
    alpha: ShrinkAlpha = "lw"
    min_observations: int = 30
    max_period_volatility: float = 0.025
    nan_policy: Literal["fill_zero", "reject"] = "fill_zero"

    def __post_init__(self) -> None:
        if isinstance(self.alpha, str):
            if self.alpha != "lw":
                raise ValueError("alpha string must be 'lw'")
        else:
            if not math.isfinite(float(self.alpha)) or not 0.0 <= float(self.alpha) <= 1.0:
                raise ValueError("numeric alpha must be in [0,1]")
        if self.min_observations < 5:
            raise ValueError("min_observations must be at least 5")
        if not math.isfinite(self.max_period_volatility) or self.max_period_volatility <= 0:
            raise ValueError("max_period_volatility must be positive")


@dataclass(frozen=True, slots=True)
class CovarianceEstimate:
    assets: tuple[str, ...]
    observations: int
    sample_covariance: tuple[tuple[float, ...], ...]
    shrink_target: tuple[tuple[float, ...], ...]
    covariance: tuple[tuple[float, ...], ...]
    correlation: tuple[tuple[float, ...], ...]
    shrinkage_alpha: float
    schema_version: str = PHASE52_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _to_matrix(
    returns_by_asset: Mapping[str, Sequence[float]],
    assets: Sequence[str],
    *,
    nan_policy: str,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    lengths: set[int] = set()
    for asset in assets:
        if asset not in returns_by_asset:
            raise KeyError(f"missing return history for {asset}")
        values = np.asarray([float(value) for value in returns_by_asset[asset]], dtype=float)
        lengths.add(len(values))
        columns.append(values)
    if len(lengths) != 1:
        raise ValueError("all return histories must be aligned to equal length")
    if not columns or next(iter(lengths), 0) <= 0:
        raise ValueError("return history must not be empty")
    matrix = np.column_stack(columns)
    if nan_policy == "reject" and not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite return history rejected")
    if nan_policy == "fill_zero":
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


def _empirical_covariance(centered: np.ndarray) -> np.ndarray:
    if centered.ndim != 2 or centered.shape[0] == 0:
        raise ValueError("centered return matrix must be non-empty 2D")
    return np.asarray(centered.T.dot(centered) / centered.shape[0], dtype=float)


def _constant_variance_target(sample_covariance: np.ndarray) -> np.ndarray:
    n = sample_covariance.shape[0]
    target = np.eye(n, dtype=float)
    target *= float(np.mean(np.diag(sample_covariance)))
    return target


def _ledoit_wolf_const_var_alpha(
    centered: np.ndarray,
    sample_covariance: np.ndarray,
    target: np.ndarray,
) -> float:
    """Ledoit-Wolf constant-variance shrink parameter.

    Clean-room implementation of the estimator equation used by Qlib's
    ShrinkCovEstimator(const_var): alpha = clip((phi/gamma)/T, 0, 1).
    """
    t = centered.shape[0]
    y = centered ** 2
    phi = float(np.sum(y.T.dot(y) / t - sample_covariance ** 2))
    gamma = float(np.linalg.norm(sample_covariance - target, "fro") ** 2)
    if gamma <= 1e-18:
        return 1.0
    kappa = phi / gamma
    return float(max(0.0, min(1.0, kappa / t)))


def estimate_shrunk_covariance(
    returns_by_asset: Mapping[str, Sequence[float]],
    *,
    assets: Sequence[str] | None = None,
    config: CovarianceRiskConfig = CovarianceRiskConfig(),
) -> CovarianceEstimate:
    selected = tuple(assets) if assets is not None else tuple(sorted(returns_by_asset))
    if not selected:
        raise ValueError("covariance estimation requires assets")
    if len(set(selected)) != len(selected):
        raise ValueError("asset list must be unique")

    matrix = _to_matrix(returns_by_asset, selected, nan_policy=config.nan_policy)
    if matrix.shape[0] < config.min_observations:
        raise ValueError(
            f"insufficient covariance observations: {matrix.shape[0]} < {config.min_observations}"
        )

    centered = matrix - np.mean(matrix, axis=0)
    sample = _empirical_covariance(centered)
    target = _constant_variance_target(sample)
    alpha = (
        _ledoit_wolf_const_var_alpha(centered, sample, target)
        if config.alpha == "lw"
        else float(config.alpha)
    )
    covariance = (1.0 - alpha) * sample + alpha * target

    variances = np.clip(np.diag(covariance), 0.0, None)
    vola = np.sqrt(variances)
    denom = np.outer(vola, vola)
    correlation = np.zeros_like(covariance)
    valid = denom > 1e-18
    correlation[valid] = covariance[valid] / denom[valid]
    for index in range(len(selected)):
        correlation[index, index] = 1.0 if vola[index] > 0 else 0.0

    return CovarianceEstimate(
        assets=selected,
        observations=matrix.shape[0],
        sample_covariance=tuple(tuple(float(v) for v in row) for row in sample),
        shrink_target=tuple(tuple(float(v) for v in row) for row in target),
        covariance=tuple(tuple(float(v) for v in row) for row in covariance),
        correlation=tuple(tuple(float(v) for v in row) for row in correlation),
        shrinkage_alpha=alpha,
    )


@dataclass(frozen=True, slots=True)
class CovarianceRiskOverlay:
    original_weights: Mapping[str, float]
    scaled_weights: Mapping[str, float]
    portfolio_variance_before: float
    portfolio_volatility_before: float
    portfolio_variance_after: float
    portfolio_volatility_after: float
    volatility_scale: float
    released_weight_to_cash: float
    marginal_risk_contributions: Mapping[str, float]
    normalized_risk_contributions: Mapping[str, float]
    max_abs_pairwise_correlation: float
    covariance: CovarianceEstimate
    schema_version: str = PHASE52_SCHEMA_VERSION
    risk_only_shrinks: bool = True
    shadow_only: bool = True
    live_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "original_weights": dict(self.original_weights),
            "scaled_weights": dict(self.scaled_weights),
            "portfolio_variance_before": self.portfolio_variance_before,
            "portfolio_volatility_before": self.portfolio_volatility_before,
            "portfolio_variance_after": self.portfolio_variance_after,
            "portfolio_volatility_after": self.portfolio_volatility_after,
            "volatility_scale": self.volatility_scale,
            "released_weight_to_cash": self.released_weight_to_cash,
            "marginal_risk_contributions": dict(self.marginal_risk_contributions),
            "normalized_risk_contributions": dict(self.normalized_risk_contributions),
            "max_abs_pairwise_correlation": self.max_abs_pairwise_correlation,
            "covariance": self.covariance.to_dict(),
            "risk_only_shrinks": self.risk_only_shrinks,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }


def _portfolio_variance(weights: np.ndarray, covariance: np.ndarray) -> float:
    value = float(weights @ covariance @ weights)
    if value < 0 and abs(value) <= 1e-15:
        return 0.0
    if value < 0:
        raise ValueError("covariance produced negative portfolio variance")
    return value


def apply_covariance_risk_overlay(
    plan: PortfolioBookPlan,
    returns_by_asset: Mapping[str, Sequence[float]],
    *,
    config: CovarianceRiskConfig = CovarianceRiskConfig(),
) -> CovarianceRiskOverlay:
    """Apply a Qlib-style shrunk covariance risk overlay to Phase 44 weights.

    The portfolio brain still decides sign and relative conviction. This layer
    estimates joint risk from aligned return history and may only scale the
    entire book down proportionally when predicted period volatility exceeds a
    preregistered limit. It never flips direction or redistributes removed risk.
    """
    original = dict(plan.risk.weights)
    assets = tuple(sorted(original))
    if not assets:
        raise ValueError("covariance overlay requires a non-empty portfolio book")

    estimate = estimate_shrunk_covariance(
        returns_by_asset,
        assets=assets,
        config=config,
    )
    covariance = np.asarray(estimate.covariance, dtype=float)
    weights = np.asarray([float(original[asset]) for asset in assets], dtype=float)
    before_var = _portfolio_variance(weights, covariance)
    before_vol = math.sqrt(before_var)

    scale = 1.0
    if before_vol > config.max_period_volatility:
        scale = config.max_period_volatility / before_vol
    scale = max(0.0, min(1.0, scale))
    scaled = weights * scale

    after_var = _portfolio_variance(scaled, covariance)
    after_vol = math.sqrt(after_var)
    if after_vol > config.max_period_volatility + 1e-12:
        raise ValueError("covariance risk scaling failed to satisfy volatility limit")

    # Euler-style marginal/risk contributions at the original requested book.
    sigma_w = covariance @ weights
    marginal = {
        asset: float(sigma_w[index])
        for index, asset in enumerate(assets)
    }
    contribution_values = weights * sigma_w
    if before_var > 1e-18:
        normalized = {
            asset: float(contribution_values[index] / before_var)
            for index, asset in enumerate(assets)
        }
    else:
        normalized = {asset: 0.0 for asset in assets}

    corr = np.asarray(estimate.correlation, dtype=float)
    off_diag = [
        abs(float(corr[i, j]))
        for i in range(len(assets))
        for j in range(i + 1, len(assets))
        if math.isfinite(float(corr[i, j]))
    ]
    max_corr = max(off_diag, default=0.0)

    scaled_map = {
        asset: float(scaled[index])
        for index, asset in enumerate(assets)
    }
    for asset in assets:
        original_abs = abs(float(original[asset]))
        scaled_abs = abs(scaled_map[asset])
        if scaled_abs > original_abs + 1e-12:
            raise ValueError("risk overlay attempted to increase position size")
        if original[asset] != 0 and scaled_map[asset] != 0:
            if math.copysign(1.0, original[asset]) != math.copysign(1.0, scaled_map[asset]):
                raise ValueError("risk overlay attempted to flip position direction")

    released = max(
        0.0,
        sum(abs(float(value)) for value in original.values())
        - sum(abs(float(value)) for value in scaled_map.values()),
    )
    return CovarianceRiskOverlay(
        original_weights=dict(sorted(original.items())),
        scaled_weights=dict(sorted(scaled_map.items())),
        portfolio_variance_before=before_var,
        portfolio_volatility_before=before_vol,
        portfolio_variance_after=after_var,
        portfolio_volatility_after=after_vol,
        volatility_scale=scale,
        released_weight_to_cash=released,
        marginal_risk_contributions=marginal,
        normalized_risk_contributions=normalized,
        max_abs_pairwise_correlation=max_corr,
        covariance=estimate,
    )
