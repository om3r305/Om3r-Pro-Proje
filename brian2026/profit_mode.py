from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Mapping, Protocol
import json
import math

from .counterfactual_learner import CausalCounterfactualLearner, _clip
from .curriculum_runner import PolicyObservation
from .market_gym import MarketGymConfig, TargetAllocation

PROFIT_MODE_SCHEMA_VERSION = "brian.profit-seeking-shadow.v1"
PROFIT_POLICY_VERSION = "profit-seeking-shadow-v1"


def _hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ProfitSeekingConfig:
    """Locked cost-aware objective layered on a frozen trained learner.

    The policy does not change the learner's model weights. It only decides whether the
    learner's predicted directional edge remains positive after uncertainty and a
    conservative estimated entry+exit cost budget are charged.
    """

    round_trip_cost_multiplier: float = 2.0
    max_positions: int = 3
    max_asset_weight: float = 0.25
    max_gross_exposure: float = 0.75
    min_strength: float = 0.25
    shadow_only: bool = True
    schema_version: str = PROFIT_MODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.round_trip_cost_multiplier < 1.0 or not math.isfinite(self.round_trip_cost_multiplier):
            raise ValueError("round_trip_cost_multiplier must be finite and >= 1")
        if self.max_positions < 1:
            raise ValueError("max_positions must be positive")
        if not 0 < self.max_asset_weight <= 1 or not 0 < self.max_gross_exposure <= 1:
            raise ValueError("profit mode must remain unlevered")
        if self.max_asset_weight > self.max_gross_exposure:
            raise ValueError("max_asset_weight cannot exceed max_gross_exposure")
        if not 0 < self.min_strength <= 1:
            raise ValueError("min_strength must be in (0,1]")
        if not self.shadow_only:
            raise ValueError("profit-seeking mode is hard shadow-only")


class FrozenShadowPolicy(Protocol):
    @property
    def policy_version(self) -> str: ...

    @property
    def training_state_id(self) -> str: ...

    def act(self, observation: PolicyObservation) -> TargetAllocation: ...


class FrozenNativePolicy:
    """Read-only copy of the learner's existing action rule for an exam baseline."""

    def __init__(self, learner: CausalCounterfactualLearner) -> None:
        self.learner = learner
        self._state_id = learner.training_state_id

    @property
    def policy_version(self) -> str:
        return "frozen-native-counterfactual-v1"

    @property
    def training_state_id(self) -> str:
        return self._state_id

    def act(self, observation: PolicyObservation) -> TargetAllocation:
        if self.learner.training_state_id != self._state_id:
            raise RuntimeError("frozen native learner state changed during evaluation")
        features = self.learner._feature_map(observation.visible_frames)
        allocation = self.learner._allocation(features, observation)
        if self.learner.training_state_id != self._state_id:
            raise RuntimeError("frozen native policy mutated learner state")
        return allocation


class ProfitSeekingShadowPolicy:
    """Cost-aware growth objective over a frozen causal learner.

    This policy is deliberately not an execution engine and has no learning callbacks.
    It cannot promote itself, change code, use credentials, or send orders.
    """

    def __init__(
        self,
        learner: CausalCounterfactualLearner,
        gym_config: MarketGymConfig,
        config: ProfitSeekingConfig = ProfitSeekingConfig(),
    ) -> None:
        self.learner = learner
        self.gym_config = gym_config
        self.config = config
        self._learner_state_id = learner.training_state_id
        self._policy_state_id = _hash({
            "schema_version": PROFIT_MODE_SCHEMA_VERSION,
            "policy_version": PROFIT_POLICY_VERSION,
            "learner_state_id": self._learner_state_id,
            "gym_config": asdict(gym_config),
            "profit_config": asdict(config),
        })

    @property
    def policy_version(self) -> str:
        return PROFIT_POLICY_VERSION

    @property
    def training_state_id(self) -> str:
        return self._policy_state_id

    def _gross_budget(self, observation: PolicyObservation) -> float:
        learner_budget = self.learner._gross_budget(observation)
        return min(float(learner_budget), self.config.max_gross_exposure)

    def act(self, observation: PolicyObservation) -> TargetAllocation:
        if self.learner.training_state_id != self._learner_state_id:
            raise RuntimeError("profit-mode learner state changed during frozen evaluation")

        features = self.learner._feature_map(observation.visible_frames)
        budget = self._gross_budget(observation)
        if budget <= 1e-12:
            return TargetAllocation()

        current = dict(observation.current_weights)
        candidates: list[tuple[float, str, float]] = []
        learner_cfg = self.learner.config
        round_trip_cost_rate = self.gym_config.one_way_cost_rate * self.config.round_trip_cost_multiplier

        for asset, vector in sorted(features.items()):
            state = self.learner._models.get(asset)
            if state is None or state.weighted_samples < learner_cfg.min_weighted_samples_per_asset:
                continue

            prediction = _clip(
                self.learner._predict(state, vector),
                -learner_cfg.max_label_abs,
                learner_cfg.max_label_abs,
            )
            uncertainty = max(learner_cfg.min_uncertainty, state.error_ewma)
            long_edge = prediction - learner_cfg.risk_aversion * uncertainty
            short_edge = -prediction - learner_cfg.risk_aversion * uncertainty
            direction = 1.0 if long_edge >= short_edge else -1.0
            raw_edge = max(long_edge, short_edge)
            if raw_edge <= learner_cfg.min_abs_edge:
                continue

            native_strength = _clip(
                raw_edge / max(learner_cfg.min_abs_edge * 4.0, 1e-12),
                self.config.min_strength,
                1.0,
            )
            desired = direction * min(self.config.max_asset_weight, learner_cfg.max_asset_weight) * native_strength
            incremental_turnover = abs(desired - current.get(asset, 0.0))
            expected_cost = round_trip_cost_rate * incremental_turnover
            net_edge = raw_edge - expected_cost
            if net_edge <= learner_cfg.min_abs_edge:
                continue

            net_strength = _clip(
                net_edge / max(learner_cfg.min_abs_edge * 4.0, 1e-12),
                self.config.min_strength,
                1.0,
            )
            desired = direction * min(self.config.max_asset_weight, learner_cfg.max_asset_weight) * net_strength
            candidates.append((net_edge, asset, desired))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        weights: dict[str, float] = {}
        remaining = budget
        for _, asset, desired in candidates[: self.config.max_positions]:
            magnitude = min(abs(desired), remaining, self.config.max_asset_weight)
            if magnitude <= 1e-12:
                break
            weights[asset] = math.copysign(magnitude, desired)
            remaining -= magnitude
            if remaining <= 1e-12:
                break

        allocation = TargetAllocation.from_mapping(weights)
        if allocation.gross_exposure > self.config.max_gross_exposure + 1e-12:
            raise ValueError("profit mode exceeded hard gross exposure limit")
        if self.learner.training_state_id != self._learner_state_id:
            raise RuntimeError("profit-mode action mutated frozen learner state")
        return allocation


def policy_contract(policy: FrozenShadowPolicy) -> dict[str, object]:
    return {
        "policy_version": policy.policy_version,
        "training_state_id": policy.training_state_id,
        "shadow_only": True,
        "live_execution": False,
        "automatic_promotion": False,
    }
