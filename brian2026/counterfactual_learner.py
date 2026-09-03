from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Mapping, Sequence
import json
import math

import numpy as np

from .curriculum_runner import PolicyObservation
from .experience_memory import EpisodeExperience
from .market_gym import GymEpisodeResult, GymFrame, TargetAllocation, TRAINING_EVIDENCE_CLASS
from .portfolio import DEVELOPMENT_CUTOFF

LEARNER_SCHEMA_VERSION = "brian.counterfactual-learner.v1"
POLICY_VERSION = "causal-counterfactual-learner-v1"
FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "return_1",
    "return_3",
    "trend_5",
    "realized_vol",
    "bar_range",
    "relative_momentum",
    "market_momentum",
    "market_dispersion",
    "volume_change",
)


def _hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(text.encode("utf-8")).hexdigest()


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _safe_log_ratio(right: float, left: float) -> float:
    if left <= 0 or right <= 0:
        raise ValueError("learner price ratios require positive values")
    return math.log(right / left)


@dataclass(frozen=True, slots=True)
class CounterfactualLearnerConfig:
    lookback: int = 16
    learning_rate: float = 0.025
    l2: float = 0.0005
    error_ewma_alpha: float = 0.05
    min_weighted_samples_per_asset: float = 12.0
    min_abs_edge: float = 0.0015
    risk_aversion: float = 0.75
    min_uncertainty: float = 0.0015
    max_label_abs: float = 0.25
    max_positions: int = 3
    max_asset_weight: float = 0.25
    max_gross_exposure: float = 0.75
    turnover_penalty_bps: float = 15.0
    real_replay_weight: float = 1.0
    block_bootstrap_weight: float = 0.50
    stress_bootstrap_weight: float = 0.25
    drawdown_throttle_1: float = 0.15
    drawdown_throttle_2: float = 0.30
    drawdown_flatten: float = 0.50

    def __post_init__(self) -> None:
        if self.lookback < 3:
            raise ValueError("lookback must be >= 3")
        if not 0 < self.learning_rate <= 0.25 or self.l2 < 0:
            raise ValueError("invalid learner optimization configuration")
        if not 0 < self.error_ewma_alpha <= 1:
            raise ValueError("error_ewma_alpha must be in (0,1]")
        if self.min_weighted_samples_per_asset < 1 or self.min_abs_edge < 0:
            raise ValueError("invalid learner warmup/edge configuration")
        if self.risk_aversion < 0 or self.min_uncertainty < 0 or self.max_label_abs <= 0:
            raise ValueError("invalid learner risk configuration")
        if self.max_positions < 1:
            raise ValueError("max_positions must be positive")
        if not 0 < self.max_asset_weight <= 1 or not 0 < self.max_gross_exposure <= 1:
            raise ValueError("learner allocation must remain unlevered")
        if self.max_asset_weight > self.max_gross_exposure:
            raise ValueError("max_asset_weight cannot exceed max_gross_exposure")
        if self.turnover_penalty_bps < 0:
            raise ValueError("turnover penalty must be non-negative")
        if min(self.real_replay_weight, self.block_bootstrap_weight, self.stress_bootstrap_weight) <= 0:
            raise ValueError("world-mode training weights must be positive")
        if not 0 < self.drawdown_throttle_1 < self.drawdown_throttle_2 < self.drawdown_flatten < 1:
            raise ValueError("drawdown throttles must be strictly ordered in (0,1)")


@dataclass(slots=True)
class _AssetState:
    weights: list[float]
    weighted_samples: float = 0.0
    updates: int = 0
    error_ewma: float = 0.01

    @classmethod
    def fresh(cls) -> "_AssetState":
        return cls([0.0] * len(FEATURE_NAMES))


@dataclass(frozen=True, slots=True)
class DecisionTrainingRecord:
    decision_step: int
    features: tuple[tuple[str, tuple[float, ...]], ...]
    allocation: tuple[tuple[str, float], ...]


class CausalCounterfactualLearner:
    """Deterministic full-information learner for the Market Gym.

    It never receives future frames while making a decision. Features/actions are cached
    during an episode and model updates occur only after the episode is fully resolved.
    For each decision it learns the feasible open-to-open return of every observed asset,
    not only the asset it happened to choose, which gives counterfactual BUY/SELL/WAIT
    experience without inventing market paths.
    """

    def __init__(self, config: CounterfactualLearnerConfig = CounterfactualLearnerConfig()) -> None:
        self.config = config
        self._models: dict[str, _AssetState] = {}
        self._episode_records: list[DecisionTrainingRecord] = []
        self._episodes_learned = 0
        self._transitions_learned = 0
        self._weighted_transitions = 0.0
        self._counterfactual_regret_sum = 0.0
        self._counterfactual_decisions = 0
        self._last_experience_id: str | None = None

    @property
    def policy_version(self) -> str:
        return POLICY_VERSION

    @property
    def training_state_id(self) -> str:
        return _hash(self.state_dict())

    @staticmethod
    def _validate_frame_sources(frames: Sequence[GymFrame]) -> None:
        for frame in frames:
            for bar in frame.bars:
                source_ts = bar.source_timestamp
                if source_ts is not None and source_ts >= DEVELOPMENT_CUTOFF:
                    raise ValueError("2026 source data is INVALID_CONTAMINATED and forbidden")

    def _feature_map(self, frames: Sequence[GymFrame]) -> dict[str, tuple[float, ...]]:
        if not frames:
            raise ValueError("learner needs visible market frames")
        self._validate_frame_sources(frames)
        current = frames[-1]
        current_assets = current.by_asset()
        previous_assets = frames[-2].by_asset() if len(frames) >= 2 else {}

        return_1_by_asset: dict[str, float] = {}
        for asset, bar in current_assets.items():
            previous = previous_assets.get(asset)
            return_1_by_asset[asset] = _safe_log_ratio(bar.close, previous.close) if previous is not None else 0.0
        market_returns = np.asarray(tuple(return_1_by_asset.values()), dtype=float)
        market_momentum = float(np.mean(market_returns)) if len(market_returns) else 0.0
        market_dispersion = float(np.std(market_returns)) if len(market_returns) > 1 else 0.0

        out: dict[str, tuple[float, ...]] = {}
        lookback_frames = tuple(frames[-self.config.lookback:])
        for asset, current_bar in current_assets.items():
            history = [frame.by_asset().get(asset) for frame in lookback_frames]
            bars = [bar for bar in history if bar is not None and bar.tradable]
            close_returns = [
                _safe_log_ratio(right.close, left.close)
                for left, right in zip(bars, bars[1:])
            ]
            r1 = return_1_by_asset.get(asset, 0.0)
            if len(bars) >= 4:
                r3 = _safe_log_ratio(bars[-1].close, bars[-4].close)
            elif len(bars) >= 2:
                r3 = _safe_log_ratio(bars[-1].close, bars[0].close)
            else:
                r3 = 0.0
            trend = float(np.mean(close_returns[-5:])) if close_returns else 0.0
            vol = float(np.std(close_returns[-self.config.lookback:])) if len(close_returns) > 1 else 0.0
            bar_range = max(0.0, current_bar.high / current_bar.low - 1.0)
            relative = r1 - market_momentum
            volume_change = 0.0
            previous = previous_assets.get(asset)
            if (
                previous is not None
                and current_bar.volume is not None
                and previous.volume is not None
                and current_bar.volume > 0
                and previous.volume > 0
            ):
                volume_change = _clip(math.log(current_bar.volume / previous.volume) / 3.0, -1.0, 1.0)

            vector = (
                1.0,
                _clip(r1 * 20.0, -3.0, 3.0),
                _clip(r3 * 10.0, -3.0, 3.0),
                _clip(trend * 20.0, -3.0, 3.0),
                _clip(vol * 20.0, 0.0, 3.0),
                _clip(bar_range * 10.0, 0.0, 3.0),
                _clip(relative * 20.0, -3.0, 3.0),
                _clip(market_momentum * 20.0, -3.0, 3.0),
                _clip(market_dispersion * 20.0, 0.0, 3.0),
                volume_change,
            )
            if len(vector) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in vector):
                raise ValueError("learner produced invalid causal features")
            out[asset] = vector
        return out

    def _model(self, asset: str) -> _AssetState:
        state = self._models.get(asset)
        if state is None:
            state = _AssetState.fresh()
            self._models[asset] = state
        return state

    @staticmethod
    def _predict(state: _AssetState, features: Sequence[float]) -> float:
        return float(np.dot(np.asarray(state.weights, dtype=float), np.asarray(features, dtype=float)))

    def _gross_budget(self, observation: PolicyObservation) -> float:
        drawdown = max(0.0, 1.0 - observation.equity / max(observation.starting_equity, 1e-12))
        if drawdown >= self.config.drawdown_flatten:
            return 0.0
        if drawdown >= self.config.drawdown_throttle_2:
            return self.config.max_gross_exposure * 0.25
        if drawdown >= self.config.drawdown_throttle_1:
            return self.config.max_gross_exposure * 0.50
        return self.config.max_gross_exposure

    def _allocation(self, features: Mapping[str, tuple[float, ...]], observation: PolicyObservation) -> TargetAllocation:
        budget = self._gross_budget(observation)
        if budget <= 1e-12:
            return TargetAllocation()
        current = dict(observation.current_weights)
        candidates: list[tuple[float, str, float]] = []
        turnover_rate = self.config.turnover_penalty_bps / 10_000.0

        for asset, vector in sorted(features.items()):
            state = self._models.get(asset)
            if state is None or state.weighted_samples < self.config.min_weighted_samples_per_asset:
                continue
            prediction = _clip(self._predict(state, vector), -self.config.max_label_abs, self.config.max_label_abs)
            uncertainty = max(self.config.min_uncertainty, state.error_ewma)
            long_edge = prediction - self.config.risk_aversion * uncertainty
            short_edge = -prediction - self.config.risk_aversion * uncertainty
            direction = 1.0 if long_edge >= short_edge else -1.0
            raw_edge = max(long_edge, short_edge)
            if raw_edge <= self.config.min_abs_edge:
                continue
            strength = _clip(raw_edge / max(self.config.min_abs_edge * 4.0, 1e-12), 0.25, 1.0)
            desired = direction * self.config.max_asset_weight * strength
            score = raw_edge - turnover_rate * abs(desired - current.get(asset, 0.0))
            if score > self.config.min_abs_edge:
                candidates.append((score, asset, desired))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        weights: dict[str, float] = {}
        remaining = budget
        for _, asset, desired in candidates[:self.config.max_positions]:
            magnitude = min(abs(desired), remaining, self.config.max_asset_weight)
            if magnitude <= 1e-12:
                break
            weights[asset] = math.copysign(magnitude, desired)
            remaining -= magnitude
            if remaining <= 1e-12:
                break
        return TargetAllocation.from_mapping(weights)

    def act(self, observation: PolicyObservation) -> TargetAllocation:
        if observation.step_index == 0:
            if self._episode_records:
                raise RuntimeError("previous episode was not resolved before the next episode started")
        elif self._episode_records and observation.step_index != self._episode_records[-1].decision_step + 1:
            raise ValueError("learner observations must be sequential inside an episode")
        features = self._feature_map(observation.visible_frames)
        allocation = self._allocation(features, observation)
        record = DecisionTrainingRecord(
            decision_step=observation.step_index,
            features=tuple(sorted(features.items())),
            allocation=allocation.weights,
        )
        self._episode_records.append(record)
        return allocation

    def learn_after_episode(self, experience: EpisodeExperience) -> None:
        if not experience.training_only or experience.evidence_class != TRAINING_EVIDENCE_CLASS:
            raise ValueError("learner accepts only TRAINING_ONLY episode experience")
        self._last_experience_id = experience.experience_id

    def _mode_weight(self, world_mode: str) -> float:
        if world_mode == "REAL_REPLAY":
            return self.config.real_replay_weight
        if world_mode == "BLOCK_BOOTSTRAP":
            return self.config.block_bootstrap_weight
        if world_mode == "STRESS_BOOTSTRAP":
            return self.config.stress_bootstrap_weight
        raise ValueError(f"unknown curriculum world mode: {world_mode}")

    def _update_asset(self, asset: str, features: Sequence[float], label: float, sample_weight: float) -> None:
        state = self._model(asset)
        x = np.asarray(features, dtype=float)
        weights = np.asarray(state.weights, dtype=float)
        prediction = float(np.dot(weights, x))
        error = prediction - label
        gradient = sample_weight * error * x + self.config.l2 * weights
        gradient = np.clip(gradient, -5.0, 5.0)
        weights = np.clip(weights - self.config.learning_rate * gradient, -1.0, 1.0)
        state.weights = weights.astype(float).tolist()
        absolute_error = abs(error)
        alpha = self.config.error_ewma_alpha
        state.error_ewma = (1.0 - alpha) * state.error_ewma + alpha * absolute_error
        state.weighted_samples += sample_weight
        state.updates += 1

    def learn_after_resolved_episode(
        self,
        experience: EpisodeExperience,
        result: GymEpisodeResult,
        resolved_frames: tuple[GymFrame, ...],
    ) -> None:
        if self._last_experience_id != experience.experience_id:
            raise ValueError("resolved episode does not match the released training summary")
        if not result.shadow_only or result.evidence_class != TRAINING_EVIDENCE_CLASS:
            raise ValueError("resolved gym result must remain TRAINING_ONLY")
        if len(resolved_frames) < 3:
            raise ValueError("counterfactual learning requires at least three resolved frames")
        self._validate_frame_sources(resolved_frames)
        sample_weight = self._mode_weight(experience.world_mode)

        records = tuple(self._episode_records)
        for record in records:
            # Target chosen at decision d executes at open(d+1), remains exposed through
            # the next gap, and can be changed at open(d+2). Therefore open(d+1)->open(d+2)
            # is the causal counterfactual holding return for that target decision.
            if record.decision_step + 2 >= len(resolved_frames):
                continue
            entry = resolved_frames[record.decision_step + 1].by_asset()
            exit_ = resolved_frames[record.decision_step + 2].by_asset()
            labels: dict[str, float] = {}
            for asset, vector in record.features:
                left, right = entry.get(asset), exit_.get(asset)
                if left is None or right is None or not left.tradable or not right.tradable:
                    continue
                label = _clip(right.open / left.open - 1.0, -self.config.max_label_abs, self.config.max_label_abs)
                labels[asset] = label
                self._update_asset(asset, vector, label, sample_weight)
                self._transitions_learned += 1
                self._weighted_transitions += sample_weight

            if labels:
                chosen = sum(float(weight) * labels.get(asset, 0.0) for asset, weight in record.allocation)
                best_single = self.config.max_asset_weight * max(abs(value) for value in labels.values())
                self._counterfactual_regret_sum += max(0.0, best_single - chosen)
                self._counterfactual_decisions += 1

        self._episodes_learned += 1
        self._episode_records.clear()
        self._last_experience_id = None

    def training_manifest(self) -> dict[str, object]:
        models = {
            asset: {
                "weighted_samples": state.weighted_samples,
                "updates": state.updates,
                "error_ewma": state.error_ewma,
            }
            for asset, state in sorted(self._models.items())
        }
        return {
            "schema_version": LEARNER_SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "episodes_learned": self._episodes_learned,
            "transitions_learned": self._transitions_learned,
            "weighted_transitions": self._weighted_transitions,
            "counterfactual_decisions": self._counterfactual_decisions,
            "mean_single_asset_counterfactual_regret": (
                self._counterfactual_regret_sum / self._counterfactual_decisions
                if self._counterfactual_decisions else 0.0
            ),
            "asset_models": models,
            "training_only": True,
            "shadow_only": True,
        }

    def state_dict(self) -> dict[str, object]:
        if self._episode_records:
            # Checkpoints are shard boundaries and must never capture a half-resolved episode.
            raise RuntimeError("cannot checkpoint learner with a live unresolved episode")
        return {
            "schema_version": LEARNER_SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "feature_names": FEATURE_NAMES,
            "config": asdict(self.config),
            "models": {
                asset: {
                    "weights": state.weights,
                    "weighted_samples": state.weighted_samples,
                    "updates": state.updates,
                    "error_ewma": state.error_ewma,
                }
                for asset, state in sorted(self._models.items())
            },
            "episodes_learned": self._episodes_learned,
            "transitions_learned": self._transitions_learned,
            "weighted_transitions": self._weighted_transitions,
            "counterfactual_regret_sum": self._counterfactual_regret_sum,
            "counterfactual_decisions": self._counterfactual_decisions,
            "training_only": True,
            "shadow_only": True,
        }

    @classmethod
    def from_state(cls, payload: Mapping[str, object]) -> "CausalCounterfactualLearner":
        if payload.get("schema_version") != LEARNER_SCHEMA_VERSION or payload.get("policy_version") != POLICY_VERSION:
            raise ValueError("unsupported learner checkpoint")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("learner checkpoint feature contract mismatch")
        config_raw = payload.get("config")
        if not isinstance(config_raw, Mapping):
            raise ValueError("learner checkpoint is missing config")
        learner = cls(CounterfactualLearnerConfig(**dict(config_raw)))
        models_raw = payload.get("models", {})
        if not isinstance(models_raw, Mapping):
            raise ValueError("learner checkpoint models must be a mapping")
        for asset, raw in models_raw.items():
            if not isinstance(raw, Mapping):
                raise ValueError("invalid learner asset checkpoint")
            weights = [float(value) for value in raw.get("weights", ())]
            if len(weights) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in weights):
                raise ValueError("invalid learner checkpoint weights")
            learner._models[str(asset)] = _AssetState(
                weights=weights,
                weighted_samples=float(raw.get("weighted_samples", 0.0)),
                updates=int(raw.get("updates", 0)),
                error_ewma=float(raw.get("error_ewma", 0.01)),
            )
        learner._episodes_learned = int(payload.get("episodes_learned", 0))
        learner._transitions_learned = int(payload.get("transitions_learned", 0))
        learner._weighted_transitions = float(payload.get("weighted_transitions", 0.0))
        learner._counterfactual_regret_sum = float(payload.get("counterfactual_regret_sum", 0.0))
        learner._counterfactual_decisions = int(payload.get("counterfactual_decisions", 0))
        if not bool(payload.get("training_only", False)) or not bool(payload.get("shadow_only", False)):
            raise ValueError("learner checkpoint must remain training-only shadow research")
        return learner
