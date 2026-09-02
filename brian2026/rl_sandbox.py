from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence
import math

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .portfolio import DEVELOPMENT_CUTOFF

TargetPosition = Literal[-1, 0, 1]
Signal = Literal["BUY", "SELL", "WAIT"]
Partition = Literal["train", "validation", "test"]
ACTIONS: tuple[TargetPosition, ...] = (-1, 0, 1)
SCHEMA_VERSION = "brian.rl-sandbox.v1"


@dataclass(frozen=True, slots=True)
class RLSandboxConfig:
    fee_bps: float = 10.0
    assumed_spread_bps: float = 2.0
    slippage_bps: float = 1.0
    risk_penalty: float = 0.20
    gamma: float = 0.97
    fitted_q_iterations: int = 5
    trees: int = 96
    min_samples_leaf: int = 25
    min_action_advantage_pct: float = 0.015
    max_train_transitions: int = 250_000
    random_state: int = 30

    def __post_init__(self) -> None:
        if min(self.fee_bps, self.assumed_spread_bps, self.slippage_bps, self.risk_penalty) < 0:
            raise ValueError("cost/risk parameters must be non-negative")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        if self.fitted_q_iterations < 1 or self.trees < 1 or self.min_samples_leaf < 1:
            raise ValueError("invalid learner configuration")
        if self.max_train_transitions < 100:
            raise ValueError("max_train_transitions is too small")

    @property
    def round_trip_cost_bps(self) -> float:
        return 2.0 * self.fee_bps + self.assumed_spread_bps + 2.0 * self.slippage_bps


@dataclass(frozen=True, slots=True)
class SandboxTransition:
    observation_timestamp: float
    execution_timestamp: float
    state: tuple[float, ...]
    position_before: TargetPosition
    action: TargetPosition
    reward_pct: float
    next_state: tuple[float, ...]
    next_position: TargetPosition
    terminal: bool
    next_open: float
    next_close: float
    adverse_excursion_pct: float
    shadow_only: bool = True
    schema_version: str = SCHEMA_VERSION

    def manifest(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SandboxDecision:
    timestamp: float
    target_position: TargetPosition
    signal: Signal
    q_short: float
    q_flat: float
    q_long: float
    advantage_pct: float
    abstained: bool
    shadow_only: bool = True
    schema_version: str = SCHEMA_VERSION

    def manifest(self) -> dict:
        return asdict(self)


def _finite_state(values: Sequence[float]) -> tuple[float, ...]:
    out: list[float] = []
    for value in values:
        number = float(value)
        out.append(number if math.isfinite(number) else 0.0)
    return tuple(out)


def _encode_features(features: Sequence[float]) -> tuple[float, ...]:
    """Encode value + explicit missing flag so unavailable evidence is not confused with zero."""
    out: list[float] = []
    for value in features:
        number = float(value)
        missing = not math.isfinite(number)
        out.extend((0.0 if missing else number, 1.0 if missing else 0.0))
    return tuple(out)


def state_with_position(features: Sequence[float], position: TargetPosition) -> tuple[float, ...]:
    if position not in ACTIONS:
        raise ValueError("invalid position")
    base = _encode_features(features)
    return base + (
        1.0 if position == -1 else 0.0,
        1.0 if position == 0 else 0.0,
        1.0 if position == 1 else 0.0,
    )


def _transition_cost_pct(before: TargetPosition, after: TargetPosition,
                         config: RLSandboxConfig) -> float:
    # Exposure change 0->1 is one side of a round trip; -1->1 closes and reopens.
    exposure_change = abs(int(after) - int(before))
    one_way_bps = config.fee_bps + config.assumed_spread_bps / 2.0 + config.slippage_bps
    return exposure_change * one_way_bps / 100.0


def _adverse_excursion_pct(action: TargetPosition, next_open: float,
                           next_high: float, next_low: float) -> float:
    if action == 1:
        return min(0.0, (next_low / next_open - 1.0) * 100.0)
    if action == -1:
        return min(0.0, (1.0 - next_high / next_open) * 100.0)
    return 0.0


def counterfactual_transition(*, observation_timestamp: float, execution_timestamp: float,
                              current_features: Sequence[float], next_features: Sequence[float],
                              position_before: TargetPosition, action: TargetPosition,
                              next_open: float, next_high: float, next_low: float, next_close: float,
                              terminal: bool = False,
                              config: RLSandboxConfig = RLSandboxConfig()) -> SandboxTransition:
    if observation_timestamp >= DEVELOPMENT_CUTOFF or execution_timestamp >= DEVELOPMENT_CUTOFF:
        raise ValueError("2026 data is INVALID_CONTAMINATED and forbidden")
    if execution_timestamp <= observation_timestamp:
        raise ValueError("execution must occur strictly after the observed completed bar")
    if position_before not in ACTIONS or action not in ACTIONS:
        raise ValueError("invalid target position")
    if min(next_open, next_high, next_low, next_close) <= 0:
        raise ValueError("prices must be positive")
    if next_low > min(next_open, next_close) or next_high < max(next_open, next_close):
        raise ValueError("invalid next-bar OHLC")

    gross = int(action) * (next_close / next_open - 1.0) * 100.0
    cost = _transition_cost_pct(position_before, action, config)
    adverse = _adverse_excursion_pct(action, next_open, next_high, next_low)
    reward = gross - cost - config.risk_penalty * abs(adverse)
    return SandboxTransition(
        float(observation_timestamp), float(execution_timestamp),
        state_with_position(current_features, position_before), position_before, action,
        float(reward), state_with_position(next_features, action), action, bool(terminal),
        float(next_open), float(next_close), float(adverse),
    )


def build_counterfactual_transitions(timestamps: Sequence[float], opens: Sequence[float],
                                     highs: Sequence[float], lows: Sequence[float],
                                     closes: Sequence[float], feature_matrix: np.ndarray,
                                     indices: Sequence[int], *,
                                     starting_positions: Sequence[TargetPosition] = ACTIONS,
                                     config: RLSandboxConfig = RLSandboxConfig()) -> tuple[SandboxTransition, ...]:
    t = np.asarray(timestamps, dtype=float)
    o = np.asarray(opens, dtype=float)
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    x = np.asarray(feature_matrix, dtype=float)
    if not (len(t) == len(o) == len(h) == len(l) == len(c) == len(x)):
        raise ValueError("market arrays and features must align")
    if x.ndim != 2:
        raise ValueError("feature_matrix must be two-dimensional")
    out: list[SandboxTransition] = []
    selected = [int(i) for i in indices]
    for offset, i in enumerate(selected):
        if i < 0 or i + 1 >= len(t):
            raise ValueError("transition index requires a next bar")
        if t[i] >= DEVELOPMENT_CUTOFF or t[i + 1] >= DEVELOPMENT_CUTOFF:
            raise ValueError("2026 data is INVALID_CONTAMINATED and forbidden")
        terminal = offset == len(selected) - 1 or (offset + 1 < len(selected) and selected[offset + 1] != i + 1)
        for before in starting_positions:
            for action in ACTIONS:
                out.append(counterfactual_transition(
                    observation_timestamp=t[i], execution_timestamp=t[i + 1],
                    current_features=x[i], next_features=x[i + 1], position_before=before,
                    action=action, next_open=o[i + 1], next_high=h[i + 1], next_low=l[i + 1],
                    next_close=c[i + 1], terminal=terminal, config=config,
                ))
    return tuple(out)


class ConservativeFittedQAgent:
    """Deterministic research-only Fitted-Q challenger. It has no exchange execution surface."""

    def __init__(self, config: RLSandboxConfig = RLSandboxConfig()) -> None:
        self.config = config
        self.models: dict[TargetPosition, ExtraTreesRegressor] = {}
        self.fit_partition: str | None = None
        self.state_width: int | None = None

    def _model(self, action: TargetPosition, iteration: int) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(
            n_estimators=self.config.trees,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state + 100 * iteration + (action + 1),
            n_jobs=-1,
        )

    @staticmethod
    def _rows(transitions: Sequence[SandboxTransition]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = np.asarray([row.state for row in transitions], dtype=float)
        next_states = np.asarray([row.next_state for row in transitions], dtype=float)
        actions = np.asarray([int(row.action) for row in transitions], dtype=int)
        rewards = np.asarray([row.reward_pct for row in transitions], dtype=float)
        return states, next_states, actions, rewards

    def fit(self, transitions: Sequence[SandboxTransition], *, partition: Partition = "train") -> "ConservativeFittedQAgent":
        if partition != "train":
            raise ValueError("RL learner may fit only on train")
        if not transitions:
            raise ValueError("training transitions are required")
        if any(row.observation_timestamp >= DEVELOPMENT_CUTOFF or row.execution_timestamp >= DEVELOPMENT_CUTOFF for row in transitions):
            raise ValueError("2026 data is INVALID_CONTAMINATED and forbidden")

        rows = list(transitions)
        if len(rows) > self.config.max_train_transitions:
            # Deterministic evenly-spaced reduction, not outcome-based sampling.
            keep = np.linspace(0, len(rows) - 1, self.config.max_train_transitions, dtype=int)
            rows = [rows[int(i)] for i in keep]
        states, next_states, actions, rewards = self._rows(rows)
        self.state_width = states.shape[1]
        terminals = np.asarray([row.terminal for row in rows], dtype=bool)

        targets = rewards.copy()
        models: dict[TargetPosition, ExtraTreesRegressor] = {}
        for iteration in range(self.config.fitted_q_iterations):
            current: dict[TargetPosition, ExtraTreesRegressor] = {}
            for action in ACTIONS:
                mask = actions == action
                if int(mask.sum()) < 2:
                    raise ValueError(f"insufficient transitions for action {action}")
                model = self._model(action, iteration)
                model.fit(states[mask], targets[mask])
                current[action] = model
            if iteration + 1 < self.config.fitted_q_iterations:
                future = np.column_stack([current[action].predict(next_states) for action in ACTIONS])
                bootstrap = np.max(future, axis=1)
                targets = rewards + self.config.gamma * np.where(terminals, 0.0, bootstrap)
            models = current
        self.models = models
        self.fit_partition = partition
        return self

    def q_values(self, state: Sequence[float]) -> Mapping[TargetPosition, float]:
        if self.fit_partition != "train" or not self.models or self.state_width is None:
            raise RuntimeError("RL agent is not fitted")
        row = np.asarray(_finite_state(state), dtype=float)[None, :]
        if row.shape[1] != self.state_width:
            raise ValueError("state width differs from fitted model")
        return {action: float(self.models[action].predict(row)[0]) for action in ACTIONS}

    def decide(self, *, timestamp: float, features: Sequence[float],
               current_position: TargetPosition) -> SandboxDecision:
        if timestamp >= DEVELOPMENT_CUTOFF:
            raise ValueError("2026 data is INVALID_CONTAMINATED and forbidden")
        state = state_with_position(features, current_position)
        values = self.q_values(state)
        best = max(ACTIONS, key=lambda action: (values[action], -abs(action), action))
        flat = values[0]
        advantage = values[best] - flat
        abstained = best != 0 and advantage < self.config.min_action_advantage_pct
        target: TargetPosition = 0 if abstained else best
        signal: Signal = "WAIT" if target == 0 else "BUY" if target == 1 else "SELL"
        return SandboxDecision(
            float(timestamp), target, signal,
            values[-1], values[0], values[1], float(max(0.0, values[target] - flat)), abstained,
        )

    @property
    def shadow_only(self) -> bool:
        return True
