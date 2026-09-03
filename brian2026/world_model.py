from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Literal, Mapping, Sequence
import json
import math

import numpy as np

from .market_gym import GymBar, GymFrame, TRAINING_EVIDENCE_CLASS
from .portfolio import DEVELOPMENT_CUTOFF

WORLD_MODEL_SCHEMA_VERSION = "brian.world-model.v1"
WorldMode = Literal["REAL_REPLAY", "BLOCK_BOOTSTRAP", "STRESS_BOOTSTRAP"]


def _hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HistoricalBar:
    asset_id: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

    def __post_init__(self) -> None:
        if not self.asset_id.strip() or not math.isfinite(float(self.timestamp)):
            raise ValueError("historical bar requires asset and finite timestamp")
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("historical prices must be positive")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("invalid historical OHLC")
        if self.volume is not None and (not math.isfinite(float(self.volume)) or self.volume < 0):
            raise ValueError("historical volume must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MultiAssetHistory:
    dataset_id: str
    series: tuple[tuple[str, tuple[HistoricalBar, ...]], ...]

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.series:
            raise ValueError("dataset_id and historical series are required")
        ids = [asset for asset, _ in self.series]
        if len(ids) != len(set(ids)):
            raise ValueError("history cannot contain duplicate assets")
        for asset, bars in self.series:
            if not bars or any(bar.asset_id != asset for bar in bars):
                raise ValueError("history series asset mismatch")
            if any(right.timestamp <= left.timestamp for left, right in zip(bars, bars[1:])):
                raise ValueError("history bars must be strictly chronological")
            if any(bar.timestamp >= DEVELOPMENT_CUTOFF for bar in bars):
                raise ValueError("2026 historical development data is INVALID_CONTAMINATED and forbidden")

    @classmethod
    def from_mapping(cls, dataset_id: str, series: Mapping[str, Sequence[HistoricalBar]]) -> "MultiAssetHistory":
        rows = tuple(sorted((str(asset), tuple(bars)) for asset, bars in series.items()))
        return cls(dataset_id, rows)

    def as_mapping(self) -> dict[str, tuple[HistoricalBar, ...]]:
        return dict(self.series)

    @property
    def asset_ids(self) -> tuple[str, ...]:
        return tuple(asset for asset, _ in self.series)


@dataclass(frozen=True, slots=True)
class WorldModelConfig:
    horizon_steps: int = 512
    block_length: int = 32
    seed: int = 33
    stress_return_scale: float = 1.75

    def __post_init__(self) -> None:
        if self.horizon_steps < 2 or self.block_length < 2:
            raise ValueError("world horizon and block length must be >= 2")
        if self.stress_return_scale < 1.0 or not math.isfinite(float(self.stress_return_scale)):
            raise ValueError("stress_return_scale must be finite and >= 1")


@dataclass(frozen=True, slots=True)
class SourceBlock:
    start_index: int
    transition_count: int
    first_source_timestamp: float
    last_source_timestamp: float


@dataclass(frozen=True, slots=True)
class WorldReceipt:
    world_id: str
    mode: WorldMode
    seed: int
    episode_index: int
    source_dataset_id: str
    asset_ids: tuple[str, ...]
    source_blocks: tuple[SourceBlock, ...]
    generator_config: dict[str, object]
    synthetic: bool
    training_only: bool = True
    evidence_class: str = TRAINING_EVIDENCE_CLASS
    shadow_only: bool = True
    schema_version: str = WORLD_MODEL_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class WorldEpisode:
    frames: tuple[GymFrame, ...]
    receipt: WorldReceipt

    def __post_init__(self) -> None:
        if len(self.frames) < 2:
            raise ValueError("world episode requires at least two frames")
        if any(right.timestamp <= left.timestamp for left, right in zip(self.frames, self.frames[1:])):
            raise ValueError("world frames must be strictly chronological")


def align_history_intersection(history: MultiAssetHistory) -> tuple[GymFrame, ...]:
    """Exact timestamp intersection; deliberately never forward-fills closed markets."""
    mappings: dict[str, dict[float, HistoricalBar]] = {}
    for asset, bars in history.series:
        mappings[asset] = {float(bar.timestamp): bar for bar in bars}
    common: set[float] | None = None
    for rows in mappings.values():
        common = set(rows) if common is None else common.intersection(rows)
    timestamps = sorted(common or ())
    if len(timestamps) < 2:
        raise ValueError("multi-asset history has insufficient exact timestamp overlap")
    frames: list[GymFrame] = []
    for timestamp in timestamps:
        bars = tuple(
            GymBar(
                asset_id=asset,
                timestamp=timestamp,
                open=mappings[asset][timestamp].open,
                high=mappings[asset][timestamp].high,
                low=mappings[asset][timestamp].low,
                close=mappings[asset][timestamp].close,
                volume=mappings[asset][timestamp].volume,
                source_timestamp=timestamp,
            )
            for asset in sorted(mappings)
        )
        frames.append(GymFrame(timestamp, bars))
    return tuple(frames)


def _scale_ratio(ratio: float, scale: float) -> float:
    if ratio <= 0:
        raise ValueError("price ratio must be positive")
    return math.exp(math.log(ratio) * scale)


class BrianWorldModel:
    """Creates training worlds from causal historical development data.

    Synthetic worlds are curriculum/robustness data only. They are never forecast
    probabilities and never constitute real-world scientific evidence.
    """

    def __init__(self, history: MultiAssetHistory, config: WorldModelConfig = WorldModelConfig()) -> None:
        self.history = history
        self.config = config
        self.aligned = align_history_intersection(history)
        if len(self.aligned) < 3:
            raise ValueError("world model needs at least three synchronized frames")

    def _rng(self, episode_index: int) -> np.random.Generator:
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        return np.random.default_rng(np.random.SeedSequence([self.config.seed, int(episode_index)]))

    def _receipt(self, *, mode: WorldMode, episode_index: int, blocks: Sequence[SourceBlock], synthetic: bool) -> WorldReceipt:
        payload = {
            "schema_version": WORLD_MODEL_SCHEMA_VERSION,
            "mode": mode,
            "seed": self.config.seed,
            "episode_index": episode_index,
            "source_dataset_id": self.history.dataset_id,
            "asset_ids": self.history.asset_ids,
            "source_blocks": [asdict(block) for block in blocks],
            "generator_config": asdict(self.config),
            "synthetic": synthetic,
            "training_only": True,
            "evidence_class": TRAINING_EVIDENCE_CLASS,
        }
        world_id = _hash(payload)
        return WorldReceipt(
            world_id=world_id,
            mode=mode,
            seed=self.config.seed,
            episode_index=episode_index,
            source_dataset_id=self.history.dataset_id,
            asset_ids=self.history.asset_ids,
            source_blocks=tuple(blocks),
            generator_config=asdict(self.config),
            synthetic=synthetic,
        )

    def real_replay(self, *, episode_index: int = 0) -> WorldEpisode:
        horizon = min(self.config.horizon_steps + 1, len(self.aligned))
        max_start = len(self.aligned) - horizon
        rng = self._rng(episode_index)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        frames = self.aligned[start:start + horizon]
        block = SourceBlock(start, len(frames) - 1, frames[0].timestamp, frames[-1].timestamp)
        return WorldEpisode(frames, self._receipt(mode="REAL_REPLAY", episode_index=episode_index, blocks=(block,), synthetic=False))

    def _sample_transition_indices(self, episode_index: int) -> tuple[list[int], tuple[SourceBlock, ...]]:
        # Transition index j means source[j-1] -> source[j], so j starts at 1.
        available_transitions = len(self.aligned) - 1
        block_len = min(self.config.block_length, available_transitions)
        if block_len < 2:
            raise ValueError("insufficient history for block bootstrap")
        rng = self._rng(episode_index)
        sampled: list[int] = []
        blocks: list[SourceBlock] = []
        while len(sampled) < self.config.horizon_steps:
            remaining = self.config.horizon_steps - len(sampled)
            take = min(block_len, remaining)
            max_start = len(self.aligned) - take
            # Need source[start-1], therefore start >= 1.
            start = int(rng.integers(1, max_start + 1))
            indices = list(range(start, start + take))
            sampled.extend(indices)
            blocks.append(SourceBlock(
                start_index=start,
                transition_count=take,
                first_source_timestamp=self.aligned[start - 1].timestamp,
                last_source_timestamp=self.aligned[start + take - 1].timestamp,
            ))
        return sampled, tuple(blocks)

    def bootstrap(self, *, episode_index: int = 0, stress: bool = False) -> WorldEpisode:
        sampled, blocks = self._sample_transition_indices(episode_index)
        mode: WorldMode = "STRESS_BOOTSTRAP" if stress else "BLOCK_BOOTSTRAP"
        scale = self.config.stress_return_scale if stress else 1.0
        assets = self.history.asset_ids
        synthetic_close = {asset: 100.0 for asset in assets}
        first_bars = tuple(GymBar(asset, 0.0, 100.0, 100.0, 100.0, 100.0, source_timestamp=self.aligned[sampled[0] - 1].timestamp) for asset in assets)
        frames: list[GymFrame] = [GymFrame(0.0, first_bars)]

        for step, source_index in enumerate(sampled, start=1):
            previous = self.aligned[source_index - 1].by_asset()
            source = self.aligned[source_index].by_asset()
            bars: list[GymBar] = []
            for asset in assets:
                left = previous[asset]
                right = source[asset]
                gap_ratio = _scale_ratio(right.open / left.close, scale)
                close_ratio = _scale_ratio(right.close / right.open, scale)
                high_ratio = _scale_ratio(right.high / right.open, scale)
                low_ratio = _scale_ratio(right.low / right.open, scale)
                open_price = synthetic_close[asset] * gap_ratio
                close_price = open_price * close_ratio
                high_price = open_price * max(1.0, high_ratio, close_ratio)
                low_price = open_price * min(1.0, low_ratio, close_ratio)
                if low_price <= 0:
                    raise ValueError("synthetic stress transform produced non-positive price")
                bars.append(GymBar(
                    asset_id=asset,
                    timestamp=float(step),
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=right.volume,
                    source_timestamp=right.source_timestamp,
                ))
                synthetic_close[asset] = close_price
            frames.append(GymFrame(float(step), tuple(bars)))

        receipt = self._receipt(mode=mode, episode_index=episode_index, blocks=blocks, synthetic=True)
        return WorldEpisode(tuple(frames), receipt)

    def generate(self, mode: WorldMode, *, episode_index: int = 0) -> WorldEpisode:
        if mode == "REAL_REPLAY":
            return self.real_replay(episode_index=episode_index)
        if mode == "BLOCK_BOOTSTRAP":
            return self.bootstrap(episode_index=episode_index, stress=False)
        if mode == "STRESS_BOOTSTRAP":
            return self.bootstrap(episode_index=episode_index, stress=True)
        raise ValueError(f"unsupported world mode: {mode}")
