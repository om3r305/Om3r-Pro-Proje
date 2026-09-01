from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import time

EXPERIMENT_SCHEMA_VERSION = "brian.experiment.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    dataset_hash: str
    code_version: str
    config: Mapping[str, Any]
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    start_timestamp: float
    end_timestamp: float
    costs: Mapping[str, float]
    simulator_settings: Mapping[str, Any]
    metrics: Mapping[str, float | int]
    random_seed: int | None = None
    creation_timestamp: float = field(default_factory=time.time)
    schema_version: str = EXPERIMENT_SCHEMA_VERSION
    experiment_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.end_timestamp < self.start_timestamp:
            raise ValueError("experiment date range is reversed")
        object.__setattr__(self, "experiment_id", "")
        payload = asdict(self)
        payload.pop("experiment_id", None)
        object.__setattr__(self, "experiment_id", sha256(_canonical(payload)).hexdigest())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        content = _canonical(self.to_dict()) + b"\n"
        if target.exists():
            if target.read_bytes() != content:
                raise FileExistsError(f"immutable experiment already exists: {target}")
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target
