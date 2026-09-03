from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import math
import os
import tempfile


SCHEMA = "brian.intelligence-capture.v1"


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=lambda x: str(x))}
    if isinstance(value, (tuple, list)):
        return [_canonical(x) for x in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("capture payload cannot contain NaN or infinity")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported capture payload type: {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _hash(value: Any) -> str:
    return sha256(_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class IntelligenceCapture:
    provider: str
    record_type: str
    observed_at: float
    captured_at: float
    payload: Mapping[str, Any]
    provenance_uri: str | None = None
    provider_record_id: str | None = None
    schema_version: str = SCHEMA

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.record_type.strip():
            raise ValueError("provider and record_type are required")
        if not math.isfinite(float(self.observed_at)) or not math.isfinite(float(self.captured_at)):
            raise ValueError("capture timestamps must be finite")
        if self.captured_at < self.observed_at:
            raise ValueError("captured_at cannot precede observed_at")
        _canonical(self.payload)

    @property
    def payload_hash(self) -> str:
        return _hash(self.payload)

    @property
    def capture_id(self) -> str:
        return _hash({
            "schema_version": self.schema_version,
            "provider": self.provider,
            "record_type": self.record_type,
            "observed_at": float(self.observed_at),
            "captured_at": float(self.captured_at),
            "payload_hash": self.payload_hash,
            "provenance_uri": self.provenance_uri,
            "provider_record_id": self.provider_record_id,
        })

    def manifest(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "payload": _canonical(self.payload),
            "payload_hash": self.payload_hash,
            "capture_id": self.capture_id,
        }


class IntelligenceStore:
    """Immutable point-in-time capture store.

    One content-addressed JSON file per observation avoids accidental rewrites and
    makes future replay provenance explicit. This is a research data store only;
    it has no execution surface.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _target(self, capture: IntelligenceCapture) -> Path:
        return self.root / "captures" / capture.provider / capture.record_type / f"{capture.capture_id}.json"

    def put(self, capture: IntelligenceCapture) -> Path:
        target = self._target(capture)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = _json_bytes(capture.manifest())
        if target.exists():
            if target.read_bytes() != content:
                raise FileExistsError("immutable intelligence capture conflict")
            return target
        descriptor, temporary = tempfile.mkstemp(prefix=".intel-", suffix=".json", dir=target.parent)
        os.close(descriptor)
        try:
            Path(temporary).write_bytes(content)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target

    def read(self, capture_id: str, *, provider: str, record_type: str) -> dict[str, Any]:
        if not capture_id.strip() or not provider.strip() or not record_type.strip():
            raise ValueError("capture/provider/record_type are required")
        target = self.root / "captures" / provider / record_type / f"{capture_id}.json"
        return json.loads(target.read_text(encoding="utf-8"))
