from __future__ import annotations

from hashlib import sha256
from typing import Mapping, Sequence
import json
import math

PORTABLE_STATE_FINGERPRINT_VERSION = "brian.portable-state-fingerprint.v1"
PORTABLE_SIGNIFICANT_DIGITS = 14


def _normalize_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("portable state fingerprint requires finite floats")
    if value == 0.0:
        return 0.0
    return float(format(value, f".{PORTABLE_SIGNIFICANT_DIGITS}g"))


def _normalize(value: object) -> object:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return _normalize_float(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    raise TypeError(f"unsupported portable fingerprint value: {type(value)!r}")


def portable_state_payload(state: Mapping[str, object]) -> dict[str, object]:
    """Return a cross-runtime audit representation without mutating raw checkpoint values.

    NumPy/BLAS implementations can disagree in the last few floating-point bits while
    producing materially identical model state. The portable audit payload normalizes only
    those sub-machine-scale tails. The checkpoint itself remains full precision.
    """

    normalized = _normalize(dict(state))
    if not isinstance(normalized, dict):
        raise TypeError("portable state payload must remain a mapping")
    return {
        "fingerprint_version": PORTABLE_STATE_FINGERPRINT_VERSION,
        "significant_digits": PORTABLE_SIGNIFICANT_DIGITS,
        "state": normalized,
    }


def portable_state_fingerprint(state: Mapping[str, object]) -> str:
    payload = portable_state_payload(state)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return sha256(text.encode("utf-8")).hexdigest()
