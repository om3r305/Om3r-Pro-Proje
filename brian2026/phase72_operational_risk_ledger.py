from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

from .evidence_ledger import content_hash
from .phase56_pretrade_risk_engine import TradingState
from .phase68_operational_risk_governor import (
    OperationalRiskGovernor,
    OperationalRiskPolicy,
    OperationalRiskReceipt,
)

PHASE72_SCHEMA_VERSION = "brian.phase72-operational-risk-ledger.v1"


class OperationalRiskLedgerError(ValueError):
    pass


def _policy_hash(policy: OperationalRiskPolicy) -> str:
    return content_hash({
        "schema_version": PHASE72_SCHEMA_VERSION,
        "policy": asdict(policy),
    })


def _receipt_from_dict(payload: Mapping[str, object]) -> OperationalRiskReceipt:
    raw = dict(payload)
    try:
        receipt = OperationalRiskReceipt(
            timestamp=float(raw["timestamp"]),
            previous_state=str(raw["previous_state"]),  # type: ignore[arg-type]
            trading_state=str(raw["trading_state"]),  # type: ignore[arg-type]
            recommended_state=str(raw["recommended_state"]),  # type: ignore[arg-type]
            reasons=tuple(str(value) for value in raw["reasons"]),  # type: ignore[union-attr]
            max_drawdown_fraction=float(raw["max_drawdown_fraction"]),
            window_loss_fraction=float(raw["window_loss_fraction"]),
            qualifying_stoplosses=int(raw["qualifying_stoplosses"]),
            stoploss_lock_until=(
                None
                if raw.get("stoploss_lock_until") is None
                else float(raw["stoploss_lock_until"])
            ),
            blocked_assets=tuple(
                str(value) for value in raw["blocked_assets"]  # type: ignore[union-attr]
            ),
            consecutive_execution_failures=int(raw["consecutive_execution_failures"]),
            reconciliation_failures=int(raw["reconciliation_failures"]),
            unknown_order_outcomes=int(raw["unknown_order_outcomes"]),
            market_data_age_seconds=float(raw["market_data_age_seconds"]),
            manual_halt=bool(raw["manual_halt"]),
            manual_release_requested=bool(raw["manual_release_requested"]),
            halt_latched=bool(raw["halt_latched"]),
            receipt_id=str(raw["receipt_id"]),
            execution_failure_lock_until=(
                None
                if raw.get("execution_failure_lock_until") is None
                else float(raw["execution_failure_lock_until"])
            ),
            reconciliation_failure_lock_until=(
                None
                if raw.get("reconciliation_failure_lock_until") is None
                else float(raw["reconciliation_failure_lock_until"])
            ),
            asset_cooldown_until=tuple(
                (str(asset), float(until))
                for asset, until in raw.get("asset_cooldown_until", ())  # type: ignore[union-attr]
            ),
            schema_version=str(
                raw.get(
                    "schema_version",
                    "brian.phase68-operational-risk-governor.v1",
                )
            ),
            shadow_only=bool(raw.get("shadow_only", True)),
            live_execution=bool(raw.get("live_execution", False)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OperationalRiskLedgerError(
            f"invalid operational-risk receipt payload: {exc}"
        ) from exc
    if not receipt.verify_identity():
        raise OperationalRiskLedgerError(
            "operational-risk receipt content hash mismatch"
        )
    return receipt


@dataclass(frozen=True, slots=True)
class OperationalRiskLedgerEntry:
    sequence: int
    previous_entry_id: str | None
    policy_hash: str
    receipt: OperationalRiskReceipt
    schema_version: str = PHASE72_SCHEMA_VERSION
    entry_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("ledger sequence must be non-negative")
        if self.sequence == 0 and self.previous_entry_id is not None:
            raise ValueError("first ledger entry cannot have previous_entry_id")
        if self.sequence > 0 and (
            self.previous_entry_id is None or len(self.previous_entry_id) != 64
        ):
            raise ValueError("non-first ledger entry requires previous content id")
        if len(self.policy_hash) != 64:
            raise ValueError("policy_hash must be a content hash")
        if not self.receipt.verify_identity():
            raise ValueError("receipt content hash mismatch")
        object.__setattr__(
            self,
            "entry_id",
            content_hash(self.identity_payload()),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "previous_entry_id": self.previous_entry_id,
            "policy_hash": self.policy_hash,
            "receipt": self.receipt.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload["entry_id"] = self.entry_id
        return payload


@dataclass(frozen=True, slots=True)
class OperationalRiskAppendReceipt:
    entry_id: str
    receipt_id: str
    sequence: int
    duplicate: bool
    trading_state: TradingState
    halt_latched: bool


class OperationalRiskLedger:
    """Append-only restart state for the Phase68 operational risk governor."""

    def __init__(
        self,
        policy: OperationalRiskPolicy,
        *,
        initial_state: TradingState = "ACTIVE",
    ) -> None:
        if initial_state not in ("ACTIVE", "REDUCING", "HALTED"):
            raise ValueError("invalid initial_state")
        self.policy = policy
        self.policy_hash = _policy_hash(policy)
        self.initial_state = initial_state
        self._entries: list[OperationalRiskLedgerEntry] = []
        self._entry_by_receipt_id: dict[str, OperationalRiskLedgerEntry] = {}

    @property
    def entries(self) -> tuple[OperationalRiskLedgerEntry, ...]:
        return tuple(self._entries)

    @property
    def current_state(self) -> TradingState:
        if not self._entries:
            return self.initial_state
        return self._entries[-1].receipt.trading_state

    @property
    def halt_latched(self) -> bool:
        if not self._entries:
            return self.initial_state == "HALTED"
        return self._entries[-1].receipt.halt_latched

    def append(
        self,
        receipt: OperationalRiskReceipt,
    ) -> OperationalRiskAppendReceipt:
        if not receipt.verify_identity():
            raise OperationalRiskLedgerError(
                "operational-risk receipt content hash mismatch"
            )

        duplicate = self._entry_by_receipt_id.get(receipt.receipt_id)
        if duplicate is not None:
            if duplicate.receipt.to_dict() != receipt.to_dict():
                raise OperationalRiskLedgerError(
                    "receipt_id already exists with different evidence"
                )
            return OperationalRiskAppendReceipt(
                entry_id=duplicate.entry_id,
                receipt_id=receipt.receipt_id,
                sequence=duplicate.sequence,
                duplicate=True,
                trading_state=duplicate.receipt.trading_state,
                halt_latched=duplicate.receipt.halt_latched,
            )

        expected_previous = self.current_state
        if receipt.previous_state != expected_previous:
            raise OperationalRiskLedgerError(
                f"receipt previous_state {receipt.previous_state} does not match "
                f"ledger state {expected_previous}"
            )
        if self._entries and receipt.timestamp <= self._entries[-1].receipt.timestamp:
            raise OperationalRiskLedgerError(
                "distinct operational-risk receipts must advance timestamp"
            )

        entry = OperationalRiskLedgerEntry(
            sequence=len(self._entries),
            previous_entry_id=(
                None if not self._entries else self._entries[-1].entry_id
            ),
            policy_hash=self.policy_hash,
            receipt=receipt,
        )
        self._entries.append(entry)
        self._entry_by_receipt_id[receipt.receipt_id] = entry
        return OperationalRiskAppendReceipt(
            entry_id=entry.entry_id,
            receipt_id=receipt.receipt_id,
            sequence=entry.sequence,
            duplicate=False,
            trading_state=receipt.trading_state,
            halt_latched=receipt.halt_latched,
        )

    def governor(self) -> OperationalRiskGovernor:
        if not self._entries:
            return OperationalRiskGovernor(
                self.policy,
                initial_state=self.initial_state,
            )
        return OperationalRiskGovernor.from_receipt(
            self.policy,
            self._entries[-1].receipt,
        )

    def manifest(self) -> dict[str, object]:
        rows = [entry.to_dict() for entry in self._entries]
        ledger_hash = content_hash({
            "policy_hash": self.policy_hash,
            "initial_state": self.initial_state,
            "entries": rows,
        })
        return {
            "schema_version": PHASE72_SCHEMA_VERSION,
            "append_only": True,
            "policy": asdict(self.policy),
            "policy_hash": self.policy_hash,
            "initial_state": self.initial_state,
            "entries": rows,
            "entry_count": len(rows),
            "head_entry_id": None if not rows else rows[-1]["entry_id"],
            "current_state": self.current_state,
            "halt_latched": self.halt_latched,
            "ledger_hash": ledger_hash,
            "shadow_only": True,
            "live_execution": False,
        }


def restore_operational_risk_ledger(
    manifest: Mapping[str, object],
) -> OperationalRiskLedger:
    raw = dict(manifest)
    if raw.get("append_only") is not True:
        raise OperationalRiskLedgerError("operational-risk ledger is not append-only")
    if raw.get("shadow_only") is not True or raw.get("live_execution") is not False:
        raise OperationalRiskLedgerError(
            "operational-risk ledger crossed the live boundary"
        )

    policy_raw = raw.get("policy")
    entries_raw = raw.get("entries")
    if not isinstance(policy_raw, Mapping) or not isinstance(entries_raw, Sequence):
        raise OperationalRiskLedgerError(
            "operational-risk ledger is missing policy/entries"
        )
    try:
        policy_values = dict(policy_raw)
        # PostgreSQL jsonb can render integral floats without a decimal point.
        # Normalize the float-valued policy fields before recomputing identity.
        for numeric_key in (
            "max_drawdown_fraction",
            "max_daily_loss_fraction",
            "stoploss_required_profit",
            "max_market_data_age_seconds",
        ):
            if numeric_key in policy_values:
                policy_values[numeric_key] = float(policy_values[numeric_key])
        policy = OperationalRiskPolicy(**policy_values)
    except (TypeError, ValueError) as exc:
        raise OperationalRiskLedgerError(f"invalid operational-risk policy: {exc}") from exc

    expected_policy_hash = _policy_hash(policy)
    if raw.get("policy_hash") != expected_policy_hash:
        raise OperationalRiskLedgerError("operational-risk policy hash mismatch")

    initial_state = str(raw.get("initial_state", "ACTIVE"))
    if initial_state not in ("ACTIVE", "REDUCING", "HALTED"):
        raise OperationalRiskLedgerError("invalid operational-risk initial_state")
    ledger = OperationalRiskLedger(
        policy,
        initial_state=initial_state,  # type: ignore[arg-type]
    )

    previous_entry_id: str | None = None
    for sequence, raw_entry in enumerate(entries_raw):
        if not isinstance(raw_entry, Mapping):
            raise OperationalRiskLedgerError("invalid operational-risk ledger entry")
        entry_data = dict(raw_entry)
        if int(entry_data.get("sequence", -1)) != sequence:
            raise OperationalRiskLedgerError("operational-risk ledger sequence mismatch")
        if entry_data.get("previous_entry_id") != previous_entry_id:
            raise OperationalRiskLedgerError("operational-risk ledger hash chain is broken")
        if entry_data.get("policy_hash") != expected_policy_hash:
            raise OperationalRiskLedgerError("ledger entry policy hash mismatch")
        receipt_raw = entry_data.get("receipt")
        if not isinstance(receipt_raw, Mapping):
            raise OperationalRiskLedgerError("ledger entry receipt is missing")
        receipt = _receipt_from_dict(receipt_raw)
        appended = ledger.append(receipt)
        expected_entry_id = str(entry_data.get("entry_id", ""))
        if appended.entry_id != expected_entry_id:
            raise OperationalRiskLedgerError("operational-risk entry content hash mismatch")
        previous_entry_id = appended.entry_id

    rebuilt = ledger.manifest()
    if int(raw.get("entry_count", -1)) != len(ledger.entries):
        raise OperationalRiskLedgerError("operational-risk entry_count mismatch")
    for key in (
        "head_entry_id",
        "current_state",
        "halt_latched",
        "ledger_hash",
    ):
        if raw.get(key) != rebuilt.get(key):
            raise OperationalRiskLedgerError(
                f"operational-risk manifest {key} mismatch"
            )
    return ledger
