from brian2026.archive import ArchiveSpec, SUPPORTED_SYMBOLS
from brian2026.phase35_training import PHASE35_SYMBOLS


def test_phase35_symbols_are_supported_by_verified_archive_adapter() -> None:
    assert set(PHASE35_SYMBOLS).issubset(set(SUPPORTED_SYMBOLS))
    for symbol in PHASE35_SYMBOLS:
        spec = ArchiveSpec(symbol, "1m", 2023, 1)
        assert spec.symbol == symbol
        assert spec.timeframe == "1m"
