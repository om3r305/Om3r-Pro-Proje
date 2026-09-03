# Brian Phase 3.6 — Frozen Profit Exam

Phase 3.6 is a preregistered development exam for the exact learner checkpoint produced by successful Phase 3.5 run `33766345728`.

## Locked source learner

- Source artifact: `brian-phase35-training-33766345728`
- Episodes learned: 1,000
- Portable fingerprint: `b534b611543fcf449a371faad208be20ccf7782343996d08b2bd554ed7f720b9`
- Raw state id: `de90c35af3525d591f17e2489e64e9c5ebd84f8124e344927d7c829623688d36`
- The checkpoint is downloaded from the successful Phase 3.5 artifact; Phase 3.6 does not retrain it.

## Reserved development exam

The two months reserved before Phase 3.5 training are opened exactly once here:

- 2024-03
- 2024-04

Universe: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT at canonical 5m resolution from checksum-verified public Binance Vision monthly archives.

Learning is disabled for the entire exam. The learner raw state id and portable fingerprint must remain unchanged.

## Preregistered evaluation

Both frozen native policy and cost-aware `PROFIT_SEEKING_SHADOW` policy are evaluated. Profit mode is additionally tested under 1.0x, 1.5x and 2.0x simulated transaction-cost assumptions.

A `DEVELOPMENT_CANDIDATE` requires all of the following without post-hoc threshold changes:

- March net return > 0
- April net return > 0
- combined net return > 0
- combined net step profit factor > 1
- combined maximum drawdown <= 10%
- at least 20 combined active policy steps
- profit mode combined return is not worse than frozen native
- 1.5x transaction-cost stress remains net positive
- no virtual ruin

The 2.0x cost result is reported as an additional stress diagnostic but is not a candidate gate.

This is development evidence, not a pristine final holdout and not proof of profitability. 2026 remains permanently `INVALID_CONTAMINATED`. There is no automatic promotion, live exchange execution, credential use or self-modification.
