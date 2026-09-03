# Brian Phase 3.7 — Prospective Live Shadow

Phase 3.7 starts a forward-only development experiment from deployment time. It does not replay historical outcomes and it does not use 2026 as a pristine holdout. The previously contaminated 2026 historical holdout remains invalid.

The frozen Phase 3.5 learner from successful run `33766345728` is pinned by raw state ID `de90c35af3525d591f17e2489e64e9c5ebd84f8124e344927d7c829623688d36` and portable fingerprint `b534b611543fcf449a371faad208be20ccf7782343996d08b2bd554ed7f720b9`. Learning is disabled during this experiment.

Every five minutes the collector reads only public Binance Spot data for BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT and XRPUSDT. Closed 5-minute candles provide the causal feature context. Current `bookTicker` bid/ask quotes provide the forward observation price and real top-of-book spread. The first scored portfolio state starts at $500 at the first successful live capture; earlier prices are used only as visible lookback context and do not generate backfilled PnL.

Two frozen policies run side by side: the native counterfactual learner allocation rule and `PROFIT_SEEKING_SHADOW`. Portfolio PnL is marked from one actually observed live mid-price snapshot to the next. Rebalancing costs use the configured 10 bps fee plus 1 bps slippage plus half of the actually observed spread. There is no leverage, broker, exchange credential, order API, automatic promotion or self-modifying code.

Raw market responses are compressed and stored in the private intelligence bucket before the derived tick is persisted. Experiment and tick records are RLS-protected and append-only. The experiment is evidence-classed `PROSPECTIVE_DEVELOPMENT_SHADOW`, not final-holdout evidence.

The first formal review is preregistered for no earlier than seven elapsed calendar days and at least 20 active profit-policy ticks. Until those minimums are satisfied, interim results are descriptive only. Passing any future development gate still cannot enable live execution automatically.
