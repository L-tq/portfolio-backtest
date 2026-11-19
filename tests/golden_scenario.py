from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtest.models import (
    BacktestConfig,
    BacktestRequest,
    BenchmarkSeries,
    PricePanel,
    SignalPanel,
)


def golden_request() -> BacktestRequest:
    close = np.array(
        [
            [10.0, 20.0],
            [10.0, 20.0],
            [11.0, 19.0],
            [12.0, 18.0],
            [12.0, 17.0],
            [10.0, 18.0],
            [9.0, 19.0],
            [9.0, 20.0],
        ],
        dtype=np.float64,
    )
    signals = np.array(
        [
            [1, 1],
            [1, 1],
            [-1, -1],
            [-1, -1],
            [0, 0],
            [0, 0],
            [1, 0],
            [1, 0],
        ],
        dtype=np.int8,
    )
    dates = pd.bdate_range("2025-01-02", periods=len(close)).to_numpy(dtype="datetime64[D]")
    return BacktestRequest(
        prices=PricePanel(
            dates=dates,
            symbols=("ALPHA", "BETA"),
            open=close,
            high=close * 1.03,
            low=close * 0.97,
            close=close,
        ),
        signals=SignalPanel(
            dates=dates,
            symbols=("ALPHA", "BETA"),
            signals=signals,
        ),
        benchmark=BenchmarkSeries(
            dates=dates,
            values=np.array([100.0, 101.0, 100.5, 102.0, 101.0, 103.0, 104.0, 103.5]),
            ticker="SYNTH",
        ),
        config=BacktestConfig(
            initial_cash=10_000.0,
            target_n=2,
            rebalance_days=2,
            execution_lag_bars=1,
            commission_rate=0.001,
            borrow_rate=0.10,
            maintenance_margin=0.30,
            settlement_days=1,
            mode="full",
        ),
    )
