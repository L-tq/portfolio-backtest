from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtest.models import BacktestConfig, BacktestRequest, PricePanel, SignalPanel


def make_request(
    closes: list[list[float]],
    signals: list[list[int]],
    *,
    initial_cash: float = 1000.0,
    target_n: int = 1,
    rebalance_days: int = 1,
    execution_lag_bars: int = 0,
    commission_rate: float = 0.0,
    borrow_rate: float = 0.0,
    maintenance_margin: float = 0.3,
    delisting_haircut: float = 1.0,
    settlement_days: int = 0,
    mode: str = "full",
    delisting: list[list[bool]] | None = None,
    benchmark_values: list[float] | None = None,
) -> BacktestRequest:
    close = np.asarray(closes, dtype=np.float64)
    signal = np.asarray(signals, dtype=np.int8)
    if close.ndim != 2 or signal.shape != close.shape:
        raise ValueError("test panels must be rectangular and aligned")
    dates = pd.bdate_range("2024-01-02", periods=len(close)).to_numpy(dtype="datetime64[D]")
    prices = PricePanel(
        dates=dates,
        symbols=tuple(f"S{index}" for index in range(close.shape[1])),
        open=close,
        high=close * 1.02,
        low=close * 0.98,
        close=close,
        delisting_mask=(
            np.asarray(delisting, dtype=np.bool_)
            if delisting is not None
            else np.zeros_like(close, dtype=np.bool_)
        ),
    )
    signal_panel = SignalPanel(
        dates=dates,
        symbols=prices.symbols,
        signals=signal,
    )
    benchmark = None
    if benchmark_values is not None:
        from portfolio_backtest.models import BenchmarkSeries

        benchmark = BenchmarkSeries(
            dates=dates,
            values=np.asarray(benchmark_values, dtype=np.float64),
            ticker="TEST",
        )
    return BacktestRequest(
        prices=prices,
        signals=signal_panel,
        config=BacktestConfig(
            initial_cash=initial_cash,
            target_n=target_n,
            rebalance_days=rebalance_days,
            execution_lag_bars=execution_lag_bars,
            commission_rate=commission_rate,
            borrow_rate=borrow_rate,
            maintenance_margin=maintenance_margin,
            delisting_haircut=delisting_haircut,
            settlement_days=settlement_days,
            mode=mode,
        ),
        benchmark=benchmark,
    )


@pytest.fixture
def simple_request() -> BacktestRequest:
    return make_request(
        [[10.0, 20.0], [10.0, 20.0], [11.0, 19.0], [9.0, 21.0]],
        [[1, -1], [1, -1], [0, 0], [0, 0]],
        target_n=2,
        commission_rate=0.001,
        settlement_days=1,
        benchmark_values=[100.0, 101.0, 99.0, 103.0],
    )
