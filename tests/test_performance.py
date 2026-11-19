from __future__ import annotations

import os
import time

import numpy as np
import pytest

from portfolio_backtest.kernel import simulate_windows


def _simulate(
    prices: np.ndarray,
    signals: np.ndarray,
    windows: np.ndarray,
    max_fills: int,
) -> tuple:
    return simulate_windows(
        prices,
        prices,
        prices,
        prices,
        np.zeros_like(signals, dtype=np.bool_),
        signals,
        windows,
        max_fills,
        1_000_000.0,
        100,
        5,
        1,
        0.001,
        0.10,
        0.30,
        1.0,
        0,
    )


@pytest.mark.skipif(
    os.environ.get("RUN_PERFORMANCE_TESTS") != "1",
    reason="set RUN_PERFORMANCE_TESTS=1 to run the 2,520 x 3,000 benchmark",
)
def test_full_and_parallel_walk_forward_performance() -> None:
    bars = 2_520
    instruments = 3_000
    target = 100
    rng = np.random.default_rng(20261120)
    prices = 20.0 + np.cumsum(rng.normal(0.0, 0.02, (bars, instruments)), axis=0)
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    signals = np.zeros((bars, instruments), dtype=np.int8)
    for bar in range(0, bars, 5):
        chosen = rng.choice(instruments, size=target, replace=False)
        direction = 1 if (bar // 5) % 2 == 0 else -1
        signals[bar, chosen] = direction

    warmup_prices = np.ascontiguousarray(np.full((12, instruments), 20.0))
    warmup_signals = np.zeros((12, instruments), dtype=np.int8)
    warmup_signals[0, :target] = 1
    warmup_signals[5, :target] = -1
    _simulate(
        warmup_prices,
        warmup_signals,
        np.array([[0, 11]], dtype=np.int64),
        500,
    )

    full_start = time.perf_counter()
    _simulate(
        prices,
        signals,
        np.array([[0, bars - 1]], dtype=np.int64),
        10_000,
    )
    full_elapsed = time.perf_counter() - full_start
    assert full_elapsed < 15.0, f"full-window core took {full_elapsed:.3f}s"

    edges = np.linspace(0, bars, 5, dtype=np.int64)
    windows = np.column_stack((edges[:-1], edges[1:] - 1)).astype(np.int64)
    walk_start = time.perf_counter()
    _simulate(prices, signals, windows, 10_000)
    walk_elapsed = time.perf_counter() - walk_start
    assert walk_elapsed < 15.0, f"parallel walk-forward core took {walk_elapsed:.3f}s"
