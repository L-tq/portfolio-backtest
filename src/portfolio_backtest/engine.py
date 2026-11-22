from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .kernel import FILL_COLUMNS, simulate_windows
from .metrics import METRIC_NAMES, aggregate_walk_forward, calculate_metrics
from .models import (
    BacktestRequest,
    BacktestResult,
    BenchmarkSeries,
    ScopeResult,
    WalkForwardSummary,
)


class TargetSlotOverflowError(ValueError):
    """Raised when a signal bar requests more directions than target_n."""


class OutputMemoryError(MemoryError):
    """Raised when preallocated kernel buffers exceed the configured guard."""


@dataclass(frozen=True)
class _WindowSpec:
    name: str
    start: int
    stop: int
    start_date: date
    end_date: date


def _execution_dates(start: int, stop: int, request: BacktestRequest) -> list[int]:
    config = request.config
    dates = []
    signal_bar = start
    while signal_bar <= stop:
        execution_bar = signal_bar + config.execution_lag_bars
        if execution_bar <= stop:
            dates.append(execution_bar)
        signal_bar += config.rebalance_days
    return dates


def validate_target_slots(request: BacktestRequest, windows: tuple[_WindowSpec, ...]) -> None:
    target = request.config.target_n
    for window in windows:
        for bar in _execution_dates(window.start, window.stop, request):
            signal_bar = bar - request.config.execution_lag_bars
            desired = int(np.count_nonzero(request.signals.signals[signal_bar]))
            if desired > target:
                date_value = pd.Timestamp(request.prices.dates[bar]).date()
                raise TargetSlotOverflowError(
                    f"{window.name} bar {date_value} requests {desired} directions "
                    f"but target_n is {target}"
                )


def _memory_estimate(
    window_count: int,
    max_length: int,
    instrument_count: int,
    max_fills: int,
    settlement_days: int,
) -> int:
    position_bytes = window_count * max_length * instrument_count * 8
    curve_bytes = window_count * max_length * 8 * 9
    fill_bytes = window_count * max_fills * 11 * 8
    pending_bytes = window_count * (max_length + settlement_days + 2) * 8
    state_bytes = window_count * instrument_count * 8 * 4
    return int(position_bytes + curve_bytes + fill_bytes + pending_bytes + state_bytes)


def _prepare_windows(request: BacktestRequest) -> tuple[_WindowSpec, ...]:
    mode = request.config.mode
    price_dates = pd.DatetimeIndex(request.prices.dates)
    windows: list[_WindowSpec] = []
    if mode in {"full", "both"}:
        windows.append(
            _WindowSpec(
                name="full",
                start=0,
                stop=len(request.prices.dates) - 1,
                start_date=price_dates[0].date(),
                end_date=price_dates[-1].date(),
            )
        )
    if mode in {"walk_forward", "both"}:
        if not request.walk_forward:
            if mode == "walk_forward":
                raise ValueError("walk-forward mode requires at least one window")
        for index, window in enumerate(request.walk_forward, start=1):
            mask = (price_dates >= pd.Timestamp(window.start_date)) & (
                price_dates <= pd.Timestamp(window.end_date)
            )
            indices = np.flatnonzero(mask)
            if len(indices) < 2:
                raise ValueError(
                    f"walk-forward window {window.start_date} to {window.end_date} "
                    "contains fewer than two price observations"
                )
            windows.append(
                _WindowSpec(
                    name=f"wf-{index:03d}",
                    start=int(indices[0]),
                    stop=int(indices[-1]),
                    start_date=window.start_date,
                    end_date=window.end_date,
                )
            )
    return tuple(windows)


def _align_benchmark(
    benchmark: BenchmarkSeries | None,
    dates: pd.DatetimeIndex,
) -> np.ndarray | None:
    if benchmark is None:
        return None
    series = pd.Series(
        benchmark.values,
        index=pd.DatetimeIndex(benchmark.dates, name="date"),
        dtype=float,
    ).sort_index()
    aligned = series.reindex(dates, method="ffill")
    # Forward-fill internal observations, but do not extend a stale benchmark
    # beyond its final source date.
    aligned = aligned.mask(aligned.index > series.index[-1])
    if pd.isna(aligned.iloc[0]):
        first = series.index[0].date()
        raise ValueError(
            f"benchmark has no observation on or before the first backtest date; "
            f"first benchmark date is {first}"
        )
    return aligned.to_numpy(dtype=np.float64)


def _build_scope(
    *,
    window: _WindowSpec,
    window_id: int,
    request: BacktestRequest,
    raw: tuple,
    benchmark: np.ndarray | None,
) -> ScopeResult:
    (
        equity_raw,
        cash_raw,
        long_raw,
        short_raw,
        position_raw,
        fills_raw,
        fill_counts,
        _,
    ) = raw
    length = window.stop - window.start + 1
    dates = pd.DatetimeIndex(request.prices.dates[window.start : window.stop + 1])
    equity = pd.Series(equity_raw[window_id, :length], index=dates)
    cash = pd.Series(cash_raw[window_id, :length], index=dates)
    long_exposure = pd.Series(long_raw[window_id, :length], index=dates)
    short_exposure = pd.Series(short_raw[window_id, :length], index=dates)
    simple_returns = equity.pct_change(fill_method=None).fillna(0.0)
    drawdown = equity.div(equity.cummax()).sub(1.0)

    benchmark_values = None
    benchmark_returns = None
    if benchmark is not None:
        benchmark_values = pd.Series(benchmark[window.start : window.stop + 1], index=dates)
        benchmark_returns = benchmark_values.pct_change(fill_method=None)

    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "equity": equity.to_numpy(),
            "cash": cash.to_numpy(),
            "long_exposure": long_exposure.to_numpy(),
            "short_exposure": short_exposure.to_numpy(),
            "return": simple_returns.to_numpy(),
            "drawdown": drawdown.to_numpy(),
            "benchmark": benchmark_values.to_numpy() if benchmark_values is not None else np.nan,
            "benchmark_return": (
                benchmark_returns.to_numpy() if benchmark_returns is not None else np.nan
            ),
        }
    )

    count = int(fill_counts[window_id])
    fills = fills_raw[window_id, :count]
    trades = pd.DataFrame(fills, columns=FILL_COLUMNS)
    if trades.empty:
        trades = pd.DataFrame(
            columns=(
                "bar_index",
                "instrument",
                "symbol",
                "date",
                "direction",
                "action",
                "quantity",
                "price",
                "notional",
                "commission",
                "cash_after",
                "realized_pnl",
                "reason",
            )
        )
    else:
        trades["bar_index"] = trades["bar_index"].astype(int)
        trades["instrument"] = trades["instrument"].astype(int)
        trades["symbol"] = trades["instrument"].map(
            dict(enumerate(request.prices.symbols))
        )
        trades["date"] = dates[trades["bar_index"].to_numpy()]
        trades["direction"] = np.where(trades["direction"] > 0, "long", "short")
        trades["action"] = np.where(trades["action"] > 0, "open", "close")
        trades["reason"] = trades["reason"].map({0: "signal", 1: "margin", 2: "delisting"})
        trades = trades[
            [
                "date",
                "symbol",
                "direction",
                "action",
                "quantity",
                "price",
                "notional",
                "commission",
                "cash_after",
                "realized_pnl",
                "reason",
            ]
        ].reset_index(drop=True)

    snapshots = position_raw[window_id, :length]
    row_indices, instrument_indices = np.nonzero(snapshots)
    if len(row_indices):
        close_prices = request.prices.close[window.start : window.stop + 1]
        positions = pd.DataFrame(
            {
                "date": dates[row_indices],
                "symbol": [request.prices.symbols[index] for index in instrument_indices],
                "shares": snapshots[row_indices, instrument_indices],
                "close": close_prices[row_indices, instrument_indices],
            }
        )
        positions["direction"] = np.where(positions["shares"] > 0, "long", "short")
        positions["market_value"] = positions["shares"] * positions["close"]
    else:
        positions = pd.DataFrame(columns=["date", "symbol", "shares", "close", "direction", "market_value"])

    metrics = calculate_metrics(
        equity_curve,
        trades,
        annualization=request.config.annualization,
        risk_free_rate=request.config.risk_free_rate,
        initial_cash=request.config.initial_cash,
    )
    return ScopeResult(
        scope=window.name,
        start_date=window.start_date,
        end_date=window.end_date,
        equity_curve=equity_curve,
        positions=positions.reset_index(drop=True),
        trades=trades,
        metrics=metrics,
    )


def execute_backtest(
    request: BacktestRequest,
    *,
    run_id: str | None = None,
    warmup: bool = True,
) -> BacktestResult:
    windows = _prepare_windows(request)
    validate_target_slots(request, windows)
    arrays = [
        request.prices.open,
        request.prices.high,
        request.prices.low,
        request.prices.close,
        request.prices.delisting_mask,
        request.signals.signals,
    ]
    window_array = np.ascontiguousarray(
        np.array([[item.start, item.stop] for item in windows], dtype=np.int64)
    )
    lengths = window_array[:, 1] - window_array[:, 0] + 1
    max_length = int(lengths.max())
    max_fills = (
        max_length * (3 * request.config.target_n + 2)
        + request.prices.close.shape[1] * 2
    )
    estimate = _memory_estimate(
        len(windows),
        max_length,
        request.prices.close.shape[1],
        max_fills,
        request.config.settlement_days,
    )
    if estimate > request.config.memory_limit_bytes:
        raise OutputMemoryError(
            f"estimated output buffers require {estimate:,} bytes, above the "
            f"{request.config.memory_limit_bytes:,}-byte guard"
        )

    raw = simulate_windows(
        *arrays,
        window_array,
        int(max_fills),
        float(request.config.initial_cash),
        int(request.config.target_n),
        int(request.config.rebalance_days),
        int(request.config.execution_lag_bars),
        float(request.config.commission_rate),
        float(request.config.borrow_rate),
        float(request.config.maintenance_margin),
        float(request.config.delisting_haircut),
        int(request.config.settlement_days),
    )

    dates = pd.DatetimeIndex(request.prices.dates)
    benchmark = _align_benchmark(request.benchmark, dates)
    scopes = [
        _build_scope(
            window=window,
            window_id=index,
            request=request,
            raw=raw,
            benchmark=benchmark,
        )
        for index, window in enumerate(windows)
    ]
    full_window = next((scope for scope in scopes if scope.scope == "full"), None)
    walk_forward = tuple(scope for scope in scopes if scope.scope != "full")
    summary = None
    if walk_forward:
        aggregate = aggregate_walk_forward([scope.metrics for scope in walk_forward])
        summary = WalkForwardSummary(**aggregate)

    return BacktestResult(
        run_id=run_id or uuid.uuid4().hex,
        config=request.config,
        full_window=full_window,
        walk_forward=walk_forward,
        walk_forward_summary=summary,
        benchmark=request.benchmark,
    )
