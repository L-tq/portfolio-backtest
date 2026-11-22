from __future__ import annotations

import numpy as np
import pytest

from portfolio_backtest.engine import (
    OutputMemoryError,
    TargetSlotOverflowError,
    execute_backtest,
)
from portfolio_backtest.kernel import kernel_signature_summary

from tests.conftest import make_request


def test_kernel_is_nopython_and_parallel() -> None:
    request = make_request([[10.0], [10.0]], [[1], [0]])
    execute_backtest(request)
    summary = kernel_signature_summary()
    assert summary["compiled"]
    assert summary["nopython"]
    assert summary["parallel"]


def test_lag_on_off_and_last_bar_ignored() -> None:
    lag_on = execute_backtest(
        make_request([[10.0], [10.0], [10.0]], [[1], [0], [0]], execution_lag_bars=1)
    )
    assert list(lag_on.full_window.trades["date"].dt.day) == [3, 4]

    lag_off = execute_backtest(
        make_request([[10.0], [10.0], [10.0]], [[1], [0], [0]], execution_lag_bars=0)
    )
    assert list(lag_off.full_window.trades["date"].dt.day) == [2, 3]

    ignored = execute_backtest(
        make_request([[10.0], [10.0]], [[0], [1]], execution_lag_bars=1)
    )
    assert ignored.full_window.trades.empty


def test_target_overflow_aborts_before_results() -> None:
    request = make_request(
        [[10.0, 10.0], [10.0, 10.0]],
        [[1, 1], [0, 0]],
        target_n=1,
    )
    with pytest.raises(TargetSlotOverflowError, match="requests 2 directions"):
        execute_backtest(request)


def test_memory_guard_rejects_oversized_output_buffers() -> None:
    request = make_request([[10.0], [10.0]], [[1], [0]])
    request.config.memory_limit_bytes = 1
    with pytest.raises(OutputMemoryError, match="output buffers"):
        execute_backtest(request)


def test_held_duplicate_signal_does_not_fill() -> None:
    result = execute_backtest(
        make_request([[10.0], [10.0], [10.0]], [[1], [1], [1]])
    )
    assert len(result.full_window.trades) == 1
    assert result.full_window.trades.iloc[0]["action"] == "open"


def test_flip_produces_separate_close_and_open() -> None:
    result = execute_backtest(
        make_request([[10.0], [10.0], [10.0]], [[1], [-1], [-1]])
    )
    trades = result.full_window.trades
    assert len(trades) == 3
    assert list(trades.iloc[1:]["action"]) == ["close", "open"]
    assert list(trades.iloc[1:]["direction"]) == ["long", "short"]


def test_exits_precede_entries_and_reuse_settled_cash() -> None:
    result = execute_backtest(
        make_request(
            [[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]],
            [[1, 0], [0, 1], [0, 1]],
        )
    )
    trades = result.full_window.trades
    assert list(trades["action"]) == ["open", "close", "open"]
    assert list(trades["symbol"]) == ["S0", "S0", "S1"]
    assert trades.iloc[-1]["cash_after"] == pytest.approx(0.0)


def test_insufficient_signals_leave_residual_cash() -> None:
    result = execute_backtest(
        make_request(
            [[10.0, 10.0], [10.0, 10.0]],
            [[1, 0], [1, 0]],
            target_n=2,
        )
    )
    assert result.full_window.trades.iloc[0]["notional"] == pytest.approx(500.0)
    assert result.full_window.equity_curve.iloc[-1]["cash"] == pytest.approx(500.0)
    assert result.full_window.equity_curve.iloc[-1]["equity"] == pytest.approx(1000.0)


def test_short_proceeds_are_not_spendable() -> None:
    result = execute_backtest(
        make_request(
            [[10.0], [10.0], [10.0]],
            [[-1], [0], [0]],
        )
    )
    trades = result.full_window.trades
    assert trades.iloc[0]["cash_after"] == pytest.approx(0.0)
    assert result.full_window.equity_curve.iloc[1]["equity"] == pytest.approx(1000.0)
    assert trades.iloc[1]["cash_after"] == pytest.approx(1000.0)


def test_borrow_accrues_only_in_short_bucket() -> None:
    result = execute_backtest(
        make_request(
            [[10.0], [10.0], [10.0]],
            [[-1], [0], [0]],
            borrow_rate=0.252,
        )
    )
    assert result.full_window.equity_curve.iloc[-1]["equity"] == pytest.approx(999.0)
    assert result.full_window.trades.iloc[-1]["realized_pnl"] == pytest.approx(-1.0)


def test_margin_gap_open_buy_in() -> None:
    result = execute_backtest(
        make_request(
            [[10.0], [20.0]],
            [[-1], [0]],
            maintenance_margin=0.3,
        )
    )
    margin_fill = result.full_window.trades.iloc[-1]
    assert margin_fill["reason"] == "margin"
    assert margin_fill["price"] == pytest.approx(20.0)
    assert result.full_window.equity_curve.iloc[-1]["equity"] == pytest.approx(0.0)


def test_long_delisting_haircut_marks_and_closes() -> None:
    result = execute_backtest(
        make_request(
            [[10.0], [12.0]],
            [[1], [1]],
            delisting_haircut=0.5,
            delisting=[[False], [True]],
        )
    )
    trade = result.full_window.trades.iloc[-1]
    assert trade["reason"] == "delisting"
    assert trade["price"] == pytest.approx(6.0)
    assert result.full_window.equity_curve.iloc[-1]["equity"] == pytest.approx(600.0)


def test_short_delisting_haircut_uses_cut_cover_price() -> None:
    result = execute_backtest(
        make_request(
            [[10.0], [12.0]],
            [[-1], [-1]],
            delisting_haircut=0.5,
            delisting=[[False], [True]],
        )
    )
    trade = result.full_window.trades.iloc[-1]
    assert trade["direction"] == "short"
    assert trade["reason"] == "delisting"
    assert trade["price"] == pytest.approx(6.0)
    assert trade["realized_pnl"] == pytest.approx(400.0)
    assert result.full_window.equity_curve.iloc[-1]["equity"] == pytest.approx(1400.0)


@pytest.mark.parametrize("settlement_days", [0, 1, 2])
def test_settlement_release_timing(settlement_days: int) -> None:
    result = execute_backtest(
        make_request(
            [[10.0], [10.0], [10.0], [10.0], [10.0]],
            [[1], [0], [0], [0], [0]],
            settlement_days=settlement_days,
        )
    )
    cash = result.full_window.equity_curve["cash"].to_list()
    assert cash[0] == pytest.approx(0.0)
    assert cash[1] == pytest.approx(0.0 if settlement_days > 0 else 1000.0)
    if settlement_days > 0:
        assert cash[1 + settlement_days] == pytest.approx(1000.0)
        assert all(value == pytest.approx(0.0) for value in cash[1 : 1 + settlement_days])


def test_short_settlement_releases_bucket_cash_later() -> None:
    result = execute_backtest(
        make_request(
            [[10.0], [10.0], [10.0]],
            [[-1], [0], [0]],
            settlement_days=1,
        )
    )
    cash = result.full_window.equity_curve["cash"].to_list()
    assert cash[0] == pytest.approx(0.0)
    assert cash[1] == pytest.approx(0.0)
    assert cash[2] == pytest.approx(1000.0)
    assert result.full_window.equity_curve.iloc[-1]["equity"] == pytest.approx(1000.0)


def test_benchmark_relative_metrics_and_explicit_nulls() -> None:
    with_benchmark = execute_backtest(
        make_request(
            [[10.0], [11.0], [12.0], [11.0]],
            [[1], [1], [1], [1]],
            benchmark_values=[100.0, 101.0, 99.0, 102.0],
        )
    ).full_window.metrics
    without_benchmark = execute_backtest(
        make_request(
            [[10.0], [11.0], [12.0], [11.0]],
            [[1], [1], [1], [1]],
        )
    ).full_window.metrics
    assert with_benchmark["beta"] is not None
    assert with_benchmark["correlation"] is not None
    assert without_benchmark["beta"] is None
    assert without_benchmark["information_ratio"] is None


def test_short_benchmark_does_not_fill_trailing_zero_returns() -> None:
    result = execute_backtest(
        make_request(
            [[10.0], [11.0], [12.0], [11.0]],
            [[1], [1], [1], [1]],
            benchmark_values=[100.0, 101.0],
            benchmark_bars=2,
        )
    ).full_window
    curve = result.equity_curve

    assert curve["benchmark"].isna().tolist() == [False, False, True, True]
    assert curve["benchmark_return"].isna().tolist() == [True, False, True, True]
    for name in ("alpha", "beta", "correlation", "tracking_error", "information_ratio"):
        assert result.metrics[name] is None
