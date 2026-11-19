from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_backtest.engine import execute_backtest
from tests.golden_scenario import golden_request


FIXTURES = Path(__file__).parent / "fixtures" / "golden"


def assert_frame_close(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert list(actual.columns) == list(expected.columns)
    assert len(actual) == len(expected)
    for column in actual.columns:
        if column in {"date", "symbol", "direction", "action", "reason"}:
            assert actual[column].astype(str).to_list() == expected[column].astype(str).to_list()
        else:
            np.testing.assert_allclose(
                actual[column].to_numpy(dtype=float),
                expected[column].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-8,
            )


def assert_metrics_close(actual, expected) -> None:
    assert set(actual) == set(expected)
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            assert_metrics_close(actual_value, expected_value)
        elif expected_value is None:
            assert actual_value is None
        else:
            assert actual_value == pytest.approx(expected_value, rel=1e-10, abs=1e-8)


def test_golden_master() -> None:
    scope = execute_backtest(golden_request()).full_window
    assert scope is not None

    expected_equity = pd.read_csv(FIXTURES / "equity.csv", parse_dates=["date"])
    actual_equity = scope.equity_curve[
        ["date", "equity", "cash", "long_exposure", "short_exposure", "drawdown"]
    ].reset_index(drop=True)
    assert_frame_close(actual_equity, expected_equity)

    expected_fills = pd.read_csv(FIXTURES / "fills.csv", parse_dates=["date"])
    actual_fills = scope.trades[
        [
            "date",
            "symbol",
            "direction",
            "action",
            "quantity",
            "price",
            "notional",
            "commission",
            "realized_pnl",
            "reason",
        ]
    ].reset_index(drop=True)
    assert_frame_close(actual_fills, expected_fills)

    expected_positions = pd.read_csv(FIXTURES / "positions.csv", parse_dates=["date"])
    actual_positions = scope.positions[["date", "symbol", "shares"]].reset_index(drop=True)
    assert_frame_close(actual_positions, expected_positions)

    expected_metrics = json.loads((FIXTURES / "metrics.json").read_text(encoding="utf-8"))
    assert_metrics_close(scope.metrics, expected_metrics)


def test_repeated_runs_are_deterministic() -> None:
    first = execute_backtest(golden_request()).full_window
    second = execute_backtest(golden_request()).full_window
    assert first is not None and second is not None
    pd.testing.assert_frame_equal(first.equity_curve, second.equity_curve)
    pd.testing.assert_frame_equal(first.trades, second.trades)
    pd.testing.assert_frame_equal(first.positions, second.positions)
