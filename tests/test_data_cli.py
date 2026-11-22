from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_backtest.data import (
    config_from_json,
    load_benchmark,
    load_price_panel,
    load_signal_panel,
)
from portfolio_backtest.engine import execute_backtest
from portfolio_backtest.models import BacktestConfig, BacktestRequest, SignalPanel


def _write_market_data(root: Path) -> None:
    for symbol, multiplier in (("S0", 1.0), ("S1", 2.0)):
        directory = root / symbol
        directory.mkdir(parents=True)
        pd.DataFrame(
            {
                "Date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "Open": [9.0, 10.0, 11.0],
                "High": [11.0, 12.0, 13.0],
                "Low": [8.0, 9.0, 10.0],
                "Close": [10.0, 11.0, 12.0],
                "Adjusted_close": [5.0 * multiplier, 5.5 * multiplier, 6.0 * multiplier],
            }
        ).to_csv(directory / "prices.csv", index=False)


def test_market_loader_applies_adjustment_and_signal_validation(tmp_path) -> None:
    _write_market_data(tmp_path)
    panel = load_price_panel(tmp_path, ["S0", "S1"])
    np.testing.assert_allclose(panel.open[:, 0], [4.5, 5.0, 5.5])
    np.testing.assert_allclose(panel.close[:, 1], [10.0, 11.0, 12.0])
    assert panel.delisting_mask.sum() == 0

    signal_file = tmp_path / "signals.csv"
    signal_file.write_text(
        "date,symbol,signal\n2024-01-02,S0,1\n2024-01-03,S1,-1\n",
        encoding="utf-8",
    )
    signals = load_signal_panel(signal_file, panel)
    assert signals.signals[0, 0] == 1
    assert signals.signals[1, 1] == -1
    assert np.count_nonzero(signals.signals) == 2

    bad_signal = tmp_path / "bad-signal.csv"
    bad_signal.write_text("date,symbol,signal\n2024-01-02,S2,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown symbols"):
        load_signal_panel(bad_signal, panel)


def test_delisting_inference_marks_only_the_shorter_source(tmp_path) -> None:
    _write_market_data(tmp_path)
    shorter = tmp_path / "S1" / "prices.csv"
    shorter.write_text(
        "Date,Open,High,Low,Close,Adjusted_close\n"
        "2024-01-02,18,22,17,20,40\n"
        "2024-01-03,18,22,17,20,40\n",
        encoding="utf-8",
    )
    panel = load_price_panel(tmp_path, ["S0", "S1"])
    assert not panel.delisting_mask[-1, 0]
    assert panel.delisting_mask[-1, 1]

    later_end = load_price_panel(tmp_path, ["S0", "S1"], end="2024-01-31")
    assert not later_end.delisting_mask[-1, 0]
    assert later_end.delisting_mask[-1, 1]

    truncated = load_price_panel(tmp_path, ["S0", "S1"], end="2024-01-03")
    assert not truncated.delisting_mask.any()


def test_end_after_data_horizon_does_not_delist_unchanged_sources(tmp_path) -> None:
    _write_market_data(tmp_path)
    panel = load_price_panel(tmp_path, ["S0", "S1"], end="2024-01-31")
    assert not panel.delisting_mask.any()


@pytest.mark.parametrize("value", [1.5, 257, -129])
def test_invalid_signal_values_are_rejected_before_int8_conversion(
    tmp_path, value: float | int
) -> None:
    _write_market_data(tmp_path)
    panel = load_price_panel(tmp_path, ["S0", "S1"])
    signal_file = tmp_path / "invalid-signal.csv"
    signal_file.write_text(
        f"date,symbol,signal\n2024-01-02,S0,{value}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="signals may only contain"):
        load_signal_panel(signal_file, panel)

    with pytest.raises(ValueError, match="signals may only contain"):
        SignalPanel(
            dates=panel.dates[:1],
            symbols=("S0",),
            signals=np.asarray([[value]]),
        )


def test_benchmark_file_and_config_precedence(tmp_path) -> None:
    benchmark_file = tmp_path / "benchmark.csv"
    benchmark_file.write_text(
        "Date,Close,Adjusted_close\n2024-01-02,100,90\n2024-01-03,101,91\n",
        encoding="utf-8",
    )
    benchmark = load_benchmark(file=benchmark_file)
    assert benchmark is not None
    np.testing.assert_allclose(benchmark.values, [90.0, 91.0])

    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "initial_cash": 1000,
                "target_n": 2,
                "rebalance_days": 3,
                "commission_rate": 0.002,
            }
        ),
        encoding="utf-8",
    )
    config = config_from_json(config_file, {"target_n": 1})
    assert config.target_n == 1
    assert config.rebalance_days == 3
    assert config.commission_rate == pytest.approx(0.002)
    with pytest.raises(ValueError):
        config_from_json(None, {})


def test_cli_run_persists_completed_result(tmp_path) -> None:
    _write_market_data(tmp_path / "data")
    signal_file = tmp_path / "signals.csv"
    signal_file.write_text(
        "date,symbol,signal\n"
        "2024-01-02,S0,1\n"
        "2024-01-02,S1,-1\n"
        "2024-01-03,S0,0\n"
        "2024-01-03,S1,0\n",
        encoding="utf-8",
    )
    database = tmp_path / "cli.db"
    results = tmp_path / "cli-results"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "portfolio_backtest",
            "run",
            "--data-dir",
            str(tmp_path / "data"),
            "--symbols",
            "S0",
            "S1",
            "--signals",
            str(signal_file),
            "--initial-cash",
            "1000",
            "--target-n",
            "2",
            "--mode",
            "full",
            "--no-benchmark",
            "--database",
            str(database),
            "--results-dir",
            str(results),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["full_window"] is True
    with sqlite3.connect(database) as connection:
        status = connection.execute(
            "SELECT status FROM runs WHERE id = ?", (payload["run_id"],)
        ).fetchone()[0]
    assert status == "completed"


def test_local_data_integration_smoke(tmp_path) -> None:
    data_dir = os.environ.get("BT_DATA_DIR")
    if not data_dir:
        pytest.skip("BT_DATA_DIR is unset; local market-data integration is optional")
    symbols = [
        symbol
        for symbol in ("AAPL.US", "TSLA.US", "AMZN.US")
        if (Path(data_dir) / symbol / "prices.csv").exists()
    ]
    if len(symbols) < 2:
        pytest.skip("fewer than two configured market-data symbols are available")
    panel = load_price_panel(data_dir, symbols[:2], start="2025-01-02", end="2025-02-28")
    first_date = pd.Timestamp(panel.dates[0]).strftime("%Y-%m-%d")
    second_date = pd.Timestamp(panel.dates[min(6, len(panel.dates) - 1)]).strftime("%Y-%m-%d")
    signal_file = tmp_path / "local-signals.csv"
    signal_file.write_text(
        f"date,symbol,signal\n{first_date},{panel.symbols[0]},1\n"
        f"{first_date},{panel.symbols[1]},-1\n"
        f"{second_date},{panel.symbols[0]},0\n"
        f"{second_date},{panel.symbols[1]},0\n",
        encoding="utf-8",
    )
    signals = load_signal_panel(signal_file, panel)
    request = execute_backtest(
        BacktestRequest(
            prices=panel,
            signals=signals,
            config=BacktestConfig(
                initial_cash=100_000,
                target_n=2,
                mode="full",
            ),
        )
    )
    assert request.full_window is not None
    assert len(request.full_window.equity_curve) == len(panel.dates)
