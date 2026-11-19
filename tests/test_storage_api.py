from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from portfolio_backtest.api import create_app
from portfolio_backtest.engine import execute_backtest
from portfolio_backtest.storage import (
    get_metrics,
    list_runs,
    load_backtest_result,
    load_scope_series,
    record_failure,
    run_and_persist,
    save_backtest,
    set_archive,
    start_run,
    trade_blotter,
)

from tests.conftest import make_request


def _completed_run(tmp_path, initial_cash: float = 1000.0) -> tuple[str, object, object]:
    request = make_request(
        [[10.0, 20.0], [10.0, 20.0], [11.0, 19.0], [9.0, 21.0]],
        [[1, -1], [1, -1], [0, 0], [0, 0]],
        initial_cash=initial_cash,
        target_n=2,
        commission_rate=0.001,
        settlement_days=1,
        benchmark_values=[100.0, 101.0, 99.0, 103.0],
    )
    database = tmp_path / "portfolio.db"
    results = tmp_path / "results"
    result = run_and_persist(request, database_path=database, results_dir=results)
    return result.run_id, database, results


def test_storage_round_trip_and_trade_queries(tmp_path) -> None:
    run_id, database, _ = _completed_run(tmp_path)
    loaded = load_backtest_result(run_id, database_path=database)
    assert loaded.run_id == run_id
    assert loaded.full_window is not None
    assert len(loaded.full_window.equity_curve) == 4
    assert len(load_scope_series(run_id, "full", database_path=database)) == 4
    blotter = trade_blotter(
        run_id,
        database_path=database,
        symbol="S0",
        page_size=1,
    )
    assert blotter["pages"] >= 1
    assert all(item["symbol"] == "S0" for item in blotter["items"])
    metrics = get_metrics(run_id, "full", database_path=database)
    assert "beta" in metrics
    assert metrics["long"]["details"]["count"] >= 1


def test_archive_filter_and_delete(tmp_path) -> None:
    run_id, database, results = _completed_run(tmp_path)
    assert list_runs(database_path=database)[0]["id"] == run_id
    set_archive(run_id, True, database_path=database)
    assert list_runs(database_path=database, archived=True)[0]["id"] == run_id
    assert list_runs(database_path=database, archived=False) == []

    client = TestClient(create_app(database, results))
    with client:
        denied = client.delete(f"/api/runs/{run_id}")
        assert denied.status_code == 409
        deleted = client.delete(f"/api/runs/{run_id}?confirm=true")
        assert deleted.status_code == 204
    assert list_runs(database_path=database) == []
    assert not (results / run_id).exists()


def test_atomic_storage_failure_exposes_no_results(tmp_path, monkeypatch) -> None:
    request = make_request(
        [[10.0], [10.0], [10.0]],
        [[1], [1], [0]],
        benchmark_values=[100.0, 101.0, 102.0],
    )
    database = tmp_path / "failed.db"
    results = tmp_path / "failed-results"
    run_id = start_run(request, database_path=database, results_dir=results)
    result = execute_backtest(request, run_id=run_id)

    def explode(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", explode)
    with pytest.raises(OSError, match="simulated disk failure"):
        save_backtest(result, request, database_path=database, results_dir=results)
    record_failure(
        run_id,
        OSError("simulated disk failure"),
        database_path=database,
        results_dir=results,
    )
    with sqlite3.connect(database) as connection:
        status = connection.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()[0]
        equity_rows = connection.execute(
            "SELECT COUNT(*) FROM equity_curve WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    assert status == "failed"
    assert equity_rows == 0


def test_slot_overflow_failure_is_persisted_without_completed_results(tmp_path) -> None:
    request = make_request(
        [[10.0, 10.0], [10.0, 10.0]],
        [[1, 1], [0, 0]],
        target_n=1,
    )
    database = tmp_path / "overflow.db"
    results = tmp_path / "overflow-results"
    with pytest.raises(ValueError, match="requests 2 directions"):
        run_and_persist(request, database_path=database, results_dir=results)
    runs = list_runs(database_path=database)
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM equity_curve WHERE run_id = ?", (runs[0]["id"],)
        ).fetchone()[0] == 0
    assert not (results / runs[0]["id"]).exists()


def test_webui_pages_endpoints_and_local_plotly(tmp_path) -> None:
    first, database, results = _completed_run(tmp_path)
    second, _, _ = _completed_run(tmp_path, initial_cash=2500.0)
    app = create_app(database, results)
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert first[:12] in index.text
        assert "Backtest library" in index.text

        detail = client.get(f"/runs/{first}")
        assert detail.status_code == 200
        assert "Powering chart" not in detail.text
        assert "View equity data" in detail.text
        assert "Trade blotter" in detail.text

        series = client.get(f"/api/runs/{first}/series?scope=full")
        assert series.status_code == 200
        assert series.json()[0]["date"] == "2024-01-02"
        assert "return" in series.json()[0]

        comparison = client.get(f"/compare?runs={first}&runs={second}")
        assert comparison.status_code == 200
        assert "Normalized equity" in comparison.text

        archive = client.post(f"/api/runs/{first}/archive", json={"archived": True})
        assert archive.status_code == 200
        assert archive.json()["archived"] is True

        trades = client.get(
            f"/api/runs/{first}/trades?scope=full&direction=long&page_size=10"
        )
        assert trades.status_code == 200
        assert all(item["direction"] == "long" for item in trades.json()["items"])

        plotly = client.get("/vendor/plotly.min.js")
        assert plotly.status_code == 200
        assert "Plotly" in plotly.text[:2000]


def test_walk_forward_isolation_and_summary(tmp_path) -> None:
    from portfolio_backtest.models import WalkForwardWindow

    request = make_request(
        [[10.0], [10.0], [11.0], [12.0], [12.0], [13.0], [14.0]],
        [[1], [1], [0], [0], [1], [1], [0]],
        mode="both",
    )
    request.walk_forward = (
        WalkForwardWindow(start_date="2024-01-02", end_date="2024-01-08"),
        WalkForwardWindow(start_date="2024-01-09", end_date="2024-01-10"),
    )
    result = execute_backtest(request)
    assert result.full_window is not None
    assert len(result.walk_forward) == 2
    assert result.walk_forward_summary is not None
    assert all(
        scope.equity_curve.iloc[0]["equity"] == pytest.approx(1000.0)
        for scope in result.walk_forward
    )
    assert result.walk_forward_summary.mean["total_return"] is not None
