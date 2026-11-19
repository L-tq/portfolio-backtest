from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .data import canonical_config_json, config_hash, panel_hash, source_tree_hash
from .engine import execute_backtest
from .metrics import METRIC_NAMES, calculate_metrics
from .models import (
    BacktestConfig,
    BacktestRequest,
    BacktestResult,
    BenchmarkSeries,
    ScopeResult,
    WalkForwardSummary,
)


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_DATABASE = DEFAULT_RESULTS_DIR / "portfolio.db"


class RunNotFoundError(KeyError):
    pass


class ConfirmationRequiredError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: str | Path) -> sqlite3.Connection:
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def database(path: str | Path = DEFAULT_DATABASE) -> Iterator[sqlite3.Connection]:
    connection = _connect(path)
    try:
        yield connection
    finally:
        connection.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0 CHECK (archived IN (0, 1)),
    package_version TEXT NOT NULL,
    source_tree_hash TEXT NOT NULL,
    data_panel_hash TEXT NOT NULL,
    benchmark_hash TEXT,
    config_hash TEXT NOT NULL,
    failure_details TEXT,
    started_at TEXT,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS config (
    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS equity_curve (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    scope TEXT NOT NULL,
    path TEXT NOT NULL,
    rows INTEGER NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    PRIMARY KEY (run_id, scope)
);
CREATE TABLE IF NOT EXISTS benchmark_series (
    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    ticker TEXT,
    path TEXT NOT NULL,
    rows INTEGER NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    scope TEXT NOT NULL,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    shares REAL NOT NULL,
    close REAL NOT NULL,
    direction TEXT NOT NULL,
    market_value REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS positions_scope_idx
ON positions(run_id, scope, date, symbol);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    scope TEXT NOT NULL,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    notional REAL NOT NULL,
    commission REAL NOT NULL,
    cash_after REAL NOT NULL,
    realized_pnl REAL,
    reason TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS trades_scope_idx
ON trades(run_id, scope, date, symbol);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    scope TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL,
    benchmark_value REAL,
    details_json TEXT,
    PRIMARY KEY (run_id, scope, name)
);
CREATE TABLE IF NOT EXISTS walk_forward_windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    scope TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    UNIQUE (run_id, scope)
);
"""


def initialize_database(path: str | Path = DEFAULT_DATABASE) -> None:
    with database(path) as connection:
        connection.executescript(SCHEMA)
        connection.commit()


def _benchmark_hash(benchmark: BenchmarkSeries | None) -> str | None:
    if benchmark is None:
        return None
    digest = hashlib.sha256()
    digest.update(benchmark.dates.tobytes())
    digest.update(benchmark.values.tobytes())
    return digest.hexdigest()


def start_run(
    request: BacktestRequest,
    *,
    database_path: str | Path = DEFAULT_DATABASE,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> str:
    initialize_database(database_path)
    run_id = uuid.uuid4().hex
    now = utc_now()
    run_directory = Path(results_dir) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    with database(database_path) as connection:
        connection.execute(
            """
            INSERT INTO runs (
                id, created_at, updated_at, status, archived, package_version,
                source_tree_hash, data_panel_hash, benchmark_hash, config_hash, started_at
            ) VALUES (?, ?, ?, 'running', 0, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                now,
                now,
                __version__,
                source_tree_hash(),
                panel_hash(request.prices, request.signals, request.benchmark),
                _benchmark_hash(request.benchmark),
                config_hash(request.config),
                now,
            ),
        )
        connection.execute(
            "INSERT INTO config (run_id, config_json) VALUES (?, ?)",
            (run_id, canonical_config_json(request.config)),
        )
        connection.commit()
    return run_id


def record_failure(
    run_id: str,
    error: BaseException,
    *,
    database_path: str | Path = DEFAULT_DATABASE,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> None:
    initialize_database(database_path)
    now = utc_now()
    details = f"{type(error).__name__}: {error}"
    with database(database_path) as connection:
        connection.execute(
            """
            UPDATE runs
            SET status = 'failed', updated_at = ?, completed_at = ?, failure_details = ?
            WHERE id = ?
            """,
            (now, now, details, run_id),
        )
        connection.commit()
    run_directory = Path(results_dir) / run_id
    if run_directory.exists():
        shutil.rmtree(run_directory)


def _benchmark_metrics(
    scope: ScopeResult, request: BacktestRequest
) -> dict[str, float | None]:
    benchmark = scope.equity_curve["benchmark"].to_numpy(dtype=float)
    valid = np.isfinite(benchmark)
    if not valid.any():
        return {name: None for name in METRIC_NAMES}
    normalized = benchmark / benchmark[valid][0] * request.config.initial_cash
    frame = pd.DataFrame(
        {
            "equity": normalized,
            "return": pd.Series(normalized).pct_change(fill_method=None).fillna(0.0),
            "drawdown": pd.Series(normalized).div(pd.Series(normalized).cummax()).sub(1.0),
            "long_exposure": 0.0,
            "short_exposure": 0.0,
            "benchmark": benchmark,
            "benchmark_return": scope.equity_curve["benchmark_return"].to_numpy(),
        }
    )
    values = calculate_metrics(
        frame,
        pd.DataFrame(),
        annualization=request.config.annualization,
        risk_free_rate=request.config.risk_free_rate,
        initial_cash=request.config.initial_cash,
    )
    return {name: values.get(name) for name in METRIC_NAMES}


def save_backtest(
    result: BacktestResult,
    request: BacktestRequest,
    *,
    database_path: str | Path = DEFAULT_DATABASE,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> BacktestResult:
    initialize_database(database_path)
    results_root = Path(results_dir).resolve()
    run_directory = (results_root / result.run_id).resolve()
    if results_root not in run_directory.parents:
        raise ValueError("run directory escaped the results root")
    run_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    scopes = [scope for scope in (result.full_window, *result.walk_forward) if scope]
    catalog: dict[str, tuple[str, int, str, str]] = {}

    try:
        for scope in scopes:
            path = run_directory / f"{scope.scope}.parquet"
            temporary = path.with_suffix(".parquet.tmp")
            scope.equity_curve.to_parquet(temporary, index=False)
            temporary.replace(path)
            written.append(path)
            catalog[scope.scope] = (
                str(path),
                len(scope.equity_curve),
                str(scope.start_date),
                str(scope.end_date),
            )

        benchmark_path = None
        benchmark_catalog = None
        if request.benchmark is not None:
            benchmark_path = run_directory / "benchmark.parquet"
            benchmark_frame = pd.DataFrame(
                {
                    "date": pd.DatetimeIndex(request.benchmark.dates),
                    "value": request.benchmark.values,
                }
            )
            benchmark_frame.to_parquet(benchmark_path, index=False)
            written.append(benchmark_path)
            benchmark_catalog = (
                request.benchmark.ticker,
                str(benchmark_path),
                len(benchmark_frame),
                str(request.benchmark.dates[0]),
                str(request.benchmark.dates[-1]),
            )

        now = utc_now()
        with database(database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE runs
                SET status = 'completed', updated_at = ?, completed_at = ?,
                    failure_details = NULL
                WHERE id = ?
                """,
                (now, now, result.run_id),
            )
            for scope in scopes:
                path, rows, start_date, end_date = catalog[scope.scope]
                connection.execute(
                    """
                    INSERT INTO equity_curve
                    (run_id, scope, path, rows, start_date, end_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (result.run_id, scope.scope, path, rows, start_date, end_date),
                )
                position_frame = scope.positions.copy()
                position_frame["date"] = pd.to_datetime(position_frame["date"]).astype(str)
                connection.executemany(
                    """
                    INSERT INTO positions
                    (run_id, scope, date, symbol, shares, close, direction, market_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            result.run_id,
                            scope.scope,
                            row.date,
                            row.symbol,
                            float(row.shares),
                            float(row.close),
                            row.direction,
                            float(row.market_value),
                        )
                        for row in position_frame.itertuples(index=False)
                    ),
                )
                trade_frame = scope.trades.copy()
                if not trade_frame.empty:
                    trade_frame["date"] = pd.to_datetime(trade_frame["date"]).astype(str)
                    connection.executemany(
                        """
                        INSERT INTO trades
                        (run_id, scope, date, symbol, direction, action, quantity, price,
                         notional, commission, cash_after, realized_pnl, reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            (
                                result.run_id,
                                scope.scope,
                                row.date,
                                row.symbol,
                                row.direction,
                                row.action,
                                float(row.quantity),
                                float(row.price),
                                float(row.notional),
                                float(row.commission),
                                float(row.cash_after),
                                float(row.realized_pnl),
                                row.reason,
                            )
                            for row in trade_frame.itertuples(index=False)
                        ),
                    )
                benchmark_metrics = _benchmark_metrics(scope, request)
                for name in (*METRIC_NAMES, "long", "short"):
                    details = None
                    value = scope.metrics.get(name)
                    if name in {"long", "short"}:
                        details = json.dumps(scope.metrics.get(name), sort_keys=True)
                        value = None
                    connection.execute(
                        """
                        INSERT INTO metrics
                        (run_id, scope, name, value, benchmark_value, details_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            result.run_id,
                            scope.scope,
                            name,
                            value,
                            benchmark_metrics.get(name),
                            details,
                        ),
                    )
                if scope.scope != "full":
                    connection.execute(
                        """
                        INSERT INTO walk_forward_windows
                        (run_id, scope, start_date, end_date, metrics_json)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            result.run_id,
                            scope.scope,
                            str(scope.start_date),
                            str(scope.end_date),
                            json.dumps(scope.metrics, sort_keys=True, default=str),
                        ),
                    )

            if benchmark_catalog is not None:
                connection.execute(
                    """
                    INSERT INTO benchmark_series
                    (run_id, ticker, path, rows, start_date, end_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.run_id,
                        benchmark_catalog[0],
                        benchmark_catalog[1],
                        benchmark_catalog[2],
                        benchmark_catalog[3],
                        benchmark_catalog[4],
                    ),
                )
            if result.walk_forward_summary is not None:
                connection.execute(
                    """
                    INSERT INTO walk_forward_windows
                    (run_id, scope, start_date, end_date, metrics_json)
                    VALUES (?, 'summary', ?, ?, ?)
                    """,
                    (
                        result.run_id,
                        str(min(scope.start_date for scope in result.walk_forward)),
                        str(max(scope.end_date for scope in result.walk_forward)),
                        result.walk_forward_summary.model_dump_json(),
                    ),
                )
            connection.commit()
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise

    result.database_path = Path(database_path)
    result.results_dir = results_root
    return result


def run_and_persist(
    request: BacktestRequest,
    *,
    database_path: str | Path = DEFAULT_DATABASE,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> BacktestResult:
    run_id = start_run(request, database_path=database_path, results_dir=results_dir)
    try:
        result = execute_backtest(request, run_id=run_id)
        return save_backtest(
            result,
            request,
            database_path=database_path,
            results_dir=results_dir,
        )
    except BaseException as error:
        record_failure(
            run_id,
            error,
            database_path=database_path,
            results_dir=results_dir,
        )
        raise


def _run_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["archived"] = bool(item["archived"])
    return item


def list_runs(
    *,
    database_path: str | Path = DEFAULT_DATABASE,
    status: str | None = None,
    archived: bool | None = None,
    query: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    initialize_database(database_path)
    clauses = []
    values: list[Any] = []
    if status:
        clauses.append("status = ?")
        values.append(status)
    if archived is not None:
        clauses.append("archived = ?")
        values.append(int(archived))
    if query:
        clauses.append("(id LIKE ? OR config_hash LIKE ? OR data_panel_hash LIKE ?)")
        token = f"%{query}%"
        values.extend((token, token, token))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    values.extend((limit, offset))
    with database(database_path) as connection:
        rows = connection.execute(
            f"""
            SELECT runs.*, config.config_json
            FROM runs
            LEFT JOIN config ON config.run_id = runs.id
            {where}
            ORDER BY runs.created_at DESC
            LIMIT ? OFFSET ?
            """,
            values,
        ).fetchall()
    results = []
    for row in rows:
        item = _run_row_to_dict(row)
        config_json = item.pop("config_json", None)
        item["config"] = json.loads(config_json) if config_json else None
        results.append(item)
    return results


def get_run(
    run_id: str, *, database_path: str | Path = DEFAULT_DATABASE
) -> dict[str, Any]:
    initialize_database(database_path)
    with database(database_path) as connection:
        row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFoundError(run_id)
        result = _run_row_to_dict(row)
        config_row = connection.execute(
            "SELECT config_json FROM config WHERE run_id = ?", (run_id,)
        ).fetchone()
        result["config"] = (
            json.loads(config_row["config_json"]) if config_row is not None else None
        )
        result["equity_curve"] = [
            dict(item)
            for item in connection.execute(
                "SELECT scope, path, rows, start_date, end_date FROM equity_curve "
                "WHERE run_id = ? ORDER BY scope",
                (run_id,),
            )
        ]
        result["benchmark"] = None
        benchmark = connection.execute(
            "SELECT * FROM benchmark_series WHERE run_id = ?", (run_id,)
        ).fetchone()
        if benchmark is not None:
            result["benchmark"] = dict(benchmark)
        result["walk_forward"] = [
            {
                **dict(item),
                "metrics": json.loads(item["metrics_json"]),
            }
            for item in connection.execute(
                "SELECT * FROM walk_forward_windows WHERE run_id = ? ORDER BY scope",
                (run_id,),
            )
        ]
    return result


def load_scope_series(
    run_id: str,
    scope: str,
    *,
    database_path: str | Path = DEFAULT_DATABASE,
) -> list[dict[str, Any]]:
    get_run(run_id, database_path=database_path)
    with database(database_path) as connection:
        row = connection.execute(
            "SELECT path FROM equity_curve WHERE run_id = ? AND scope = ?",
            (run_id, scope),
        ).fetchone()
    if row is None:
        raise RunNotFoundError(f"{run_id}/{scope}")
    frame = pd.read_parquet(row["path"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame.replace({np.nan: None}).to_dict(orient="records")


def trade_blotter(
    run_id: str,
    *,
    database_path: str | Path = DEFAULT_DATABASE,
    scope: str | None = None,
    symbol: str | None = None,
    direction: str | None = None,
    reason: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    get_run(run_id, database_path=database_path)
    page = max(page, 1)
    page_size = min(max(page_size, 1), 250)
    clauses = ["run_id = ?"]
    values: list[Any] = [run_id]
    for column, value in (
        ("scope", scope),
        ("symbol", symbol),
        ("direction", direction),
        ("reason", reason),
    ):
        if value:
            clauses.append(f"{column} = ?")
            values.append(value)
    where = " AND ".join(clauses)
    with database(database_path) as connection:
        total = connection.execute(
            f"SELECT COUNT(*) FROM trades WHERE {where}", values
        ).fetchone()[0]
        rows = connection.execute(
            f"SELECT * FROM trades WHERE {where} ORDER BY date, id LIMIT ? OFFSET ?",
            (*values, page_size, (page - 1) * page_size),
        ).fetchall()
    return {
        "items": [dict(row) for row in rows],
        "total": int(total),
        "page": page,
        "page_size": page_size,
        "pages": max(math_ceil(total / page_size), 1),
    }


def math_ceil(value: float) -> int:
    return int(value) if value.is_integer() else int(value) + 1


def get_metrics(
    run_id: str,
    scope: str,
    *,
    database_path: str | Path = DEFAULT_DATABASE,
) -> dict[str, Any]:
    get_run(run_id, database_path=database_path)
    with database(database_path) as connection:
        rows = connection.execute(
            "SELECT name, value, benchmark_value, details_json FROM metrics "
            "WHERE run_id = ? AND scope = ? ORDER BY name",
            (run_id, scope),
        ).fetchall()
    return {
        row["name"]: {
            "value": row["value"],
            "benchmark": row["benchmark_value"],
            "details": json.loads(row["details_json"]) if row["details_json"] else None,
        }
        for row in rows
    }


def set_archive(
    run_id: str,
    archived: bool,
    *,
    database_path: str | Path = DEFAULT_DATABASE,
) -> dict[str, Any]:
    get_run(run_id, database_path=database_path)
    with database(database_path) as connection:
        connection.execute(
            "UPDATE runs SET archived = ?, updated_at = ? WHERE id = ?",
            (int(archived), utc_now(), run_id),
        )
        connection.commit()
    return get_run(run_id, database_path=database_path)


def delete_run(
    run_id: str,
    *,
    confirm: bool,
    database_path: str | Path = DEFAULT_DATABASE,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> None:
    if not confirm:
        raise ConfirmationRequiredError("run deletion requires explicit confirmation")
    run = get_run(run_id, database_path=database_path)
    paths = [Path(item["path"]) for item in run["equity_curve"]]
    if run["benchmark"] is not None:
        paths.append(Path(run["benchmark"]["path"]))
    with database(database_path) as connection:
        connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        connection.commit()
    results_root = Path(results_dir).resolve()
    run_directory = (results_root / run_id).resolve()
    if results_root in run_directory.parents and run_directory.exists():
        shutil.rmtree(run_directory)
    for path in paths:
        path.unlink(missing_ok=True)


def load_backtest_result(
    run_id: str, *, database_path: str | Path = DEFAULT_DATABASE
) -> BacktestResult:
    run = get_run(run_id, database_path=database_path)
    config = BacktestConfig.model_validate(run["config"])
    scopes: list[ScopeResult] = []
    with database(database_path) as connection:
        for catalog in run["equity_curve"]:
            equity_curve = pd.read_parquet(catalog["path"])
            positions = pd.DataFrame(
                [
                    dict(row)
                    for row in connection.execute(
                        "SELECT date, symbol, shares, close, direction, market_value "
                        "FROM positions WHERE run_id = ? AND scope = ? ORDER BY date, symbol",
                        (run_id, catalog["scope"]),
                    )
                ]
            )
            trades = pd.DataFrame(
                [
                    dict(row)
                    for row in connection.execute(
                        "SELECT date, symbol, direction, action, quantity, price, notional, "
                        "commission, cash_after, realized_pnl, reason FROM trades "
                        "WHERE run_id = ? AND scope = ? ORDER BY date, id",
                        (run_id, catalog["scope"]),
                    )
                ]
            )
            metric_rows = connection.execute(
                "SELECT name, value, details_json FROM metrics "
                "WHERE run_id = ? AND scope = ?",
                (run_id, catalog["scope"]),
            ).fetchall()
            metrics = {}
            for row in metric_rows:
                metrics[row["name"]] = (
                    json.loads(row["details_json"]) if row["details_json"] else row["value"]
                )
            scopes.append(
                ScopeResult(
                    scope=catalog["scope"],
                    start_date=catalog["start_date"],
                    end_date=catalog["end_date"],
                    equity_curve=equity_curve,
                    positions=positions,
                    trades=trades,
                    metrics=metrics,
                )
            )
    full = next((scope for scope in scopes if scope.scope == "full"), None)
    walk_forward = tuple(scope for scope in scopes if scope.scope != "full")
    benchmark = None
    if run["benchmark"] is not None:
        benchmark_frame = pd.read_parquet(run["benchmark"]["path"])
        benchmark = BenchmarkSeries(
            dates=benchmark_frame["date"].to_numpy(dtype="datetime64[D]"),
            values=benchmark_frame["value"].to_numpy(dtype=np.float64),
            ticker=run["benchmark"]["ticker"],
        )
    summary = None
    summary_row = next(
        (item for item in run["walk_forward"] if item["scope"] == "summary"), None
    )
    if summary_row is not None:
        summary = WalkForwardSummary.model_validate(summary_row["metrics"])
    return BacktestResult(
        run_id=run_id,
        config=config,
        full_window=full,
        walk_forward=walk_forward,
        walk_forward_summary=summary,
        benchmark=benchmark,
        database_path=Path(database_path),
    )
