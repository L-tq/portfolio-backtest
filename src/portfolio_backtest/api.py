from __future__ import annotations

import importlib.resources
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

import plotly
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from .metrics import METRIC_NAMES
from .storage import (
    ConfirmationRequiredError,
    DEFAULT_DATABASE,
    DEFAULT_RESULTS_DIR,
    RunNotFoundError,
    delete_run,
    get_metrics,
    get_run,
    initialize_database,
    list_runs,
    load_scope_series,
    set_archive,
    trade_blotter,
)


METRIC_LABELS = {
    "total_return": "Total return",
    "cagr": "CAGR",
    "volatility": "Annual volatility",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "Max drawdown",
    "calmar": "Calmar",
    "hit_rate": "Closed-trade hit rate",
    "turnover": "Annual one-way turnover",
    "average_long_exposure": "Average long exposure",
    "average_short_exposure": "Average short exposure",
    "alpha": "Annual alpha",
    "beta": "Beta",
    "correlation": "Correlation",
    "tracking_error": "Tracking error",
    "information_ratio": "Information ratio",
}


class RunListItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    status: str
    archived: bool
    created_at: str
    completed_at: str | None
    package_version: str
    config_hash: str
    data_panel_hash: str
    benchmark_hash: str | None
    failure_details: str | None


class RunDetail(RunListItem):
    config: dict[str, Any] | None
    equity_curve: list[dict[str, Any]]
    benchmark: dict[str, Any] | None
    walk_forward: list[dict[str, Any]]


class ArchiveRequest(BaseModel):
    archived: bool = True


class TradeBlotterResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int
    page: int
    page_size: int
    pages: int


class SeriesPoint(BaseModel):
    model_config = ConfigDict(extra="allow")
    date: str
    equity: float
    cash: float
    long_exposure: float
    short_exposure: float
    return_value: float = Field(alias="return")
    drawdown: float
    benchmark: float | None
    benchmark_return: float | None


def _format_metric(value: float | None, name: str) -> str:
    if value is None:
        return "—"
    if name in {
        "total_return",
        "cagr",
        "volatility",
        "max_drawdown",
        "hit_rate",
        "average_long_exposure",
        "average_short_exposure",
        "positive_window_rate",
    }:
        return f"{value:.1%}"
    return f"{value:.3f}"


def _metric_table(run: dict[str, Any], database_path: Path) -> list[dict[str, Any]]:
    scopes = [item["scope"] for item in run["equity_curve"]]
    metrics_by_scope = {
        scope: get_metrics(run["id"], scope, database_path=database_path) for scope in scopes
    }
    rows = []
    for name in METRIC_NAMES:
        rows.append(
            {
                "name": name,
                "label": METRIC_LABELS[name],
                "values": {
                    scope: {
                        "value": metrics_by_scope[scope].get(name, {}).get("value"),
                        "benchmark": metrics_by_scope[scope].get(name, {}).get("benchmark"),
                        "display": _format_metric(
                            metrics_by_scope[scope].get(name, {}).get("value"), name
                        ),
                        "benchmark_display": _format_metric(
                            metrics_by_scope[scope].get(name, {}).get("benchmark"), name
                        ),
                    }
                    for scope in scopes
                },
            }
        )
    for name, label in (("long", "Long closed trades"), ("short", "Short closed trades")):
        values = {}
        for scope in scopes:
            details = metrics_by_scope[scope].get(name, {}).get("details")
            if details:
                display = (
                    f"{details.get('count', 0)} trades · "
                    f"hit {_format_metric(details.get('hit_rate'), 'hit_rate')} · "
                    f"avg P&L {_format_metric(details.get('average_pnl'), 'number')}"
                )
            else:
                display = "—"
            values[scope] = {
                "value": None,
                "benchmark": None,
                "display": display,
                "benchmark_display": "—",
            }
        rows.append({"name": name, "label": label, "values": values})
    return rows


def create_app(
    database_path: str | Path = DEFAULT_DATABASE,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> FastAPI:
    database_path = Path(database_path)
    results_dir = Path(results_dir)
    package_root = importlib.resources.files("portfolio_backtest")
    templates = Jinja2Templates(directory=str(package_root / "templates"))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        initialize_database(database_path)
        yield

    app = FastAPI(
        title="Portfolio Backtest",
        version="0.1.0",
        description="Local research interface for completed portfolio backtests.",
        lifespan=lifespan,
    )
    app.state.database_path = database_path
    app.state.results_dir = results_dir
    app.mount(
        "/static",
        StaticFiles(directory=str(package_root / "static")),
        name="static",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/vendor/plotly.min.js", include_in_schema=False)
    def plotly_javascript() -> FileResponse:
        path = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
        if not path.exists():
            raise HTTPException(status_code=404, detail="local Plotly.js bundle is unavailable")
        return FileResponse(path, media_type="text/javascript")

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        status: str | None = None,
        archived: bool | None = None,
        q: str | None = None,
    ) -> HTMLResponse:
        runs = list_runs(
            database_path=database_path,
            status=status or None,
            archived=archived,
            query=q or None,
        )
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "runs": runs,
                "status": status or "",
                "archived": "" if archived is None else str(archived).lower(),
                "query": q or "",
            },
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(
        request: Request,
        run_id: str,
        scope: str | None = None,
        symbol: str | None = None,
        direction: Literal["long", "short"] | None = None,
        reason: Literal["signal", "margin", "delisting"] | None = None,
        page: int = Query(default=1, ge=1),
    ) -> HTMLResponse:
        try:
            run = get_run(run_id, database_path=database_path)
        except RunNotFoundError as error:
            raise HTTPException(status_code=404, detail="run not found") from error
        if run["status"] != "completed":
            return templates.TemplateResponse(
                request=request,
                name="run_unavailable.html",
                context={"run": run},
                status_code=409,
            )
        scopes = [item["scope"] for item in run["equity_curve"]]
        selected_scope = scope if scope in scopes else scopes[0]
        blotter = trade_blotter(
            run_id,
            database_path=database_path,
            scope=selected_scope,
            symbol=symbol or None,
            direction=direction,
            reason=reason,
            page=page,
            page_size=25,
        )
        query_values = {
            "scope": selected_scope,
            **({"symbol": symbol} if symbol else {}),
            **({"direction": direction} if direction else {}),
            **({"reason": reason} if reason else {}),
        }
        summary = next(
            (
                item["metrics"]
                for item in run["walk_forward"]
                if item["scope"] == "summary"
            ),
            None,
        )
        summary_names = {
            "total_return": "Total return",
            "cagr": "CAGR",
            "sharpe": "Sharpe",
            "sortino": "Sortino",
            "max_drawdown": "Max drawdown",
            "turnover": "Turnover",
        }
        summary_rows = []
        if summary:
            for name, label in summary_names.items():
                summary_rows.append(
                    {
                        "label": label,
                        "mean": _format_metric(summary["mean"].get(name), name),
                        "median": _format_metric(summary["median"].get(name), name),
                        "positive": _format_metric(
                            summary["positive_window_rate"].get(name), "positive_window_rate"
                        ),
                    }
                )
        return templates.TemplateResponse(
            request=request,
            name="detail.html",
            context={
                "run": run,
                "scopes": scopes,
                "selected_scope": selected_scope,
                "metrics": _metric_table(run, database_path),
                "trades": blotter,
                "filters": {
                    "symbol": symbol or "",
                    "direction": direction or "",
                    "reason": reason or "",
                    "query_string": urlencode(query_values),
                },
                "walk_summary": summary,
                "summary_rows": summary_rows,
            },
        )

    @app.get("/compare", response_class=HTMLResponse)
    def compare(request: Request, runs: list[str] = Query(default=[])) -> HTMLResponse:
        unique_ids = list(dict.fromkeys(runs))
        if len(unique_ids) < 2 or len(unique_ids) > 6:
            raise HTTPException(
                status_code=422,
                detail="select between 2 and 6 completed runs",
            )
        selected = []
        for run_id in unique_ids:
            try:
                run = get_run(run_id, database_path=database_path)
            except RunNotFoundError as error:
                raise HTTPException(status_code=404, detail=f"run {run_id} not found") from error
            if run["status"] != "completed":
                raise HTTPException(status_code=409, detail=f"run {run_id} is not completed")
            selected.append(run)
        available = list_runs(database_path=database_path, status="completed")
        return templates.TemplateResponse(
            request=request,
            name="compare.html",
            context={
                "runs": selected,
                "available_runs": available,
                "run_ids": unique_ids,
            },
        )

    @app.get("/api/runs", response_model=list[RunListItem])
    def api_runs(
        status: str | None = None,
        archived: bool | None = None,
        q: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict[str, Any]]:
        return list_runs(
            database_path=database_path,
            status=status,
            archived=archived,
            query=q,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/runs/{run_id}", response_model=RunDetail)
    def api_run(run_id: str) -> dict[str, Any]:
        try:
            return get_run(run_id, database_path=database_path)
        except RunNotFoundError as error:
            raise HTTPException(status_code=404, detail="run not found") from error

    @app.get("/api/runs/{run_id}/series", response_model=list[SeriesPoint])
    def api_series(run_id: str, scope: str = "full") -> list[dict[str, Any]]:
        try:
            return load_scope_series(run_id, scope, database_path=database_path)
        except RunNotFoundError as error:
            raise HTTPException(status_code=404, detail="run scope not found") from error

    @app.get("/api/runs/{run_id}/trades", response_model=TradeBlotterResponse)
    def api_trades(
        run_id: str,
        scope: str | None = None,
        symbol: str | None = None,
        direction: Literal["long", "short"] | None = None,
        reason: Literal["signal", "margin", "delisting"] | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=250),
    ) -> dict[str, Any]:
        try:
            return trade_blotter(
                run_id,
                database_path=database_path,
                scope=scope,
                symbol=symbol,
                direction=direction,
                reason=reason,
                page=page,
                page_size=page_size,
            )
        except RunNotFoundError as error:
            raise HTTPException(status_code=404, detail="run not found") from error

    @app.post("/api/runs/{run_id}/archive", response_model=RunDetail)
    def api_archive(run_id: str, body: ArchiveRequest) -> dict[str, Any]:
        try:
            return set_archive(
                run_id,
                body.archived,
                database_path=database_path,
            )
        except RunNotFoundError as error:
            raise HTTPException(status_code=404, detail="run not found") from error

    @app.delete("/api/runs/{run_id}", status_code=204)
    def api_delete(run_id: str, confirm: bool = False) -> None:
        try:
            delete_run(
                run_id,
                confirm=confirm,
                database_path=database_path,
                results_dir=results_dir,
            )
        except RunNotFoundError as error:
            raise HTTPException(status_code=404, detail="run not found") from error
        except ConfirmationRequiredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return app


app = create_app(
    os.environ.get("BT_DATABASE_PATH", DEFAULT_DATABASE),
    os.environ.get("BT_RESULTS_DIR", DEFAULT_RESULTS_DIR),
)
