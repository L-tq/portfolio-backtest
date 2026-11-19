from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


METRIC_NAMES = (
    "total_return",
    "cagr",
    "volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "hit_rate",
    "turnover",
    "average_long_exposure",
    "average_short_exposure",
    "alpha",
    "beta",
    "correlation",
    "tracking_error",
    "information_ratio",
)


def _finite(value: float | int | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _closed_trade_stats(trades: pd.DataFrame, direction: str) -> dict[str, Any]:
    if trades.empty:
        closed = trades
    else:
        closed = trades[(trades["direction"] == direction) & (trades["action"] == "close")]
    pnls = closed["realized_pnl"].dropna().to_numpy(dtype=float) if not closed.empty else np.array([])
    wins = pnls[pnls > 0]
    count = int(len(pnls))
    return {
        "count": count,
        "hit_rate": _finite(len(wins) / count) if count else None,
        "average_pnl": _finite(np.mean(pnls)) if count else None,
        "total_pnl": _finite(np.sum(pnls)) if count else None,
        "average_win": _finite(np.mean(wins)) if len(wins) else None,
        "average_loss": _finite(np.mean(pnls[pnls < 0])) if np.any(pnls < 0) else None,
    }


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    annualization: int,
    risk_free_rate: float,
    initial_cash: float,
) -> dict[str, Any]:
    if equity_curve.empty:
        return {name: None for name in METRIC_NAMES}

    equity = equity_curve["equity"].to_numpy(dtype=float)
    returns = equity_curve["return"].fillna(0.0).to_numpy(dtype=float)
    periods = max(len(equity) - 1, 1)
    years = periods / annualization
    total_return = equity[-1] / initial_cash - 1.0
    cagr = (equity[-1] / initial_cash) ** (annualization / periods) - 1.0 if equity[-1] > 0 else None

    excess = returns - risk_free_rate / annualization
    volatility = np.std(returns, ddof=1) * math.sqrt(annualization) if len(returns) > 1 else None
    sharpe = (
        np.mean(excess) / np.std(returns, ddof=1) * math.sqrt(annualization)
        if len(returns) > 1 and np.std(returns, ddof=1) > 0
        else None
    )
    downside = np.minimum(excess, 0.0)
    downside_deviation = math.sqrt(float(np.mean(np.square(downside)))) if len(downside) else None
    sortino = (
        float(np.mean(excess)) / downside_deviation * math.sqrt(annualization)
        if downside_deviation and downside_deviation > 0
        else None
    )
    max_drawdown = float(equity_curve["drawdown"].min())
    calmar = cagr / abs(max_drawdown) if cagr is not None and max_drawdown < 0 else None

    if trades.empty:
        closed_trades = trades
        traded_notional = 0.0
    else:
        closed_trades = trades
        traded_notional = float(trades["notional"].abs().sum())
    closed_trades = (
        closed_trades[closed_trades["action"] == "close"] if not closed_trades.empty else closed_trades
    )
    hit_rate = (
        float((closed_trades["realized_pnl"] > 0).mean()) if not closed_trades.empty else None
    )
    turnover = (
        traded_notional / max(float(np.mean(equity)), 1e-12) / years if years > 0 else None
    )
    avg_long = float(equity_curve["long_exposure"].div(equity_curve["equity"]).mean())
    avg_short = float(equity_curve["short_exposure"].div(equity_curve["equity"]).mean())

    benchmark_values = equity_curve["benchmark"].to_numpy(dtype=float)
    benchmark_returns = equity_curve["benchmark_return"].to_numpy(dtype=float)
    has_benchmark = np.isfinite(benchmark_returns).sum() > 2
    alpha = beta = correlation = tracking_error = information_ratio = None
    if has_benchmark:
        mask = np.isfinite(benchmark_returns) & np.isfinite(returns)
        strategy = returns[mask]
        bench = benchmark_returns[mask]
        bench_variance = float(np.var(bench, ddof=1)) if len(bench) > 1 else 0.0
        if bench_variance > 0:
            beta = float(np.cov(strategy, bench, ddof=1)[0, 1] / bench_variance)
            alpha = float(
                (np.mean(strategy) - risk_free_rate / annualization - beta * (
                    np.mean(bench) - risk_free_rate / annualization
                ))
                * annualization
            )
            if np.std(strategy, ddof=1) > 0 and np.std(bench, ddof=1) > 0:
                correlation = float(np.corrcoef(strategy, bench)[0, 1])
        active = strategy - bench
        if len(active) > 1:
            tracking_error = float(np.std(active, ddof=1) * math.sqrt(annualization))
            information_ratio = (
                float(np.mean(active) * annualization / tracking_error)
                if tracking_error and tracking_error > 0
                else None
            )

    metrics: dict[str, Any] = {
        "total_return": _finite(total_return),
        "cagr": _finite(cagr),
        "volatility": _finite(volatility),
        "sharpe": _finite(sharpe),
        "sortino": _finite(sortino),
        "max_drawdown": _finite(max_drawdown),
        "calmar": _finite(calmar),
        "hit_rate": _finite(hit_rate),
        "turnover": _finite(turnover),
        "average_long_exposure": _finite(avg_long),
        "average_short_exposure": _finite(avg_short),
        "alpha": _finite(alpha),
        "beta": _finite(beta),
        "correlation": _finite(correlation),
        "tracking_error": _finite(tracking_error),
        "information_ratio": _finite(information_ratio),
        "long": _closed_trade_stats(trades, "long"),
        "short": _closed_trade_stats(trades, "short"),
    }
    return metrics


def aggregate_walk_forward(results: list[dict[str, Any]]) -> dict[str, Any]:
    mean: dict[str, float | None] = {}
    median: dict[str, float | None] = {}
    positive: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        values = [item.get(name) for item in results]
        numeric = np.array([value for value in values if value is not None], dtype=float)
        mean[name] = _finite(np.mean(numeric)) if len(numeric) else None
        median[name] = _finite(np.median(numeric)) if len(numeric) else None
        positive[name] = _finite(np.mean(numeric > 0)) if len(numeric) else None
    return {"mean": mean, "median": median, "positive_window_rate": positive}
