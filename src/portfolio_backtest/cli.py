from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .api import create_app
from .data import load_backtest_request, load_dotenv_files
from .storage import DEFAULT_DATABASE, DEFAULT_RESULTS_DIR, run_and_persist


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="portfolio-backtest")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run and persist a portfolio backtest")
    run.add_argument("--data-dir", help="root containing <symbol>/prices.csv")
    run.add_argument("--symbols", nargs="+", required=True)
    run.add_argument("--signals", required=True, help="long-form date,symbol,signal CSV")
    run.add_argument("--start")
    run.add_argument("--end")
    run.add_argument("--config", help="JSON BacktestConfig file")
    run.add_argument("--walk-forward", help="walk-forward start/end CSV")
    benchmark = run.add_mutually_exclusive_group()
    benchmark.add_argument("--benchmark-ticker")
    benchmark.add_argument("--benchmark-file")
    benchmark.add_argument("--no-benchmark", action="store_true")
    run.add_argument("--database", default=str(DEFAULT_DATABASE))
    run.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))

    run.add_argument("--initial-cash", type=float)
    run.add_argument("--target-n", type=int)
    run.add_argument("--rebalance-days", type=int)
    run.add_argument("--execution-lag-bars", type=int)
    run.add_argument("--commission-rate", type=float)
    run.add_argument("--borrow-rate", type=float)
    run.add_argument("--maintenance-margin", type=float)
    run.add_argument("--delisting-haircut", type=float)
    run.add_argument("--settlement-days", type=int)
    run.add_argument("--annualization", type=int)
    run.add_argument("--risk-free-rate", type=float)
    run.add_argument("--mode", choices=("full", "walk_forward", "both"))

    web = subparsers.add_parser("web", help="start the local FastAPI WebUI")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--database", default=str(DEFAULT_DATABASE))
    web.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    web.add_argument("--reload", action="store_true")
    return parser


def _config_overrides(args: argparse.Namespace) -> dict:
    names = (
        "initial_cash",
        "target_n",
        "rebalance_days",
        "execution_lag_bars",
        "commission_rate",
        "borrow_rate",
        "maintenance_margin",
        "delisting_haircut",
        "settlement_days",
        "annualization",
        "risk_free_rate",
        "mode",
    )
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def _run_command(args: argparse.Namespace) -> int:
    from .data import config_from_json

    data_dir = args.data_dir or os.environ.get("BT_DATA_DIR")
    if not data_dir:
        raise ValueError("provide --data-dir or set BT_DATA_DIR")
    config = config_from_json(args.config, _config_overrides(args))
    request = load_backtest_request(
        data_dir=data_dir,
        symbols=args.symbols,
        signal_file=args.signals,
        start=args.start,
        end=args.end,
        config=config,
        walk_forward_file=args.walk_forward,
        benchmark_ticker=args.benchmark_ticker,
        benchmark_file=args.benchmark_file,
    )
    result = run_and_persist(
        request,
        database_path=args.database,
        results_dir=args.results_dir,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "database": str(Path(args.database).resolve()),
                "results_dir": str(Path(args.results_dir).resolve()),
                "full_window": result.full_window is not None,
                "walk_forward_windows": len(result.walk_forward),
            },
            indent=2,
        )
    )
    return 0


def _web_command(args: argparse.Namespace) -> int:
    import uvicorn

    os.environ["BT_DATABASE_PATH"] = str(Path(args.database).resolve())
    os.environ["BT_RESULTS_DIR"] = str(Path(args.results_dir).resolve())
    if args.reload:
        uvicorn.run(
            "portfolio_backtest.api:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
        return 0
    app = create_app(args.database, args.results_dir)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv_files()
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run_command(args)
        if args.command == "web":
            return _web_command(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
