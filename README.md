# Portfolio Backtest

A local, deterministic portfolio backtesting engine with a Numba-compiled
simulation kernel, SQLite metadata, Parquet series, and an accessible FastAPI
research UI.

## Install

```powershell
python -m pip install -e ".[test]"
```

## Run a backtest

```powershell
portfolio-backtest run `
  --data-dir D:\path\to\data `
  --symbols AAPL.US MSFT.US `
  --signals .\signals.csv `
  --start 2024-01-01 `
  --end 2025-12-31 `
  --initial-cash 100000 `
  --target-n 10
```

Signals use the long form `date,symbol,signal`, where `1` is long, `-1` is
short, and `0` is flat. Prices are loaded from
`<data-dir>/<symbol>/prices.csv`. Benchmark tickers use the same layout.

Walk-forward files contain inclusive `start_date,end_date` rows. Each window is
run independently from the configured initial cash.

## Open the UI

```powershell
portfolio-backtest web
```

The server binds to `127.0.0.1:8000` by default.

## Configuration

JSON configuration accepts the fields on `BacktestConfig`. The defaults are:
five-bar rebalance cadence, one-bar execution lag, 10 bps commission, 10%
annual borrow, 30% maintenance margin, no settlement delay, 252 annualization
days, and both full-window and walk-forward execution. Command-line overrides
take precedence over JSON, which takes precedence over package defaults.
