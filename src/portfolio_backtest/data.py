from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .models import (
    BacktestConfig,
    BacktestRequest,
    BenchmarkSeries,
    PricePanel,
    SignalPanel,
    WalkForwardWindow,
    _validated_signal_values,
    windows_from_mapping,
)


def load_dotenv_files() -> None:
    load_dotenv(Path.cwd() / ".env", override=False)
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def _read_symbol(data_dir: Path, symbol: str) -> pd.DataFrame:
    candidates = (
        data_dir / symbol / "prices.csv",
        data_dir / f"{symbol}.csv",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"no price file found for {symbol} under {data_dir}")

    frame = pd.read_csv(path)
    required = {"Date", "Open", "High", "Low", "Close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    frame["Date"] = pd.to_datetime(frame["Date"], utc=False).dt.normalize()
    if "Adjusted_close" in frame.columns:
        ratio = frame["Adjusted_close"].astype(float) / frame["Close"].astype(float)
    else:
        ratio = pd.Series(1.0, index=frame.index)
    for column in ("Open", "High", "Low", "Close"):
        frame[column] = frame[column].astype(float) * ratio
    frame = (
        frame[["Date", "Open", "High", "Low", "Close"]]
        .dropna()
        .drop_duplicates("Date", keep="last")
        .sort_values("Date")
    )
    return frame


def load_price_panel(
    data_dir: str | Path,
    symbols: Iterable[str],
    start: date | str | None = None,
    end: date | str | None = None,
) -> PricePanel:
    directory = Path(data_dir)
    raw_symbols = [str(symbol) for symbol in symbols]
    if len(set(raw_symbols)) != len(raw_symbols):
        raise ValueError("symbols must be unique")
    symbol_list = tuple(raw_symbols)
    if not symbol_list:
        raise ValueError("at least one symbol is required")
    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None

    frames: dict[str, pd.DataFrame] = {}
    source_last_dates: dict[str, pd.Timestamp] = {}
    for symbol in symbol_list:
        source = _read_symbol(directory, symbol)
        source_last_dates[symbol] = source["Date"].max()
        frame = source
        if start_ts is not None:
            frame = frame[frame["Date"] >= start_ts]
        if end_ts is not None:
            frame = frame[frame["Date"] <= end_ts]
        if frame.empty:
            raise ValueError(f"{symbol} has no observations in the requested range")
        frames[symbol] = frame.set_index("Date")

    common = frames[symbol_list[0]].index
    for frame in frames.values():
        common = common.intersection(frame.index, sort=False)
    common = common.sort_values()
    if len(common) < 2:
        raise ValueError("the selected symbols need at least two common trading dates")

    data_horizon = max(source_last_dates.values())
    effective_end = data_horizon if end_ts is None else min(end_ts, data_horizon)
    delisting = np.zeros((len(common), len(symbol_list)), dtype=np.bool_)
    closes = []
    for column, symbol in enumerate(symbol_list):
        frame = frames[symbol].loc[common]
        closes.append(frame["Close"].to_numpy(dtype=np.float64))
        if source_last_dates[symbol] < effective_end:
            delisting[-1, column] = True

    return PricePanel(
        dates=common.to_numpy(dtype="datetime64[D]"),
        symbols=symbol_list,
        open=np.column_stack([frames[s]["Open"].loc[common].to_numpy() for s in symbol_list]),
        high=np.column_stack([frames[s]["High"].loc[common].to_numpy() for s in symbol_list]),
        low=np.column_stack([frames[s]["Low"].loc[common].to_numpy() for s in symbol_list]),
        close=np.column_stack(closes),
        delisting_mask=delisting,
    )


def load_signal_panel(path: str | Path, prices: PricePanel) -> SignalPanel:
    frame = pd.read_csv(path)
    required = {"date", "symbol", "signal"}
    if not required.issubset(frame.columns):
        raise ValueError("signal CSV must contain date,symbol,signal")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str)
    numeric_signals = pd.to_numeric(frame["signal"], errors="raise").to_numpy()
    frame["signal"] = _validated_signal_values(numeric_signals)

    unknown_symbols = set(frame["symbol"]).difference(prices.symbols)
    if unknown_symbols:
        raise ValueError(f"signal file contains unknown symbols: {sorted(unknown_symbols)}")
    price_dates = pd.DatetimeIndex(prices.dates)
    unknown_dates = pd.DatetimeIndex(frame["date"].unique()).difference(price_dates)
    if len(unknown_dates):
        raise ValueError(f"signal dates are not in the price panel: {list(unknown_dates[:5])}")

    signals = np.zeros((len(prices.dates), len(prices.symbols)), dtype=np.int8)
    date_lookup = {value: index for index, value in enumerate(price_dates)}
    symbol_lookup = {value: index for index, value in enumerate(prices.symbols)}
    for row in frame.itertuples(index=False):
        signals[date_lookup[pd.Timestamp(row.date)], symbol_lookup[row.symbol]] = row.signal
    return SignalPanel(dates=prices.dates, symbols=prices.symbols, signals=signals)


def load_walk_forward(path: str | Path) -> tuple[WalkForwardWindow, ...]:
    frame = pd.read_csv(path)
    return windows_from_mapping(frame)


def _read_benchmark_frame(path: Path) -> BenchmarkSeries:
    frame = pd.read_csv(path)
    if "Date" in frame.columns:
        date_column = "Date"
    elif "date" in frame.columns:
        date_column = "date"
    else:
        raise ValueError(f"{path} needs a Date column")
    value_column = "Adjusted_close" if "Adjusted_close" in frame.columns else "Close"
    if "Close" not in frame.columns and "Adjusted_close" not in frame.columns:
        raise ValueError(f"{path} needs Close or Adjusted_close")
    frame[date_column] = pd.to_datetime(frame[date_column]).dt.normalize()
    frame[value_column] = pd.to_numeric(frame[value_column], errors="raise")
    frame = (
        frame[[date_column, value_column]]
        .dropna()
        .drop_duplicates(date_column, keep="last")
        .sort_values(date_column)
    )
    return BenchmarkSeries(
        dates=frame[date_column].to_numpy(dtype="datetime64[D]"),
        values=frame[value_column].to_numpy(dtype=np.float64),
    )


def load_benchmark(
    *,
    file: str | Path | None = None,
    ticker: str | None = None,
    data_dir: str | Path | None = None,
) -> BenchmarkSeries | None:
    if file is None and ticker is None:
        return None
    if file is not None:
        series = _read_benchmark_frame(Path(file))
        series.ticker = ticker or Path(file).stem
        return series
    if data_dir is None:
        raise ValueError("a benchmark ticker requires BT_DATA_DIR or --data-dir")
    path = Path(data_dir) / str(ticker) / "prices.csv"
    series = _read_benchmark_frame(path)
    series.ticker = ticker
    return series


def load_backtest_request(
    *,
    data_dir: str | Path,
    symbols: Iterable[str],
    signal_file: str | Path,
    start: date | str | None = None,
    end: date | str | None = None,
    config: BacktestConfig | None = None,
    walk_forward_file: str | Path | None = None,
    benchmark_ticker: str | None = None,
    benchmark_file: str | Path | None = None,
) -> BacktestRequest:
    if config is None:
        raise ValueError("BacktestConfig is required")
    prices = load_price_panel(data_dir, symbols, start=start, end=end)
    signals = load_signal_panel(signal_file, prices)
    benchmark_data_dir = os.environ.get("BT_DATA_DIR") or data_dir
    benchmark = load_benchmark(
        file=benchmark_file,
        ticker=benchmark_ticker,
        data_dir=benchmark_data_dir,
    )
    windows = load_walk_forward(walk_forward_file) if walk_forward_file else ()
    return BacktestRequest(
        prices=prices,
        signals=signals,
        config=config,
        benchmark=benchmark,
        walk_forward=windows,
    )


def config_from_json(path: str | Path | None, overrides: dict | None = None) -> BacktestConfig:
    values: dict = {}
    if path is not None:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("configuration JSON must contain an object")
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return BacktestConfig.model_validate(values)


def canonical_config_json(config: BacktestConfig) -> str:
    return json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def config_hash(config: BacktestConfig) -> str:
    return hash_bytes(canonical_config_json(config).encode("utf-8"))


def panel_hash(
    prices: PricePanel,
    signals: SignalPanel | None = None,
    benchmark: BenchmarkSeries | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(prices.dates.tobytes())
    digest.update("\0".join(prices.symbols).encode("utf-8"))
    for array in (prices.open, prices.high, prices.low, prices.close, prices.delisting_mask):
        digest.update(np.ascontiguousarray(array).tobytes())
    if signals is not None:
        digest.update(signals.dates.tobytes())
        digest.update("\0".join(signals.symbols).encode("utf-8"))
        digest.update(np.ascontiguousarray(signals.signals).tobytes())
    if benchmark is not None:
        digest.update(benchmark.dates.tobytes())
        digest.update(benchmark.values.tobytes())
    return digest.hexdigest()


def source_tree_hash(root: str | Path | None = None) -> str:
    base = Path(root or Path(__file__).resolve().parents[1])
    digest = hashlib.sha256()
    ignored = {"__pycache__", ".pytest_cache", ".numba_cache"}
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if any(part in ignored for part in path.parts) or path.suffix == ".pyc":
            continue
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
