from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ArrayConfig = ConfigDict(arbitrary_types_allowed=True)


def _date_array(values: Any) -> np.ndarray:
    dates = np.asarray(values, dtype="datetime64[D]")
    if dates.ndim != 1:
        raise ValueError("dates must be a one-dimensional array")
    if dates.size > 1 and np.any(dates[1:] <= dates[:-1]):
        raise ValueError("dates must be strictly increasing")
    return np.ascontiguousarray(dates)


def _matrix(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(array)


class PricePanel(BaseModel):
    """Aligned, adjusted OHLC panels using the shared trading calendar."""

    model_config = ArrayConfig

    dates: np.ndarray
    symbols: tuple[str, ...]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    delisting_mask: np.ndarray | None = None

    @field_validator("dates", mode="before")
    @classmethod
    def validate_dates(cls, value: Any) -> np.ndarray:
        return _date_array(value)

    @field_validator("symbols", mode="before")
    @classmethod
    def validate_symbols(cls, value: Any) -> tuple[str, ...]:
        symbols = tuple(str(item) for item in value)
        if not symbols:
            raise ValueError("at least one symbol is required")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique")
        return symbols

    @field_validator("open", "high", "low", "close", mode="before")
    @classmethod
    def validate_matrix(cls, value: Any, info: Any) -> np.ndarray:
        return _matrix(value, info.field_name)

    @model_validator(mode="after")
    def validate_panel(self) -> "PricePanel":
        shape = (len(self.dates), len(self.symbols))
        if not shape[0]:
            raise ValueError("the price panel must contain at least one date")
        for name in ("open", "high", "low", "close"):
            array = getattr(self, name)
            if array.shape != shape:
                raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
            if np.any(array <= 0):
                raise ValueError(f"{name} must be strictly positive")
        if np.any(self.high + 1e-12 < np.maximum(self.open, self.close)):
            raise ValueError("high must be at least open and close")
        if np.any(self.low - 1e-12 > np.minimum(self.open, self.close)):
            raise ValueError("low must be at most open and close")

        if self.delisting_mask is None:
            mask = np.zeros(shape, dtype=np.bool_)
        else:
            mask = np.asarray(self.delisting_mask, dtype=np.bool_)
            if mask.shape != shape:
                raise ValueError(f"delisting_mask shape must be {shape}")
            mask = np.ascontiguousarray(mask)
        object.__setattr__(self, "delisting_mask", mask)
        return self

    def slice(self, start: int, stop: int) -> "PricePanel":
        return PricePanel(
            dates=self.dates[start:stop],
            symbols=self.symbols,
            open=self.open[start:stop],
            high=self.high[start:stop],
            low=self.low[start:stop],
            close=self.close[start:stop],
            delisting_mask=self.delisting_mask[start:stop],
        )


class SignalPanel(BaseModel):
    """Dense desired directions: 1 long, -1 short, and 0 flat."""

    model_config = ArrayConfig

    dates: np.ndarray
    symbols: tuple[str, ...]
    signals: np.ndarray

    @field_validator("dates", mode="before")
    @classmethod
    def validate_dates(cls, value: Any) -> np.ndarray:
        return _date_array(value)

    @field_validator("symbols", mode="before")
    @classmethod
    def validate_symbols(cls, value: Any) -> tuple[str, ...]:
        symbols = tuple(str(item) for item in value)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("signal symbols must be unique and non-empty")
        return symbols

    @field_validator("signals", mode="before")
    @classmethod
    def validate_signals(cls, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.int8)
        if array.ndim != 2:
            raise ValueError("signals must be a two-dimensional array")
        if not np.isin(array, (-1, 0, 1)).all():
            raise ValueError("signals may only contain -1, 0, and 1")
        return np.ascontiguousarray(array)

    @model_validator(mode="after")
    def validate_shape(self) -> "SignalPanel":
        expected = (len(self.dates), len(self.symbols))
        if self.signals.shape != expected:
            raise ValueError(f"signals shape must be {expected}")
        return self


class BenchmarkSeries(BaseModel):
    model_config = ArrayConfig

    dates: np.ndarray
    values: np.ndarray
    ticker: str | None = None

    @field_validator("dates", mode="before")
    @classmethod
    def validate_dates(cls, value: Any) -> np.ndarray:
        return _date_array(value)

    @field_validator("values", mode="before")
    @classmethod
    def validate_values(cls, value: Any) -> np.ndarray:
        values = np.asarray(value, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("benchmark values must be finite, positive, and one-dimensional")
        return np.ascontiguousarray(values)

    @model_validator(mode="after")
    def validate_shape(self) -> "BenchmarkSeries":
        if len(self.dates) != len(self.values):
            raise ValueError("benchmark dates and values must have equal length")
        if not len(self.dates):
            raise ValueError("benchmark series may not be empty")
        return self


class WalkForwardWindow(BaseModel):
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_range(self) -> "WalkForwardWindow":
        if self.end_date < self.start_date:
            raise ValueError("walk-forward end_date must be on or after start_date")
        return self


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_cash: float = Field(gt=0)
    target_n: int = Field(gt=0)
    rebalance_days: int = Field(default=5, gt=0)
    execution_lag_bars: int = Field(default=1, ge=0)
    commission_rate: float = Field(default=0.001, ge=0, lt=1)
    borrow_rate: float = Field(default=0.10, ge=0)
    maintenance_margin: float = Field(default=0.30, ge=0, lt=1)
    delisting_haircut: float = Field(default=1.0, ge=0, le=1)
    settlement_days: int = Field(default=0, ge=0)
    annualization: int = Field(default=252, gt=0)
    risk_free_rate: float = 0.0
    mode: Literal["full", "walk_forward", "both"] = "both"
    memory_limit_bytes: int = Field(default=2_000_000_000, gt=0)


class BacktestRequest(BaseModel):
    model_config = ArrayConfig

    prices: PricePanel
    signals: SignalPanel
    config: BacktestConfig
    benchmark: BenchmarkSeries | None = None
    walk_forward: tuple[WalkForwardWindow, ...] = ()

    @model_validator(mode="after")
    def validate_alignment(self) -> "BacktestRequest":
        if not np.array_equal(self.prices.dates, self.signals.dates):
            raise ValueError("price and signal dates must align exactly")
        if self.prices.symbols != self.signals.symbols:
            raise ValueError("price and signal symbols must align exactly")
        if self.config.target_n > len(self.prices.symbols):
            raise ValueError("target_n cannot exceed the number of instruments")
        _validate_windows(self.walk_forward)
        return self


class ScopeResult(BaseModel):
    model_config = ArrayConfig

    scope: str
    start_date: date
    end_date: date
    equity_curve: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]


class WalkForwardSummary(BaseModel):
    mean: dict[str, float | None]
    median: dict[str, float | None]
    positive_window_rate: dict[str, float | None]


class BacktestResult(BaseModel):
    model_config = ArrayConfig

    run_id: str
    config: BacktestConfig
    full_window: ScopeResult | None = None
    walk_forward: tuple[ScopeResult, ...] = ()
    walk_forward_summary: WalkForwardSummary | None = None
    benchmark: BenchmarkSeries | None = None
    database_path: Path | None = None
    results_dir: Path | None = None


def _validate_windows(windows: tuple[WalkForwardWindow, ...]) -> None:
    ordered = sorted(windows, key=lambda item: (item.start_date, item.end_date))
    if list(windows) != ordered:
        raise ValueError("walk-forward windows must be sorted by start and end date")
    for previous, current in zip(ordered, ordered[1:]):
        if current.start_date <= previous.end_date:
            raise ValueError("walk-forward windows must be sorted and non-overlapping")


def windows_from_mapping(values: Any) -> tuple[WalkForwardWindow, ...]:
    if values is None:
        return ()
    if isinstance(values, pd.DataFrame):
        records = values.to_dict(orient="records")
    else:
        records = list(values)
    windows = []
    for record in records:
        if isinstance(record, WalkForwardWindow):
            windows.append(record)
            continue
        start = record.get("start_date", record.get("start"))
        end = record.get("end_date", record.get("end"))
        if start is None or end is None:
            raise ValueError("walk-forward rows need start_date and end_date")
        windows.append(WalkForwardWindow(start_date=start, end_date=end))
    _validate_windows(tuple(windows))
    return tuple(windows)
