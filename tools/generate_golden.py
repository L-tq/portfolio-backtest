from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from portfolio_backtest.engine import execute_backtest
from tests.golden_scenario import golden_request


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "golden"
    output.mkdir(parents=True, exist_ok=True)
    result = execute_backtest(golden_request())
    assert result.full_window is not None
    scope = result.full_window
    scope.equity_curve[
        ["date", "equity", "cash", "long_exposure", "short_exposure", "drawdown"]
    ].to_csv(output / "equity.csv", index=False, float_format="%.15g")
    scope.trades[
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
    ].to_csv(output / "fills.csv", index=False, float_format="%.15g")
    scope.positions[["date", "symbol", "shares"]].to_csv(
        output / "positions.csv",
        index=False,
        float_format="%.15g",
    )
    (output / "metrics.json").write_text(
        json.dumps(scope.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
