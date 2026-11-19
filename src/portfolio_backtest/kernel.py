from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit, prange


FILL_COLUMNS = (
    "bar_index",
    "instrument",
    "direction",
    "action",
    "quantity",
    "price",
    "notional",
    "commission",
    "cash_after",
    "realized_pnl",
    "reason",
)


@njit(cache=True, parallel=True)
def simulate_windows(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    delisting_mask: np.ndarray,
    signals: np.ndarray,
    windows: np.ndarray,
    max_fills: int,
    initial_cash: float,
    target_n: int,
    rebalance_days: int,
    execution_lag_bars: int,
    commission_rate: float,
    borrow_rate: float,
    maintenance_margin: float,
    delisting_haircut: float,
    settlement_days: int,
) -> tuple:
    """Run independent windows in parallel; each window is serial and deterministic."""

    window_count = windows.shape[0]
    instrument_count = signals.shape[1]
    max_length = 0
    for window_id in range(window_count):
        length = int(windows[window_id, 1] - windows[window_id, 0] + 1)
        if length > max_length:
            max_length = length

    equity = np.zeros((window_count, max_length), dtype=np.float64)
    cash_curve = np.zeros((window_count, max_length), dtype=np.float64)
    long_exposure = np.zeros((window_count, max_length), dtype=np.float64)
    short_exposure = np.zeros((window_count, max_length), dtype=np.float64)
    position_curve = np.zeros(
        (window_count, max_length, instrument_count), dtype=np.float64
    )
    fills = np.zeros((window_count, max_fills, 11), dtype=np.float64)
    fill_counts = np.zeros(window_count, dtype=np.int64)
    lengths = np.zeros(window_count, dtype=np.int64)

    pending_length = max_length + settlement_days + 2

    for window_id in prange(window_count):
        start = int(windows[window_id, 0])
        end = int(windows[window_id, 1])
        length = end - start + 1
        lengths[window_id] = length

        positions = np.zeros(instrument_count, dtype=np.float64)
        bucket_cash = np.zeros(instrument_count, dtype=np.float64)
        entry_notional = np.zeros(instrument_count, dtype=np.float64)
        entry_commission = np.zeros(instrument_count, dtype=np.float64)
        pending_cash = np.zeros(pending_length, dtype=np.float64)
        account_cash = initial_cash
        fill_count = 0

        for bar in range(start, end + 1):
            local_bar = bar - start
            if settlement_days > 0:
                account_cash += pending_cash[local_bar]
                pending_cash[local_bar] = 0.0

            signal_bar = bar - execution_lag_bars
            has_signal = (
                signal_bar >= start
                and signal_bar <= end
                and (signal_bar - start) % rebalance_days == 0
            )

            # Forced short buy-ins execute before signal-driven exits and entries.
            for instrument in range(instrument_count):
                shares = -positions[instrument]
                if shares <= 0.0 or delisting_mask[bar, instrument]:
                    continue
                threshold = bucket_cash[instrument] / (
                    shares * (1.0 + maintenance_margin)
                )
                if high_prices[bar, instrument] <= threshold:
                    continue
                cover_price = max(open_prices[bar, instrument], threshold)
                notional = shares * cover_price
                commission = notional * commission_rate
                cash_flow = bucket_cash[instrument] - notional - commission
                realized = (
                    bucket_cash[instrument]
                    - notional
                    - entry_notional[instrument]
                    - entry_commission[instrument]
                    - commission
                )
                if settlement_days == 0:
                    account_cash += cash_flow
                else:
                    release_bar = local_bar + settlement_days
                    if release_bar < pending_length:
                        pending_cash[release_bar] += cash_flow
                if fill_count < max_fills:
                    fills[window_id, fill_count, 0] = local_bar
                    fills[window_id, fill_count, 1] = instrument
                    fills[window_id, fill_count, 2] = -1.0
                    fills[window_id, fill_count, 3] = -1.0
                    fills[window_id, fill_count, 4] = shares
                    fills[window_id, fill_count, 5] = cover_price
                    fills[window_id, fill_count, 6] = notional
                    fills[window_id, fill_count, 7] = commission
                    fills[window_id, fill_count, 8] = account_cash
                    fills[window_id, fill_count, 9] = realized
                    fills[window_id, fill_count, 10] = 1.0
                    fill_count += 1
                positions[instrument] = 0.0
                bucket_cash[instrument] = 0.0
                entry_notional[instrument] = 0.0
                entry_commission[instrument] = 0.0

            # Scheduled closes and direction flips happen before entries.
            for instrument in range(instrument_count):
                current = positions[instrument]
                if current == 0.0:
                    continue
                desired = 0.0
                if has_signal:
                    desired = float(signals[signal_bar, instrument])
                is_delisting = bool(delisting_mask[bar, instrument])
                if not is_delisting:
                    if not has_signal:
                        continue
                    same_direction = (
                        (current > 0.0 and desired > 0.0)
                        or (current < 0.0 and desired < 0.0)
                    )
                    if same_direction:
                        continue

                price = close_prices[bar, instrument]
                if is_delisting:
                    price *= 1.0 - delisting_haircut
                if current > 0.0:
                    quantity = current
                    notional = quantity * price
                    commission = notional * commission_rate
                    cash_flow = notional - commission
                    realized = (
                        notional
                        - entry_notional[instrument]
                        - entry_commission[instrument]
                        - commission
                    )
                    direction = 1.0
                else:
                    quantity = -current
                    notional = quantity * price
                    commission = notional * commission_rate
                    cash_flow = bucket_cash[instrument] - notional - commission
                    realized = (
                        bucket_cash[instrument]
                        - notional
                        - entry_notional[instrument]
                        - entry_commission[instrument]
                        - commission
                    )
                    direction = -1.0

                if settlement_days == 0:
                    account_cash += cash_flow
                else:
                    release_bar = local_bar + settlement_days
                    if release_bar < pending_length:
                        pending_cash[release_bar] += cash_flow
                if fill_count < max_fills:
                    fills[window_id, fill_count, 0] = local_bar
                    fills[window_id, fill_count, 1] = instrument
                    fills[window_id, fill_count, 2] = direction
                    fills[window_id, fill_count, 3] = -1.0
                    fills[window_id, fill_count, 4] = quantity
                    fills[window_id, fill_count, 5] = price
                    fills[window_id, fill_count, 6] = notional
                    fills[window_id, fill_count, 7] = commission
                    fills[window_id, fill_count, 8] = account_cash
                    fills[window_id, fill_count, 9] = realized
                    fills[window_id, fill_count, 10] = 2.0 if is_delisting else 0.0
                    fill_count += 1
                positions[instrument] = 0.0
                bucket_cash[instrument] = 0.0
                entry_notional[instrument] = 0.0
                entry_commission[instrument] = 0.0

            if has_signal:
                held = 0
                for instrument in range(instrument_count):
                    if positions[instrument] != 0.0:
                        held += 1
                unfilled_slots = target_n - held
                if unfilled_slots > 0:
                    allocation = account_cash / unfilled_slots
                    for instrument in range(instrument_count):
                        desired = float(signals[signal_bar, instrument])
                        if (
                            desired == 0.0
                            or positions[instrument] != 0.0
                            or delisting_mask[bar, instrument]
                        ):
                            continue
                        price = close_prices[bar, instrument]
                        notional = allocation / (1.0 + commission_rate)
                        commission = notional * commission_rate
                        total_cost = notional + commission
                        if notional <= 0.0 or total_cost > account_cash + 1e-12:
                            continue
                        quantity = notional / price
                        account_cash -= total_cost
                        positions[instrument] = desired * quantity
                        entry_notional[instrument] = notional
                        entry_commission[instrument] = commission
                        if desired < 0.0:
                            bucket_cash[instrument] = 2.0 * notional
                        if fill_count < max_fills:
                            fills[window_id, fill_count, 0] = local_bar
                            fills[window_id, fill_count, 1] = instrument
                            fills[window_id, fill_count, 2] = desired
                            fills[window_id, fill_count, 3] = 1.0
                            fills[window_id, fill_count, 4] = quantity
                            fills[window_id, fill_count, 5] = price
                            fills[window_id, fill_count, 6] = notional
                            fills[window_id, fill_count, 7] = commission
                            fills[window_id, fill_count, 8] = account_cash
                            fills[window_id, fill_count, 9] = 0.0
                            fills[window_id, fill_count, 10] = 0.0
                            fill_count += 1

            # Borrow accrues after the bar's trades and modifies only the short bucket.
            for instrument in range(instrument_count):
                if positions[instrument] < 0.0:
                    bucket_cash[instrument] -= (
                        borrow_rate
                        / 252.0
                        * (-positions[instrument])
                        * close_prices[bar, instrument]
                    )

            marked_equity = account_cash
            for pending_index in range(local_bar + 1, pending_length):
                marked_equity += pending_cash[pending_index]
            long_value = 0.0
            short_value = 0.0
            for instrument in range(instrument_count):
                position = positions[instrument]
                if position == 0.0:
                    continue
                mark = close_prices[bar, instrument]
                if delisting_mask[bar, instrument]:
                    mark *= 1.0 - delisting_haircut
                if position > 0.0:
                    value = position * mark
                    marked_equity += value
                    long_value += value
                else:
                    cover_value = -position * mark
                    marked_equity += bucket_cash[instrument] - cover_value
                    short_value += cover_value
                position_curve[window_id, local_bar, instrument] = position

            equity[window_id, local_bar] = marked_equity
            cash_curve[window_id, local_bar] = account_cash
            long_exposure[window_id, local_bar] = long_value
            short_exposure[window_id, local_bar] = short_value

        fill_counts[window_id] = fill_count

    return (
        equity,
        cash_curve,
        long_exposure,
        short_exposure,
        position_curve,
        fills,
        fill_counts,
        lengths,
    )


def kernel_signature_summary() -> dict[str, Any]:
    signature = simulate_windows.nopython_signatures
    return {
        "compiled": bool(signature),
        "signature_count": len(signature),
        "parallel": simulate_windows.targetoptions.get("parallel", False),
        "nopython": True,
    }
