"""
backtest.py

Part 6: Transaction costs + turnover

Assumptions:
- weights_rebal contains target portfolio weights at month-end rebalance dates.
- Those weights become effective on the NEXT trading day after the rebalance date.
- Daily portfolio return uses yesterday's weights:
    gross_ret[t] = sum_i w[t-1,i] * r[t,i]
- Transaction costs are paid on rebalance days at the close, so they reduce wealth before the next day.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR_PROCESSED,
    RESULTS_DIR,
    WEIGHTS_PATH,
    RETURNS_PATH,
    COST_BPS_GRID,
    STRATEGY_RETURNS_FILE,
    EQUITY_CURVE_FILE,
    META_FILE,
)


@dataclass(frozen=True)
class BacktestInputs:
    weights_m: pd.DataFrame        # index: rebalance dates (month-end), columns: tickers, values: weights
    daily_returns: pd.DataFrame    # index: daily dates, columns: tickers, values: simple returns


def _assert_exists(path: Path, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}\n"
            f"Check your config.py paths. From your file tree, you likely want:\n"
            f"  WEIGHTS_PATH = DATA_DIR_PROCESSED / 'weights_top10_eq.parquet'\n"
            f"  RETURNS_PATH = DATA_DIR_PROCESSED / 'returns.parquet'\n"
        )


def _load_inputs() -> BacktestInputs:
    _assert_exists(WEIGHTS_PATH, "weights file (monthly weights)")
    _assert_exists(RETURNS_PATH, "returns file (daily returns)")

    weights_m = pd.read_parquet(WEIGHTS_PATH)
    daily_returns = pd.read_parquet(RETURNS_PATH)

    # Normalize index types
    weights_m.index = pd.to_datetime(weights_m.index)
    daily_returns.index = pd.to_datetime(daily_returns.index)

    # Defensive: ensure numeric, fill NaNs
    weights_m = weights_m.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    daily_returns = daily_returns.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Align tickers intersection only
    common = sorted(set(weights_m.columns).intersection(daily_returns.columns))
    if not common:
        raise ValueError(
            "No overlapping tickers between weights and daily returns.\n"
            f"weights columns sample: {list(weights_m.columns)[:10]}\n"
            f"returns columns sample: {list(daily_returns.columns)[:10]}\n"
            "If you have tickers like BRK-B vs BRK.B, normalize them upstream."
        )
    weights_m = weights_m[common].copy()
    daily_returns = daily_returns[common].copy()

    # Sort indices
    weights_m = weights_m.sort_index()
    daily_returns = daily_returns.sort_index()

    return BacktestInputs(weights_m=weights_m, daily_returns=daily_returns)


def _next_trading_day(daily_index: pd.DatetimeIndex, dt: pd.Timestamp) -> pd.Timestamp | None:
    """
    Return first trading day strictly AFTER dt using searchsorted.
    """
    pos = daily_index.searchsorted(dt, side="right")
    if pos >= len(daily_index):
        return None
    return daily_index[pos]


def _build_daily_weights(weights_m: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Expand month-end weights to daily weights:
    - weights at rebalance date r become effective on next trading day after r
    - then forward-filled until next rebalance effective day
    """
    rebalance_dates = pd.DatetimeIndex(pd.to_datetime(weights_m.index)).sort_values()

    effective_rows = []
    for r in rebalance_dates:
        eff = _next_trading_day(daily_index, r)
        if eff is not None:
            effective_rows.append((r, eff))

    if not effective_rows:
        raise ValueError(
            "All rebalance dates are after the last available trading day in returns.\n"
            "Check the date ranges of your returns vs weights."
        )

    eff_map = pd.DataFrame(effective_rows, columns=["rebalance_date", "effective_date"]).set_index("rebalance_date")

    # weights on effective dates
    w_eff = weights_m.loc[eff_map.index].copy()
    w_eff.index = pd.DatetimeIndex(eff_map["effective_date"].values)

    # reindex to all daily dates and forward fill
    w_daily = w_eff.reindex(daily_index).ffill().fillna(0.0)

    # Ensure fully invested on days with nonzero weights (normalize minor float drift)
    row_sum = w_daily.sum(axis=1)
    nonzero = row_sum > 0
    off = nonzero & (~np.isclose(row_sum.values, 1.0))
    if off.any():
        w_daily.loc[off] = w_daily.loc[off].div(row_sum[off], axis=0)

    return w_daily


def run_backtest(cost_bps: int, out_dir: Path) -> None:
    """
    Run backtest with turnover-based trading costs:
      turnover(t) = sum_i |w(t) - w(t-1)|
      cost(t) = turnover(t) * (cost_bps / 10,000)
      net_return(t) = sum_i w(t) * r_i(t) - cost(t)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = _load_inputs()
    daily_index = inp.daily_returns.index

    # Build daily weights with next-day application to avoid look-ahead
    w_daily = _build_daily_weights(inp.weights_m, daily_index)

    # Gross daily strategy return: sum_i w(t) * r_i(t)
    gross = (w_daily * inp.daily_returns).sum(axis=1)

    # Trading costs via turnover
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative.")
    bps_rate = cost_bps / 10_000.0

    w_prev = w_daily.shift(1).fillna(0.0)
    turnover = (w_daily - w_prev).abs().sum(axis=1)
    cost = turnover * bps_rate

    net = gross - cost

    # Save strategy returns
    net_df = net.to_frame("strategy_return")
    net_df.to_parquet(out_dir / STRATEGY_RETURNS_FILE)

    # Save equity curve
    equity = (1.0 + net).cumprod().rename("equity").to_frame()
    equity.to_parquet(out_dir / EQUITY_CURVE_FILE)

    # Meta for traceability
    meta_lines = [
        f"weights_path={WEIGHTS_PATH}",
        f"returns_path={RETURNS_PATH}",
        f"cost_bps={cost_bps}",
        f"bps_rate={bps_rate}",
        f"start={net.index.min().date()}",
        f"end={net.index.max().date()}",
        f"mean_daily_return={net.mean():.10f}",
        f"daily_vol={net.std(ddof=0):.10f}",
        f"turnover_mean={turnover.mean():.10f}",
        f"turnover_median={turnover.median():.10f}",
    ]
    (out_dir / META_FILE).write_text("\n".join(meta_lines), encoding="utf-8")

    print(f"Backtest complete ✅ cost={cost_bps} bps")
    print(f"Saved: {out_dir / STRATEGY_RETURNS_FILE}")
    print(f"Saved: {out_dir / EQUITY_CURVE_FILE}")


def main() -> None:
    """
    Default run (single backtest). Keeps compatibility with:
      PYTHONPATH=Src python -m factor_momentum.backtest
    """
    run_backtest(cost_bps=0, out_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()
