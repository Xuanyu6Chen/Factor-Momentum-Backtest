from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from .config import DATA_DIR_PROCESSED

RESULTS_DIR = "Results"


# =========================
# Utilities
# =========================

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def next_trading_day(trading_days: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp:
    """
    Return the first trading day strictly AFTER date d.
    """
    pos = trading_days.searchsorted(d, side="right")
    if pos >= len(trading_days):
        raise ValueError(f"No trading day after {d}.")
    return trading_days[pos]


def normalize_weights_rowwise(w: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaNs with 0 and normalize each row to sum to 1.
    If a row is all zeros, keep it all zeros.
    """
    w = w.fillna(0.0)
    s = w.sum(axis=1).replace(0, np.nan)
    return w.div(s, axis=0).fillna(0.0)


# =========================
# Backtest Engine
# =========================

def build_daily_weights(weights_rebal: pd.DataFrame, trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Convert rebalance-date weights into daily held weights.

    Rules:
      - weights formed at rebalance date d become effective on the NEXT trading day
      - weights are held constant (ffill) until next rebalance effective date
      - if d is the last trading day in the dataset, skip it (no next trading day exists)
    """
    weights_rebal = _ensure_dt_index(weights_rebal)
    trading_index = pd.DatetimeIndex(pd.to_datetime(trading_index)).sort_values()

    last_day = trading_index[-1]

    # Build mapping with end-of-sample protection
    reb_dates = []
    eff_dates = []
    for d in weights_rebal.index:
        d = pd.Timestamp(d)
        if d >= last_day:
            # Can't apply this rebalance because there is no future trading day
            continue
        reb_dates.append(d)
        eff_dates.append(next_trading_day(trading_index, d))

    if len(reb_dates) == 0:
        raise ValueError("All rebalance dates are at/after the last trading day. Nothing to backtest.")

    weights_rebal_valid = weights_rebal.loc[reb_dates].copy()
    w_eff = weights_rebal_valid.copy()
    w_eff.index = pd.DatetimeIndex(eff_dates)

    # If multiple rebalances map to same effective date, keep the last
    w_eff = w_eff[~w_eff.index.duplicated(keep="last")]

    # Expand to daily weights by forward-fill
    w_daily = w_eff.reindex(trading_index).ffill().fillna(0.0)

    return w_daily


def compute_strategy_returns(returns: pd.DataFrame, w_daily: pd.DataFrame) -> pd.Series:
    """
    Daily strategy return:
      r_strat[t] = sum_i w[i, t-1] * r[i, t]
    """
    returns = _ensure_dt_index(returns)
    w_daily = _ensure_dt_index(w_daily).reindex(returns.index).fillna(0.0)

    common = returns.columns.intersection(w_daily.columns)
    returns = returns[common]
    w_daily = w_daily[common]

    strat_ret = (w_daily.shift(1) * returns).sum(axis=1).fillna(0.0)
    strat_ret.name = "strategy_return"
    return strat_ret


def run_lookahead_checks(
    returns: pd.DataFrame,
    weights_rebal: pd.DataFrame,
    w_daily: pd.DataFrame,
    strat_ret: pd.Series,
) -> None:
    """
    Checks to catch timing mistakes / look-ahead bias.
    """
    returns = _ensure_dt_index(returns)
    weights_rebal = _ensure_dt_index(weights_rebal)
    w_daily = _ensure_dt_index(w_daily)

    trading_days = returns.index
    last_day = trading_days[-1]

    # Use the SAME rule as build_daily_weights: only rebalance dates strictly before last trading day are valid
    valid_reb = [pd.Timestamp(d) for d in weights_rebal.index if pd.Timestamp(d) < last_day]
    if len(valid_reb) == 0:
        raise AssertionError("No valid rebalance dates before the last trading day.")

    eff_dates = [next_trading_day(trading_days, d) for d in valid_reb]

    # Check 1: effective date strictly after rebalance date
    assert all(e > d for e, d in zip(eff_dates, valid_reb)), \
        "Look-ahead risk: effective date is not strictly after rebalance date."

    # Check 2: strategy should NOT match same-day weights version exactly (unless always zero exposure)
    common = returns.columns.intersection(w_daily.columns)
    same_day = (w_daily[common] * returns[common]).sum(axis=1).fillna(0.0)
    if (w_daily.abs().sum(axis=1) > 0).any():
        assert not np.allclose(same_day.values, strat_ret.values), \
            "Strategy returns match same-day weights. Did you forget shift(1)?"

    # Check 3: before first effective date, weights should be zero
    first_eff = min(pd.to_datetime(eff_dates))
    before = w_daily.loc[w_daily.index < first_eff]
    if len(before) > 0:
        assert (before.abs().sum(axis=1) == 0).all(), \
            "Non-zero weights before first effective date. Timing is likely wrong."

    print("✅ Look-ahead checks passed.")


# =========================
# Main
# =========================

def main():
    processed_dir = Path(DATA_DIR_PROCESSED)

    returns_path = processed_dir / "returns.parquet"
    weights_path = processed_dir / "weights_top10_eq.parquet"

    if not returns_path.exists():
        raise FileNotFoundError(f"Missing {returns_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path}")

    returns = pd.read_parquet(returns_path)
    weights_rebal = pd.read_parquet(weights_path)

    returns = _ensure_dt_index(returns)
    weights_rebal = _ensure_dt_index(weights_rebal)

    print("Last returns day:", returns.index.max())
    print("Last rebalance day:", weights_rebal.index.max())

    # Ensure weights are clean long-only (sum to 1)
    weights_rebal = normalize_weights_rowwise(weights_rebal)

    # Build daily weights and compute returns
    w_daily = build_daily_weights(weights_rebal, returns.index)
    strat_ret = compute_strategy_returns(returns, w_daily)

    # Checks
    run_lookahead_checks(returns, weights_rebal, w_daily, strat_ret)

    # Equity curve
    equity = (1.0 + strat_ret).cumprod()
    equity.name = "equity_curve"

    # Save outputs
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    strat_ret.to_frame().to_parquet(out_dir / "strategy_returns.parquet")
    equity.to_frame().to_parquet(out_dir / "equity_curve.parquet")
    strat_ret.to_frame().to_csv(out_dir / "strategy_returns.csv")

    print("Backtest complete ✅")
    print(f"Saved: {out_dir / 'strategy_returns.parquet'}")
    print(f"Saved: {out_dir / 'equity_curve.parquet'}")


if __name__ == "__main__":
    main()
