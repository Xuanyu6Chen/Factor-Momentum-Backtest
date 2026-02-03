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

from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# CONFIG
# =========================

COST_BPS_DEFAULT = 10.0  # 10 bps = 0.10% per $ traded

# Based on your screenshot / outputs
RETURNS_FILE = "returns.parquet"
WEIGHTS_FILE = "weights_top10_eq.parquet"


# =========================
# Helpers
# =========================

def project_root() -> Path:
    """
    When running: PYTHONPATH=Src python -m factor_momentum.backtest
    __file__ is .../Src/factor_momentum/backtest.py
    parents:
      [0] factor_momentum
      [1] Src
      [2] project root
    """
    return Path(__file__).resolve().parents[2]


def next_trading_day(trading_days: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp:
    """
    Return the first trading day STRICTLY AFTER date d.
    If d is at/after the last trading day, clamp to the last trading day.
    """
    pos = trading_days.searchsorted(d, side="right")
    if pos >= len(trading_days):
        return trading_days[-1]
    return trading_days[pos]


def build_daily_weights(weights_rebal: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Expand month-end target weights into a daily weight panel.

    Logic:
    - Each rebalance row at date d becomes effective on next_trading_day(d).
    - The weights are held from that effective date until the day before the next effective date.
    """
    weights_rebal = weights_rebal.sort_index()
    weights_rebal.index = pd.to_datetime(weights_rebal.index)

    trading_days = pd.DatetimeIndex(pd.to_datetime(trading_days)).sort_values()

    tickers = list(weights_rebal.columns)
    w_daily = pd.DataFrame(0.0, index=trading_days, columns=tickers)

    eff_dates = [next_trading_day(trading_days, pd.Timestamp(d)) for d in weights_rebal.index]

    for i, (d, eff) in enumerate(zip(weights_rebal.index, eff_dates)):
        start = eff
        if i < len(eff_dates) - 1:
            end = eff_dates[i + 1]
            mask = (w_daily.index >= start) & (w_daily.index < end)
        else:
            mask = (w_daily.index >= start)

        w_row = weights_rebal.loc[d].astype(float)

        # Defensive: normalize to sum=1 if possible
        s = float(w_row.sum())
        if s != 0:
            w_row = w_row / s
        else:
            # If a rebalance row is all zero, keep it all zero (but that would be weird)
            w_row = w_row.fillna(0.0)

        w_daily.loc[mask, :] = w_row.values

    return w_daily


def infer_rebalance_days(w_daily: pd.DataFrame) -> pd.Series:
    """
    Rebalance day = any day where weights differ from previous day.
    """
    changed = (w_daily != w_daily.shift(1)).any(axis=1)
    return changed.fillna(False)


def compute_portfolio_returns(w_daily: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series:
    """
    Gross daily portfolio return:
      gross_ret[t] = sum_i w[t-1,i] * r[t,i]
    """
    cols = w_daily.columns.intersection(asset_returns.columns)
    w = w_daily[cols].copy().sort_index()
    r = asset_returns[cols].copy().sort_index()

    # Normalize weights row-wise (if a row sums to zero, keep zeros)
    row_sum = w.sum(axis=1)
    w = w.div(row_sum.replace(0.0, np.nan), axis=0).fillna(0.0)

    gross_ret = (w.shift(1) * r).sum(axis=1)
    gross_ret.name = "gross_ret"
    return gross_ret


def compute_turnover_and_costs(
    w_daily: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cost_bps: float = COST_BPS_DEFAULT,
) -> pd.DataFrame:
    """
    Compute turnover and transaction cost per day.

    Turnover is computed on rebalance days using DRIFTED pre-trade weights:
      w_pre = w_prev * (1+r) / (1+portfolio_return)
      turnover = 0.5 * sum_i |w_target - w_pre|

    Returns df columns:
      turnover: fraction of portfolio traded
      cost: turnover * (cost_bps/10000)
    """
    w_daily = w_daily.sort_index()
    asset_returns = asset_returns.sort_index()

    cols = w_daily.columns.intersection(asset_returns.columns)
    w = w_daily[cols].copy()
    r = asset_returns[cols].copy()

    # Normalize weights (keep zeros as zeros)
    row_sum = w.sum(axis=1)
    w = w.div(row_sum.replace(0.0, np.nan), axis=0).fillna(0.0)

    reb_days = infer_rebalance_days(w)

    turnover = pd.Series(0.0, index=w.index, name="turnover")
    w_prev_all = w.shift(1)

    for t in w.index:
        if not reb_days.loc[t]:
            continue

        w_prev = w_prev_all.loc[t]
        if w_prev.isna().all() or float(w_prev.abs().sum()) == 0.0:
            # First rebalance / no previous holdings
            continue

        r_t = r.loc[t].fillna(0.0)

        # Portfolio return for day t using w_prev holdings
        port_ret_t = float((w_prev.fillna(0.0) * r_t).sum())
        denom = 1.0 + port_ret_t
        if denom <= 0:
            continue

        # Drifted pre-trade weights at close t
        w_pre = (w_prev.fillna(0.0) * (1.0 + r_t)) / denom

        # New target weights set at close t
        w_tgt = w.loc[t].fillna(0.0)

        turnover.loc[t] = 0.5 * float((w_tgt - w_pre).abs().sum())

    cost_rate = cost_bps / 10000.0
    cost = (turnover * cost_rate).rename("cost")

    return pd.concat([turnover, cost], axis=1)


# =========================
# Main
# =========================

def main(cost_bps: float = COST_BPS_DEFAULT) -> None:
    root = project_root()

    data_dir = root / "Data" / "Processed"
    results_dir = root / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    returns_path = data_dir / RETURNS_FILE
    weights_path = data_dir / WEIGHTS_FILE

    if not returns_path.exists():
        raise FileNotFoundError(f"Missing: {returns_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing: {weights_path}")

    # Load data
    returns = pd.read_parquet(returns_path)
    weights_rebal = pd.read_parquet(weights_path)

    returns.index = pd.to_datetime(returns.index)
    weights_rebal.index = pd.to_datetime(weights_rebal.index)

    returns = returns.sort_index()
    weights_rebal = weights_rebal.sort_index()

    print(f"Last returns day: {returns.index.max()}")
    print(f"Last rebalance day: {weights_rebal.index.max()}")

    # Build daily weights over all trading days
    w_daily = build_daily_weights(weights_rebal, returns.index)

    # -----------------------------
    # FIX: Trim pre-investment days
    # -----------------------------
    w_sum = w_daily.sum(axis=1)

    first_nonzero = w_sum[w_sum > 0].index.min()
    if pd.isna(first_nonzero):
        raise ValueError("Weights are zero for all days. Check weights_rebal generation.")

    # Keep only the period where the portfolio is actually investable
    w_daily = w_daily.loc[first_nonzero:]
    returns = returns.loc[first_nonzero:]

    # Diagnostics
    w_sum2 = w_daily.sum(axis=1)
    print("min sum", float(w_sum2.min()), "max sum", float(w_sum2.max()))
    print("days with sum==0:", int((w_sum2 == 0).sum()))
    print("first nonzero:", first_nonzero)

    # Compute gross returns
    gross_ret = compute_portfolio_returns(w_daily, returns)

    # Drop the first day with NaN (because of shift(1))
    gross_ret = gross_ret.dropna()
    w_daily = w_daily.loc[gross_ret.index]
    returns = returns.loc[gross_ret.index]

    # Turnover + costs
    tc = compute_turnover_and_costs(w_daily, returns, cost_bps=cost_bps)

    # Cost timing: trade at close t -> cost hits before day t+1 return
    cost_to_apply = tc["cost"].shift(1).fillna(0.0)

    # Net wealth update
    net_growth = (1.0 - cost_to_apply) * (1.0 + gross_ret.fillna(0.0))
    net_ret = net_growth - 1.0
    net_ret.name = "net_ret"

    # Equity curves
    gross_equity = (1.0 + gross_ret.fillna(0.0)).cumprod()
    gross_equity.name = "gross_equity"
    net_equity = (1.0 + net_ret.fillna(0.0)).cumprod()
    net_equity.name = "net_equity"

    # Save outputs
    strategy_out = pd.DataFrame({
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "turnover": tc["turnover"].reindex(gross_ret.index).fillna(0.0),
        "cost": tc["cost"].reindex(gross_ret.index).fillna(0.0),
    })

    equity_out = pd.DataFrame({
        "gross_equity": gross_equity,
        "net_equity": net_equity,
    })

    strategy_out_path = results_dir / "strategy_returns.parquet"
    equity_out_path = results_dir / "equity_curve.parquet"
    strategy_out.to_parquet(strategy_out_path)
    equity_out.to_parquet(equity_out_path)

    # CSV copies for quick inspection
    strategy_out.to_csv(results_dir / "strategy_returns.csv")
    equity_out.to_csv(results_dir / "equity_curve.csv")

    print("Backtest complete ✅")
    print(f"Saved: {strategy_out_path}")
    print(f"Saved: {equity_out_path}")


if __name__ == "__main__":
    main()
