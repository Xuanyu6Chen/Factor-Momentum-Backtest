import numpy as np
import pandas as pd


def infer_rebalance_days(daily_weights: pd.DataFrame) -> pd.Series:
    """
    Rebalance day = any day where weights differ from previous day.
    """
    changed = (daily_weights != daily_weights.shift(1)).any(axis=1)
    return changed.fillna(False)


def compute_turnover_and_costs(
    daily_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    daily_weights: portfolio weights indexed by trading day (post-rebalance weights for that day)
    asset_returns: daily simple returns, same index/columns
    cost_bps: cost per $ traded, e.g. 10 = 0.10%

    Returns a DataFrame with:
      turnover (one-way, fraction of portfolio traded)
      cost     (fractional cost, e.g. 0.0005 = 5 bps)
    """
    daily_weights = daily_weights.sort_index()
    asset_returns = asset_returns.sort_index()

    # Align
    cols = daily_weights.columns.intersection(asset_returns.columns)
    w = daily_weights[cols].copy()
    r = asset_returns[cols].copy()

    # Ensure weights sum to 1 (defensive)
    w = w.div(w.sum(axis=1), axis=0).fillna(0.0)

    reb_days = infer_rebalance_days(w)

    turnover = pd.Series(0.0, index=w.index, name="turnover")

    # We compute turnover on rebalance day t using:
    # w_prev = weights held during day t (i.e., yesterday's post-rebalance weights)
    # r_t    = asset returns realized on day t
    # w_pre  = drifted pre-trade weights at close t
    # w_tgt  = new target weights set at close t (stored in w.loc[t])
    #
    # IMPORTANT: This assumes your portfolio return for day t was generated using w.shift(1).
    w_prev_all = w.shift(1)

    for t in w.index:
        if not reb_days.loc[t]:
            continue
        # First rebalance day has no previous holdings -> turnover stays 0
        w_prev = w_prev_all.loc[t]
        if w_prev.isna().all():
            continue

        r_t = r.loc[t].fillna(0.0)

        # portfolio gross return on day t using holdings w_prev
        port_ret_t = float((w_prev.fillna(0.0) * r_t).sum())

        # drifted pre-trade weights at close t
        denom = 1.0 + port_ret_t
        if denom <= 0:
            # extreme case; avoid divide-by-zero / nonsense
            continue
        w_pre = (w_prev.fillna(0.0) * (1.0 + r_t)) / denom

        # new target weights at t
        w_tgt = w.loc[t].fillna(0.0)

        turnover.loc[t] = 0.5 * float((w_tgt - w_pre).abs().sum())

    cost_rate = cost_bps / 10000.0
    cost = (turnover * cost_rate).rename("cost")

    return pd.concat([turnover, cost], axis=1)
