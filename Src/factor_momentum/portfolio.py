from pathlib import Path
import pandas as pd


def _ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    # If columns are MultiIndex, keep the last level (often the ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(-1)
    return df


def build_weights_topk_equal(mom_scores: pd.DataFrame,
                            daily_returns: pd.DataFrame,
                            top_k: int = 10) -> pd.DataFrame:
    """
    Build a long-only Top-K equal-weight portfolio from momentum scores.

    Concept:
      - At each rebalance date (the dates in mom_scores), rank tickers by momentum.
      - Select the Top-K tickers (highest scores).
      - Assign equal weights to the selected tickers (1/K each; 0 for others).
      - Hold these weights every day until the next rebalance date (forward-fill).
      - Return a DAILY weights DataFrame aligned to daily_returns.index.

    Inputs:
      mom_scores:
        DataFrame indexed by rebalance dates (typically month-end trading days),
        columns = tickers, values = momentum scores (higher = better).
      daily_returns:
        DataFrame indexed by daily trading days,
        columns = tickers, values = daily returns.
      top_k:
        Number of tickers to hold at each rebalance.

    Output:
      weights:
        DataFrame indexed by daily trading days (same index as daily_returns),
        columns = tickers, values = portfolio weights (long-only, sums to ~1 when invested).
    """
    
    mom_scores = _ensure_flat_columns(mom_scores).sort_index()
    daily_returns = _ensure_flat_columns(daily_returns).sort_index()

    # Keeps only tickers that exist in both tables.
    tickers = mom_scores.columns.intersection(daily_returns.columns)
    if len(tickers) == 0:
        raise ValueError("No overlapping tickers between momentum scores and returns.")

    mom_scores = mom_scores[tickers]
    daily_returns = daily_returns[tickers]

    # Check if Rebalance dates is in a non-trading day 
    rebalance_dates = mom_scores.index.intersection(daily_returns.index)
    if len(rebalance_dates) == 0:
        raise ValueError("No momentum-score dates match daily returns dates. Check alignment.")

    w_rebal = pd.DataFrame(0.0, index=rebalance_dates, columns=tickers)

    # For each rebalance date: select Top-K winners and assign equal weights
    for d in rebalance_dates:
        scores = mom_scores.loc[d].dropna()
        if scores.empty:
            continue
        # Pick top-K
        winners = scores.nlargest(min(top_k, len(scores))).index
        # Assign equal weight to winners
        w_rebal.loc[d, winners] = 1.0 / len(winners)

    weights = w_rebal.reindex(daily_returns.index).ffill().fillna(0.0) # reindex(daily_returns.index) creates rows for every daily date

    # Normalize to sum to 1 when invested
    s = weights.sum(axis=1)
    invested = s > 0
    weights.loc[invested] = weights.loc[invested].div(s[invested], axis=0)

    return weights


def main(top_k: int = 10) -> None:
    project_root = Path(__file__).resolve().parents[2]
    processed = project_root / "Data" / "Processed"

    mom_path = processed / "mom12_1_scores.parquet"
    ret_path = processed / "returns.parquet"

    mom = pd.read_parquet(mom_path)
    rets = pd.read_parquet(ret_path)

    weights = build_weights_topk_equal(mom, rets, top_k=top_k)

    out_path = processed / f"weights_top{top_k}_eq.parquet"
    weights.to_parquet(out_path)

    print(f"Saved: {out_path}")
    print("Last 5 days weight sums:")
    print(weights.sum(axis=1).tail())
    print("Last 5 days #positions:")
    print((weights > 0).sum(axis=1).tail())
