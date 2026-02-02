from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

# Create paths for all the files 
@dataclass(frozen=True)
class SignalPaths:
    cleaned_prices_path: Path = Path("Data/Processed/prices.parquet")
    out_scores_path: Path = Path("Data/Processed/mom12_1_scores.parquet")
    out_monthly_prices_path: Path = Path("Data/Processed/prices_month_end.parquet")

#loading the daily price table
def load_prices_wide(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # optional: drop duplicate dates if any
    df = df[~df.index.duplicated(keep="last")]
    return df


def make_month_end_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Month-end prices derived from daily prices (last available trading day each month).
    Index will be calendar month-end timestamps produced by resample('M').
    """
    month_end = daily_prices.resample("ME").last()
    return month_end


def momentum_12_1(month_end_prices: pd.DataFrame) -> pd.DataFrame:
    """
    12–1 momentum at month-end t:
      mom_t = P_{t-1} / P_{t-12} - 1
    Implemented as shift(1)/shift(12) - 1 using month-end prices.
    """
    p_t_1 = month_end_prices.shift(1)
    p_t_12 = month_end_prices.shift(12)
    mom = (p_t_1 / p_t_12) - 1.0
    return mom


def coverage_report(scores: pd.DataFrame) -> pd.DataFrame:
    """
    For each month-end, how many tickers have a valid score and what % coverage that is.
    """
    n_tickers = scores.shape[1]
    valid = scores.notna().sum(axis=1)
    pct = valid / n_tickers
    rep = pd.DataFrame({"valid_tickers": valid, "pct_coverage": pct})
    return rep


def main() -> None:
    paths = SignalPaths()

    daily = load_prices_wide(paths.cleaned_prices_path)
    month_end = make_month_end_prices(daily)
    scores = momentum_12_1(month_end)

    # Save artifacts
    paths.out_monthly_prices_path.parent.mkdir(parents=True, exist_ok=True)
    month_end.to_parquet(paths.out_monthly_prices_path)
    scores.to_parquet(paths.out_scores_path)

    # Quick verification
    rep = coverage_report(scores)

    print("Saved month-end prices:", paths.out_monthly_prices_path)
    print("Saved 12–1 momentum scores:", paths.out_scores_path)
    print("\nCoverage (last 12 months):")
    print(rep.tail(12).to_string())

    # sanity checks:
    # 1) First ~12 months should be mostly NaN (needs 12 months lookback + 1 month skip)
    print("\nFirst 15 rows non-null counts:")
    print(scores.notna().sum(axis=1).head(15).to_string())


if __name__ == "__main__":
    main()