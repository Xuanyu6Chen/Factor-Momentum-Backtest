import os
import pandas as pd

from .config import DATA_DIR_RAW, DATA_DIR_PROCESSED

def main():
    # 1) Load raw prices (Adj Close only, tickers as columns)
    prices = pd.read_parquet(f"{DATA_DIR_RAW}/prices_raw.parquet")

    # 2) Make sure the index is a datetime index and sorted
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # 3) Drop duplicate dates if any 
    prices = prices[~prices.index.duplicated(keep="first")]

    # 4) Coverage filter: keep tickers with enough non-missing data (ensure enough days to analyze)
    coverage = prices.notna().mean(axis=0)      # % of days each ticker has data
    keep = coverage[coverage >= 0.95].index     # keep tickers with >=95% coverage
    prices = prices[keep]

    # 5) Forward-fill missing values 
    prices = prices.ffill()

    # 6) (Optional but useful) drop rows that are still all-NaN
    prices = prices.dropna(how="all")

    # 7) Save processed prices
    os.makedirs(DATA_DIR_PROCESSED, exist_ok=True)
    prices.to_parquet(f"{DATA_DIR_PROCESSED}/prices.parquet")

    # 8) summary
    print(f"Saved processed prices: {DATA_DIR_PROCESSED}/prices.parquet")
    print("Final shape:", prices.shape)
    print(prices.head())
    print(prices.tail())

if __name__ == "__main__":
    main()