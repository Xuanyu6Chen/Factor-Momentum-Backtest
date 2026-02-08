import os
import pandas as pd

from .config import DATA_DIR_PROCESSED

def main():
    # 1) Load cleaned processed prices from result of data_clean.py 
    prices = pd.read_parquet(f"{DATA_DIR_PROCESSED}/prices.parquet")

    # 2) Make sure dates are datetime + sorted
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # 3) Compute daily simple returns: r_t = P_t / P_{t-1} - 1
    returns = prices.pct_change()

    # 4) Drop the first row (it will be NaN because there's no previous day)
    returns = returns.dropna(how="all")

    # 5) Save returns
    os.makedirs(DATA_DIR_PROCESSED, exist_ok=True)
    returns.to_parquet(f"{DATA_DIR_PROCESSED}/returns.parquet")

    # 6) Print quick summary
    print(f"Saved returns: {DATA_DIR_PROCESSED}/returns.parquet")
    print("Returns shape:", returns.shape)
    print(returns.head())
    print(returns.tail())

if __name__ == "__main__":
    main()