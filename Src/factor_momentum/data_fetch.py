import os
import pandas as pd
import yfinance as yf

from .config import TICKERS, START_DATE, END_DATE, DATA_DIR_RAW

def main():
    df = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        progress=False,
        group_by="column",
        auto_adjust=False,  # prices are not adjusted from dividends and others
        actions=False,
    )
    
    # Error is df is empty 
    if df is None or df.empty:
        raise RuntimeError("yfinance returned empty data (check tickers / connection).")

    # Expect MultiIndex columns with level 0 like: Open, High, Low, Close, Adj Close, Volume
    if not isinstance(df.columns, pd.MultiIndex):
        raise RuntimeError(f"Expected MultiIndex columns but got: {type(df.columns)}")

    # In yfinance DataFrame, each column has a stacked header (a MultiIndex), like a 2-layer header.
    # Level 0 = Open, High, Low, Close, Adj Close, Volume; Level 1 = tickers
    level0 = df.columns.get_level_values(0)
    print("Fields returned:", sorted(set(level0)))

    if "Adj Close" not in set(level0):
        # Stop here so it’s obvious what yfinance is returning on your machine
        raise KeyError("Adj Close not found. See printed 'Fields returned' above.")

    prices = df["Adj Close"]

    # Create the folder  
    os.makedirs(DATA_DIR_RAW, exist_ok=True)
    # Save as "parquet", a file format for data analytics
    prices.to_parquet(f"{DATA_DIR_RAW}/prices_raw.parquet")
    print(f"Saved: {DATA_DIR_RAW}/prices_raw.parquet")
    print(prices.head())

if __name__ == "__main__":
    main()