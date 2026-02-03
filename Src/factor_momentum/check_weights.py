import pandas as pd

w = pd.read_parquet("Data/Processed/weights_top10_eq.parquet")

print("shape:", w.shape)
print("date range:", w.index.min(), "->", w.index.max())

s = w.sum(axis=1)
nz = (w > 0).sum(axis=1)

print("\nLast 10 weight sums:")
print(s.tail(10))

print("\nLast 10 position counts:")
print(nz.tail(10))

# show one day
d = w.index[-1]
print("\nHoldings on last day:")
print(w.loc[d][w.loc[d] > 0].sort_values(ascending=False))
