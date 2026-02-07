# Factor Momentum Backtest (12–1 Momentum, Long-Only)

## Overview
This project backtests a simple cross-sectional momentum strategy using daily price data. Each month-end, it ranks stocks by 12–1 momentum (past 12 months return excluding the most recent month), forms a long-only portfolio, and holds it until the next rebalance. The backtest includes turnover-based transaction costs and outputs daily returns and an equity curve.

## Strategy Definition

**Signal (12–1 momentum):** For each stock $i$ at rebalance date $t$,

$$
\mathrm{MOM}_{i,t}=\frac{P_{i,t-1m}}{P_{i,t-12m}}-1
$$

where $P_{i,t-1m}$ is the price one month before $t$ and $P_{i,t-12m}$ is the price twelve months before $t$.

**Portfolio:**
- Long-only
- Rebalanced at month-end
- Weights normalized to sum to 1 across selected stocks

## Backtest Methodology
- **No look-ahead:** weights are computed using information available up to the rebalance date, then applied starting from the next trading day.

**Daily portfolio return:**

$$
r_{p,t}=\sum_i w_{i,t}\,r_{i,t}
$$

- **Transaction costs:** modeled as a function of turnover at rebalances (details in the costs module).

## Repo Structure
- `Src/factor_momentum/` — core pipeline and strategy code
- `Data/Raw/` — raw pulled price data (not committed)
- `Data/Processed/` — cleaned returns/signal/weights (not committed)
- `Results/` — output artifacts (tables/plots)

## How to Run
From the repo root:
```bash
PYTHONPATH=Src python -m factor_momentum.pipeline
PYTHONPATH=Src python -m factor_momentum.backtest
