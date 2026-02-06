# Src/factor_momentum/evaluate.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import RESULTS_DIR, COST_BPS_GRID


def infer_ann_factor(dt_index: pd.DatetimeIndex) -> int:
    """
    Best-effort inference of annualization factor.
    If you're using daily returns (most likely in this backtest output), it should be ~252.
    """
    if len(dt_index) < 3:
        return 252  # safer default for daily series

    median_days = np.median(np.diff(dt_index.values).astype("timedelta64[D]").astype(int))
    if median_days >= 20:
        return 12
    if median_days <= 3:
        return 252
    return 52


def returns_to_equity(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    r = returns.dropna().astype(float)
    equity = (1.0 + r).cumprod() * start_value
    equity.name = "equity"
    return equity


def compute_drawdown(equity: pd.Series) -> pd.Series:
    eq = equity.dropna().astype(float)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    dd.name = "drawdown"
    return dd


def compute_metrics(
    returns: pd.Series,
    rf_annual: float = 0.0,
    ann_factor: int | None = None,
) -> dict:
    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("returns series is empty after dropping NaNs")

    if ann_factor is None:
        ann_factor = infer_ann_factor(pd.DatetimeIndex(r.index))

    equity = returns_to_equity(r, start_value=1.0)
    dd = compute_drawdown(equity)

    n = len(r)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0

    # CAGR
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (ann_factor / n) - 1.0

    # Vol & Sharpe
    vol_ann = r.std(ddof=1) * np.sqrt(ann_factor)

    rf_per_period = rf_annual / ann_factor
    excess = r - rf_per_period
    denom = excess.std(ddof=1)

    sharpe = np.nan
    if denom > 0:
        sharpe = (excess.mean() * ann_factor) / (denom * np.sqrt(ann_factor))

    max_dd = dd.min()

    return {
        "periods": int(n),
        "ann_factor": int(ann_factor),
        "total_return": float(total_return),
        "CAGR": float(cagr),
        "annual_vol": float(vol_ann),
        "Sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def _load_strategy_returns(run_dir: Path) -> pd.Series:
    """
    Expects: run_dir/strategy_returns.parquet with column 'strategy_return' OR a single column.
    """
    p = run_dir / "strategy_returns.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

    df = pd.read_parquet(p)

    if isinstance(df, pd.DataFrame):
        if "strategy_return" in df.columns:
            s = df["strategy_return"]
        else:
            # fallback: first column
            s = df.iloc[:, 0]
    else:
        s = df

    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s.name = "strategy_return"
    return s


def plot_equity_cost_grid(equity_map: dict[int, pd.Series], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()

    for bps, eq in equity_map.items():
        plt.plot(eq.index, eq.values, label=f"{bps} bps")

    plt.title("Equity Curve vs Trading Costs")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_drawdown_cost_grid(dd_map: dict[int, pd.Series], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()

    for bps, dd in dd_map.items():
        plt.plot(dd.index, dd.values, label=f"{bps} bps")

    plt.title("Drawdown vs Trading Costs")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def export_cost_grid_report(
    results_dir: Path = RESULTS_DIR,
    cost_bps_grid: list[int] = COST_BPS_GRID,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    equity_map: dict[int, pd.Series] = {}
    dd_map: dict[int, pd.Series] = {}

    for bps in cost_bps_grid:
        run_dir = results_dir / f"cost_{bps}bps"
        r = _load_strategy_returns(run_dir)

        m = compute_metrics(r, rf_annual=rf_annual)
        m["cost_bps"] = bps
        rows.append(m)

        eq = returns_to_equity(r, start_value=1.0)
        equity_map[bps] = eq
        dd_map[bps] = compute_drawdown(eq)

    metrics_df = pd.DataFrame(rows).sort_values("cost_bps")
    metrics_df.to_csv(results_dir / "cost_grid_metrics.csv", index=False)

    plot_equity_cost_grid(equity_map, fig_dir / "equity_curve_cost_grid.png")
    plot_drawdown_cost_grid(dd_map, fig_dir / "drawdown_cost_grid.png")

    print(f"Saved: {results_dir / 'cost_grid_metrics.csv'}")
    print(f"Saved: {fig_dir / 'equity_curve_cost_grid.png'}")
    print(f"Saved: {fig_dir / 'drawdown_cost_grid.png'}")

    return metrics_df


def main() -> None:
    export_cost_grid_report(results_dir=RESULTS_DIR, cost_bps_grid=COST_BPS_GRID, rf_annual=0.0)


if __name__ == "__main__":
    main()
