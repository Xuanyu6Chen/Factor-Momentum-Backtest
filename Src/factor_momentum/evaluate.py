from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def infer_ann_factor(dt_index: pd.DatetimeIndex) -> int:
    """
    Best-effort inference of annualization factor.
    If you're using monthly rebalances (most likely in this project), it should be 12.
    """
    if len(dt_index) < 3:
        return 12  # safe default for this momentum project

    # median spacing in days
    median_days = np.median(np.diff(dt_index.values).astype("timedelta64[D]").astype(int))

    # Rough cutoffs
    if median_days >= 20:   # ~monthly
        return 12
    if median_days <= 3:    # ~daily (business days)
        return 252
    # If it's weekly-ish
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
    ann_factor: int | None = None
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
        "periods": n,
        "ann_factor": ann_factor,
        "total_return": float(total_return),
        "CAGR": float(cagr),
        "annual_vol": float(vol_ann),
        "Sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def plot_equity(equity: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(equity.index, equity.values)
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_drawdown(drawdown: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(drawdown.index, drawdown.values)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def export_report(
    returns: pd.Series,
    results_dir: str | Path = "results",
    rf_annual: float = 0.0,
) -> dict:
    results_dir = Path(results_dir)
    fig_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(returns=returns, rf_annual=rf_annual)

    # Save metrics.csv (single row)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)

    # Equity + drawdown plots
    equity = returns_to_equity(returns, start_value=1.0)
    dd = compute_drawdown(equity)

    plot_equity(equity, fig_dir / "equity_curve.png")
    plot_drawdown(dd, fig_dir / "drawdown.png")

    return metrics
