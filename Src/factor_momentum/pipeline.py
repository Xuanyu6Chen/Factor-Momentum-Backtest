# Src/factor_momentum/pipeline.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .data_fetch import main as fetch_main
from .data_clean import main as clean_main
from .returns import main as returns_main
from .signals.momentum_12_1 import main as mom_main
from .portfolio import main as portfolio_main
from .backtest import main as backtest_main
from .evaluate import export_report


RESULTS_DIR = Path("Results")
STRATEGY_RETURNS_FILE = RESULTS_DIR / "strategy_returns.parquet"

# Choose which return series to evaluate/report:
# - "net_ret" is after costs (recommended for real performance)
# - "gross_ret" is before costs
EVAL_RET_COL = "net_ret"


def main() -> None:
    # 1) Data pipeline
    fetch_main()
    clean_main()
    returns_main()

    # 2) Signal + portfolio construction
    mom_main()
    portfolio_main()

    # 3) Backtest
    backtest_main()

    # 4) Evaluation + reporting
    if not STRATEGY_RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Expected backtest output not found: {STRATEGY_RETURNS_FILE}\n"
            "Check backtest.py output paths (Results vs results) and file name."
        )

    df = pd.read_parquet(STRATEGY_RETURNS_FILE)

    if EVAL_RET_COL not in df.columns:
        raise ValueError(
            f"{STRATEGY_RETURNS_FILE} has columns {list(df.columns)}.\n"
            f"Expected to find '{EVAL_RET_COL}'. Change EVAL_RET_COL to one of the available columns."
        )

    strategy_returns = df[EVAL_RET_COL].dropna().sort_index().astype(float)

    metrics = export_report(
        returns=strategy_returns,
        results_dir=RESULTS_DIR,
        rf_annual=0.0,
    )

    print("Evaluation complete ✅")
    print(f"Using return column: {EVAL_RET_COL}")
    print(f"Saved: {RESULTS_DIR / 'metrics.csv'}")
    print(f"Saved: {RESULTS_DIR / 'figures' / 'equity_curve.png'}")
    print(f"Saved: {RESULTS_DIR / 'figures' / 'drawdown.png'}")
    print("Key metrics:", metrics)

    print("Pipeline complete ✅")


if __name__ == "__main__":
    main()

