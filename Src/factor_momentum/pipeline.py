from pathlib import Path

from .config import RESULTS_DIR, COST_BPS_GRID
from .backtest import run_backtest

from .data_fetch import main as fetch_main
from .data_clean import main as clean_main
from .returns import main as returns_main
from .signals.momentum_12_1 import main as signal_main
from .portfolio import main as portfolio_main


def main() -> None:
    fetch_main()
    clean_main()
    returns_main()
    signal_main()
    portfolio_main()

    # 2) Robustness: run backtest under different trading costs
    for bps in COST_BPS_GRID:
        out_dir = Path(RESULTS_DIR) / f"cost_{bps}bps"
        run_backtest(cost_bps=bps, out_dir=out_dir)


if __name__ == "__main__":
    main()

