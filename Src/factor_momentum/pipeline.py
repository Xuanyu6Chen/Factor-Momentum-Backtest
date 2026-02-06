# Src/factor_momentum/pipeline.py
from pathlib import Path

from .config import RESULTS_DIR, COST_BPS_GRID
from .backtest import run_backtest

# If your pipeline already runs fetch/clean/returns/signal/portfolio, keep that.
# Example:
# from .data_fetch import main as fetch_main
# from .data_clean import main as clean_main
# from .returns import main as returns_main
# from .signal import main as signal_main
# from .portfolio import main as portfolio_main


def main() -> None:
    # 1) (Optional) Run upstream pipeline parts if you want end-to-end.
    # Comment/uncomment depending on your existing structure.
    #
    # fetch_main()
    # clean_main()
    # returns_main()
    # signal_main()
    # portfolio_main()

    # 2) Robustness: run backtest under different trading costs
    for bps in COST_BPS_GRID:
        out_dir = Path(RESULTS_DIR) / f"cost_{bps}bps"
        run_backtest(cost_bps=bps, out_dir=out_dir)


if __name__ == "__main__":
    main()


