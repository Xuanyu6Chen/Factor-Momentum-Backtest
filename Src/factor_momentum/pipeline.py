from .data_fetch import main as fetch_main
from .data_clean import main as clean_main
from .returns import main as returns_main
from .signals.momentum_12_1 import main as mom_main
from .portfolio import main as portfolio_main

def main():
    fetch_main()
    clean_main()
    returns_main()
    mom_main()
    portfolio_main()
    print("Pipeline complete ✅")

if __name__ == "__main__":
    main()
