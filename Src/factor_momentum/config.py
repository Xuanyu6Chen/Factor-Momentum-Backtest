# List of the Stock symbols
TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "BRK-B",
    "WMT", "LLY", "JPM", "V", "XOM", "JNJ", "ORCL", "MA", "MU", "COST",
    "AMD", "ABBV", "PLTR", "HD", "BAC", "NFLX", "PG", "CVX", "KO", "GE",
    "CSCO", "LRCX", "CAT", "MS", "GS", "PM", "IBM", "WFC", "RTX", "MRK",
    "AMAT", "UNH", "AXP", "TMO", "MCD", "INTC", "CRM", "KLAC", "LIN", "TMUS"
]
# The earliest date to fetch
START_DATE = "2016-01-01"
END_DATE = None

#path to store the raw data 
DATA_DIR_RAW = "Data/Raw"

#path to store the processed data 
DATA_DIR_PROCESSED = "Data/Processed"

from pathlib import Path

# ========= Project root =========
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ========= Folders (Path objects) =========
DATA_DIR = PROJECT_ROOT / "Data"
DATA_DIR_RAW = DATA_DIR / "Raw"
DATA_DIR_PROCESSED = DATA_DIR / "Processed"
RESULTS_DIR = PROJECT_ROOT / "Results"

# ========= Common file paths =========
WEIGHTS_PATH = DATA_DIR_PROCESSED / "weights_top10_eq.parquet"
RETURNS_PATH = DATA_DIR_PROCESSED / "returns.parquet"

# ========= Robustness grid =========
COST_BPS_GRID = [0, 10, 50]

# ========= Output filenames (inside a run folder) =========
STRATEGY_RETURNS_FILE = "strategy_returns.parquet"
EQUITY_CURVE_FILE = "equity_curve.parquet"
META_FILE = "run_meta.txt"
