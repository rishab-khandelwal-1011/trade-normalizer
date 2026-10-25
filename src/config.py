"""
config.py -- Central Configuration
==================================

All project-wide constants, file paths, and tunable parameters.
"""

import os
from datetime import datetime

# ==============================================================================
# Project Metadata
# ==============================================================================
PROJECT_ID = "ML_6"
PROJECT_NAME = "Stock Market Daily Trade Normalizer"

# ==============================================================================
# Directory Paths
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

TICKER_LIST_PATH = os.path.join(DATA_DIR, "nifty50_tickers.csv")

# ==============================================================================
# Data Download Settings
# ==============================================================================
# Date range for historical data (5 years)
START_DATE = "2021-10-01"
END_DATE = "2026-10-01"

# yfinance settings
DOWNLOAD_INTERVAL = "1d"        # Daily data
MAX_RETRIES = 3                 # Number of retry attempts per ticker
RETRY_DELAY_SECONDS = 2         # Delay between retries
BATCH_SIZE = 10                 # Number of tickers to download simultaneously

# ==============================================================================
# Data Cleaning Settings
# ==============================================================================
# Maximum consecutive NaN days before we flag (rather than interpolate)
MAX_CONSECUTIVE_NAN_DAYS = 5

# Outlier detection threshold (in standard deviations)
OUTLIER_STD_THRESHOLD = 3.0

# Minimum valid volume (below this is treated as missing)
MIN_VALID_VOLUME = 1

# ==============================================================================
# Feature Engineering Settings
# ==============================================================================
# Rolling window sizes (in days) for moving averages
SMA_WINDOWS = [7, 14, 21, 50, 200]
EMA_WINDOWS = [12, 26, 50]
WMA_WINDOWS = [7, 14, 21]

# Volatility window
VOLATILITY_WINDOW = 21

# ==============================================================================
# Normalization Settings
# ==============================================================================
# Methods available: 'minmax', 'zscore'
DEFAULT_NORM_METHOD = "minmax"
MINMAX_FEATURE_RANGE = (0, 1)

# ==============================================================================
# Visualization Settings
# ==============================================================================
# Default figure size
FIG_SIZE = (14, 8)
FIG_DPI = 150

# Chart style
CHART_STYLE = "seaborn-v0_8-darkgrid"

# Top N stocks to show in comparative charts (for readability)
TOP_N_STOCKS = 10

# Color palette
COLOR_PALETTE = "tab20"

# ==============================================================================
# Logging Settings
# ==============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = os.path.join(BASE_DIR, "pipeline.log")

# ==============================================================================
# Ensure all directories exist
# ==============================================================================
def ensure_directories():
    """Create all required directories if they don't exist."""
    dirs = [
        RAW_DATA_DIR,
        CLEANED_DATA_DIR,
        PROCESSED_DATA_DIR,
        CHARTS_DIR,
        REPORTS_DIR,
        NOTEBOOKS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
