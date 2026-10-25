"""
Stock Market Daily Trade Normalizer (ML_6)
==========================================

A data preprocessing pipeline for multi-stock historical datasets.

Modules:
    - config: Central configuration and constants
    - data_downloader: Download stock data via yfinance
    - data_cleaner: Clean missing values and handle outliers
    - feature_engineer: Rolling window moving averages with NumPy
    - normalizer: Min-Max and Z-Score normalization
    - visualizer: Comparative stock charts and plots
    - utils: Shared utility functions
"""

__version__ = "1.0.0"
__project_id__ = "ML_6"
__project_name__ = "Stock Market Daily Trade Normalizer"
