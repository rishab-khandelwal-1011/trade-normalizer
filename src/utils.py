"""
utils.py -- Shared Utility Functions
=====================================

Helpers for logging, file I/O, timing, and data profiling.
"""

import os
import time
import logging
import functools
import pandas as pd
import numpy as np
from src import config


# ==============================================================================
# Logging Setup
# ==============================================================================
def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name : str
        Name of the logger (typically __name__ of the calling module).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, config.LOG_LEVEL))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
        console_formatter = logging.Formatter(
            config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(config.LOG_FILE, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

    return logger


# ==============================================================================
# Timing Decorator
# ==============================================================================
def timer(func):
    """
    Decorator that logs the execution time of a function.

    Parameters
    ----------
    func : callable
        The function to wrap.

    Returns
    -------
    callable
        Wrapped function that logs its runtime.
    """
    logger = get_logger("timer")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        minutes, seconds = divmod(elapsed, 60)
        if minutes > 0:
            logger.info(f"[timer] {func.__name__} completed in {int(minutes)}m {seconds:.2f}s")
        else:
            logger.info(f"[timer] {func.__name__} completed in {seconds:.2f}s")
        return result

    return wrapper


# ==============================================================================
# File I/O Helpers
# ==============================================================================
def save_dataframe(df: pd.DataFrame, filepath: str, index: bool = True) -> None:
    """
    Save a DataFrame to CSV with directory auto-creation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filepath : str
        Full path to the output CSV file.
    index : bool, optional
        Whether to write the index column, by default True.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=index)


def load_dataframe(filepath: str, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load a DataFrame from CSV.

    Parameters
    ----------
    filepath : str
        Full path to the CSV file.
    parse_dates : bool, optional
        Whether to parse the first column as dates, by default True.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if parse_dates:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        df = pd.read_csv(filepath, index_col=0)

    return df


def list_csv_files(directory: str) -> list:
    """
    List all CSV files in a directory.

    Parameters
    ----------
    directory : str
        Path to the directory.

    Returns
    -------
    list
        Sorted list of full file paths to CSV files.
    """
    if not os.path.exists(directory):
        return []
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".csv")
    ]
    return sorted(files)


def get_ticker_from_filename(filepath: str) -> str:
    """
    Extract ticker symbol from a CSV filename.

    Parameters
    ----------
    filepath : str
        Path like 'data/raw/TCS.NS.csv'

    Returns
    -------
    str
        Ticker symbol like 'TCS.NS'
    """
    basename = os.path.basename(filepath)
    return basename.replace(".csv", "")


# ==============================================================================
# Data Profiling
# ==============================================================================
def profile_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> dict:
    """
    Generate a quick statistical profile of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to profile.
    name : str, optional
        Label for the profile, by default "DataFrame".

    Returns
    -------
    dict
        Dictionary with profiling statistics.
    """
    profile = {
        "name": name,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "numeric_summary": {},
    }

    for col in df.select_dtypes(include=[np.number]).columns:
        profile["numeric_summary"][col] = {
            "mean": round(df[col].mean(), 4),
            "std": round(df[col].std(), 4),
            "min": round(df[col].min(), 4),
            "max": round(df[col].max(), 4),
        }

    return profile


def print_profile(profile: dict) -> None:
    """
    Pretty-print a data profile dictionary.

    Parameters
    ----------
    profile : dict
        Profile dict from profile_dataframe().
    """
    logger = get_logger("profiler")
    logger.info(f"--- Profile: {profile['name']} ---")
    logger.info(f"  Shape: {profile['shape']}")
    logger.info(f"  Columns: {profile['columns']}")

    if any(v > 0 for v in profile["missing_counts"].values()):
        logger.info("  Missing values:")
        for col, count in profile["missing_counts"].items():
            if count > 0:
                pct = profile["missing_pct"][col]
                logger.info(f"    {col}: {count} ({pct}%)")
    else:
        logger.info("  No missing values [OK]")
