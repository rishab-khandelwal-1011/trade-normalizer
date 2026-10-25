"""
data_cleaner.py -- Raw Stock Data Cleaning
==========================================

Handles missing trade volumes, missing prices, and outlier detection
for all raw stock CSVs.  Produces cleaned DataFrames with an
``Outlier_Flag`` column and generates a per-stock cleaning report.
"""

import os
import numpy as np
import pandas as pd

from src import config
from src.utils import (
    get_logger,
    save_dataframe,
    load_dataframe,
    list_csv_files,
    get_ticker_from_filename,
    timer,
)

logger = get_logger(__name__)

# Columns treated as OHLC price columns
_OHLC_COLS = ["Open", "High", "Low", "Close", "Adj Close"]


# ==============================================================================
# Public API
# ==============================================================================


@timer
def clean_all_stocks() -> dict:
    """
    Clean every raw CSV and save the results to the cleaned-data directory.

    Iterates over all CSV files in ``config.RAW_DATA_DIR``, applies
    ``clean_single_stock`` to each, writes the cleaned DataFrame to
    ``config.CLEANED_DATA_DIR``, and finally generates a summary report.

    Returns
    -------
    dict
        Mapping of ticker symbol -> per-stock stats dict.  Each stats dict
        contains the keys returned by ``clean_single_stock``.
    """
    raw_files = list_csv_files(config.RAW_DATA_DIR)

    if not raw_files:
        logger.warning("No raw CSV files found in %s", config.RAW_DATA_DIR)
        return {}

    logger.info("Starting cleaning pipeline for %d stocks", len(raw_files))
    all_stats: dict = {}

    for filepath in raw_files:
        ticker = get_ticker_from_filename(filepath)
        try:
            cleaned_df, stats = clean_single_stock(filepath)

            # Persist the cleaned file
            out_path = os.path.join(config.CLEANED_DATA_DIR, f"{ticker}.csv")
            save_dataframe(cleaned_df, out_path)

            all_stats[ticker] = stats
            logger.info(
                "[OK] %s -- rows %d->%d | outliers flagged: %d",
                ticker,
                stats["original_rows"],
                stats["cleaned_rows"],
                stats["outliers_flagged"],
            )
        except Exception as exc:
            logger.error("[FAIL] %s -- cleaning failed: %s", ticker, exc, exc_info=True)

    # Generate the summary report
    if all_stats:
        generate_cleaning_report(all_stats)

    logger.info("Cleaning pipeline finished -- %d/%d stocks succeeded",
                len(all_stats), len(raw_files))
    return all_stats


def clean_single_stock(filepath: str) -> tuple[pd.DataFrame, dict]:
    """
    Load a raw stock CSV, clean it, and return the result with statistics.

    Cleaning steps (in order):

    1. Replace zero/sub-minimum volumes with NaN.
    2. Forward-fill -> backward-fill -> median-fill missing volumes.
    3. Linearly interpolate OHLC price gaps <=
       ``config.MAX_CONSECUTIVE_NAN_DAYS``; flag rows in larger gaps.
    4. Remove rows that belong to price gaps exceeding the threshold.
    5. Detect return-based outliers and add an ``Outlier_Flag`` column.

    Parameters
    ----------
    filepath : str
        Absolute path to a raw stock CSV file.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        cleaned_df : pd.DataFrame
            Cleaned DataFrame with an ``Outlier_Flag`` boolean column.
        stats : dict
            Dictionary with keys:

            - ``original_rows``       : int
            - ``cleaned_rows``        : int
            - ``missing_before_volume``: int
            - ``missing_after_volume`` : int
            - ``missing_before_close`` : int
            - ``missing_after_close``  : int
            - ``outliers_flagged``     : int
            - ``rows_removed``         : int
    """
    ticker = get_ticker_from_filename(filepath)
    df = load_dataframe(filepath)
    original_rows = len(df)

    logger.debug("%s -- loaded %d rows", ticker, original_rows)

    # ------------------------------------------------------------------
    # Capture "before" missing counts
    # ------------------------------------------------------------------
    missing_before_volume = int(
        df["Volume"].isna().sum() + (df["Volume"] <= config.MIN_VALID_VOLUME).sum()
    )
    missing_before_close = int(df["Close"].isna().sum())

    # ------------------------------------------------------------------
    # Step 1 -- Replace zero / sub-minimum volumes with NaN
    # ------------------------------------------------------------------
    df.loc[df["Volume"] <= config.MIN_VALID_VOLUME, "Volume"] = np.nan

    # ------------------------------------------------------------------
    # Step 2 -- Fill missing volumes: ffill -> bfill -> median
    # ------------------------------------------------------------------
    df["Volume"] = df["Volume"].ffill().bfill()
    vol_median = df["Volume"].median()
    df["Volume"] = df["Volume"].fillna(vol_median)

    missing_after_volume = int(df["Volume"].isna().sum())

    # ------------------------------------------------------------------
    # Step 3 -- Interpolate OHLC prices for small gaps; flag large gaps
    # ------------------------------------------------------------------
    rows_to_remove_mask = pd.Series(False, index=df.index)

    for col in _OHLC_COLS:
        if col not in df.columns:
            continue

        max_consec = _find_consecutive_nans(df[col])
        if max_consec == 0:
            continue

        if max_consec <= config.MAX_CONSECUTIVE_NAN_DAYS:
            # All gaps are small enough -- interpolate everything
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
        else:
            # Mixed gaps -- interpolate only the small ones, flag the rest
            gap_mask = _large_gap_mask(df[col], config.MAX_CONSECUTIVE_NAN_DAYS)

            # Interpolate the small-gap NaNs (set large-gap NaNs aside)
            temp = df[col].copy()
            temp[gap_mask] = np.nan  # protect large-gap positions
            temp = temp.interpolate(method="linear", limit_direction="both")

            # Write back only the interpolated (small-gap) values
            df[col] = temp
            # Mark large-gap rows for removal
            rows_to_remove_mask = rows_to_remove_mask | gap_mask

    # ------------------------------------------------------------------
    # Step 4 -- Remove rows where price gaps exceed the threshold
    # ------------------------------------------------------------------
    df = df[~rows_to_remove_mask].copy()
    rows_removed = int(rows_to_remove_mask.sum())

    missing_after_close = int(df["Close"].isna().sum()) if "Close" in df.columns else 0

    # ------------------------------------------------------------------
    # Step 5 -- Outlier detection via daily returns
    # ------------------------------------------------------------------
    df["Outlier_Flag"] = False

    if "Close" in df.columns and len(df) >= 2:
        daily_returns = df["Close"].pct_change()
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()

        if std_ret and not np.isnan(std_ret):
            outlier_mask = (
                (daily_returns - mean_ret).abs() > config.OUTLIER_STD_THRESHOLD * std_ret
            )
            df.loc[outlier_mask, "Outlier_Flag"] = True

    outliers_flagged = int(df["Outlier_Flag"].sum())

    # ------------------------------------------------------------------
    # Assemble stats
    # ------------------------------------------------------------------
    stats = {
        "original_rows": original_rows,
        "cleaned_rows": len(df),
        "missing_before_volume": missing_before_volume,
        "missing_after_volume": missing_after_volume,
        "missing_before_close": missing_before_close,
        "missing_after_close": missing_after_close,
        "outliers_flagged": outliers_flagged,
        "rows_removed": rows_removed,
    }

    logger.debug(
        "%s -- before missing (Volume=%d, Close=%d) -> after (Volume=%d, Close=%d) | "
        "outliers=%d, removed=%d",
        ticker,
        missing_before_volume,
        missing_before_close,
        missing_after_volume,
        missing_after_close,
        outliers_flagged,
        rows_removed,
    )

    return df, stats


def generate_cleaning_report(all_stats: dict) -> pd.DataFrame:
    """
    Produce a summary CSV from the per-stock cleaning statistics.

    The report is saved to ``config.REPORTS_DIR/cleaning_report.csv``.

    Parameters
    ----------
    all_stats : dict
        Mapping of ticker -> stats dict (as returned by ``clean_single_stock``).

    Returns
    -------
    pd.DataFrame
        The report DataFrame that was saved.
    """
    rows = []
    for ticker, stats in all_stats.items():
        rows.append(
            {
                "Ticker": ticker,
                "Original_Rows": stats["original_rows"],
                "Cleaned_Rows": stats["cleaned_rows"],
                "Missing_Before_Volume": stats["missing_before_volume"],
                "Missing_After_Volume": stats["missing_after_volume"],
                "Missing_Before_Close": stats["missing_before_close"],
                "Missing_After_Close": stats["missing_after_close"],
                "Outliers_Flagged": stats["outliers_flagged"],
            }
        )

    report_df = pd.DataFrame(rows)
    report_path = os.path.join(config.REPORTS_DIR, "cleaning_report.csv")
    save_dataframe(report_df, report_path, index=False)
    logger.info("Cleaning report saved -> %s (%d stocks)", report_path, len(report_df))

    return report_df


# ==============================================================================
# Private Helpers
# ==============================================================================


def _find_consecutive_nans(series: pd.Series) -> int:
    """
    Return the length of the longest consecutive-NaN run in *series*.

    Parameters
    ----------
    series : pd.Series
        Any pandas Series (typically a price or volume column).

    Returns
    -------
    int
        Maximum number of consecutive NaN values.  Returns 0 when there
        are no NaN values at all.
    """
    if series.isna().sum() == 0:
        return 0

    is_nan = series.isna().astype(int)
    # Group consecutive NaNs: whenever the value changes, a new group starts.
    groups = is_nan.ne(is_nan.shift()).cumsum()
    # Keep only NaN groups and find the largest
    nan_group_sizes = is_nan.groupby(groups).sum()
    return int(nan_group_sizes.max()) if nan_group_sizes.max() > 0 else 0


def _large_gap_mask(series: pd.Series, max_allowed: int) -> pd.Series:
    """
    Return a boolean mask that is ``True`` for NaN positions that belong
    to a consecutive-NaN run longer than *max_allowed*.

    Parameters
    ----------
    series : pd.Series
        Column to inspect.
    max_allowed : int
        Maximum gap length that is acceptable (will *not* be flagged).

    Returns
    -------
    pd.Series
        Boolean mask aligned with *series*.
    """
    is_nan = series.isna()
    if is_nan.sum() == 0:
        return pd.Series(False, index=series.index)

    is_nan_int = is_nan.astype(int)
    groups = is_nan_int.ne(is_nan_int.shift()).cumsum()

    # Size of each group
    group_sizes = is_nan_int.groupby(groups).transform("sum")

    # True only for NaN positions whose group exceeds the threshold
    return is_nan & (group_sizes > max_allowed)
