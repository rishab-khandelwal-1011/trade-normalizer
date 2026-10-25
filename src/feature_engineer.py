"""
feature_engineer.py -- Rolling-Window Feature Engineering (Pure NumPy)
=====================================================================

Engineers moving averages, volatility, and return features for cleaned
stock data using **pure NumPy** operations (no pandas ``.rolling()``).

Key Functions
-------------
- ``compute_sma``  : Simple Moving Average via ``np.convolve``
- ``compute_ema``  : Exponential Moving Average via iterative formula
- ``compute_wma``  : Weighted Moving Average with linear weights
- ``compute_rolling_std`` : Rolling standard deviation
- ``compute_daily_returns`` : Day-over-day percentage change
- ``compute_cumulative_returns`` : Running cumulative product of (1 + r)
- ``engineer_single_stock`` : Apply all features to one ticker CSV
- ``engineer_all_stocks``   : Batch-process every cleaned CSV
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

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


# =============================================================================
# Pure-NumPy Rolling Computations
# =============================================================================

def compute_sma(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the Simple Moving Average using NumPy's ``np.convolve``.

    The SMA at index *i* is the unweighted mean of the previous *window*
    values (inclusive).  ``np.convolve`` with a uniform kernel of length
    *window* produces the "valid" portion of the result; the first
    ``window - 1`` entries are set to ``NaN`` to keep the output aligned
    with the input.

    Parameters
    ----------
    data : np.ndarray
        1-D array of numeric values (e.g. closing prices).
    window : int
        Number of periods in the moving-average window.

    Returns
    -------
    np.ndarray
        Array of the same length as *data* with leading NaNs.

    Notes
    -----
    Implementation:  ``np.convolve(data, np.ones(window)/window, 'valid')``
    gives a shorter array.  We prepend ``window - 1`` NaN values so the
    result lines up with the original time axis.
    """
    kernel = np.ones(window) / window
    convolved = np.convolve(data, kernel, mode="valid")
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, convolved])


def compute_ema(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the Exponential Moving Average via iterative smoothing.

    Uses the standard EMA recurrence:

    .. math::

        \\text{EMA}_t = \\alpha \\cdot x_t + (1 - \\alpha) \\cdot \\text{EMA}_{t-1}

    where :math:`\\alpha = 2 / (\\text{window} + 1)`.  The first value is
    seeded with ``data[0]``.

    Parameters
    ----------
    data : np.ndarray
        1-D array of numeric values.
    window : int
        Span (number of periods) that determines the decay factor.

    Returns
    -------
    np.ndarray
        EMA values with the same length as *data*.

    Notes
    -----
    This is a pure-Python/NumPy loop rather than pandas ``.ewm()``.
    The loop is O(n) and operates on a pre-allocated NumPy array for
    speed.
    """
    alpha = 2.0 / (window + 1)
    ema = np.empty_like(data, dtype=np.float64)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def compute_wma(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the Weighted Moving Average with linearly increasing weights.

    Weights are ``[1, 2, ..., window]``, so more-recent observations
    receive higher weight.  The denominator is ``window * (window + 1) / 2``
    (the sum of the weight vector).

    The computation iterates over valid positions and computes the
    weighted dot product of each window slice with the weight vector.

    Parameters
    ----------
    data : np.ndarray
        1-D array of numeric values.
    window : int
        Number of periods in the window.

    Returns
    -------
    np.ndarray
        Array of same length as *data* with ``window - 1`` leading NaNs.

    Notes
    -----
    Unlike SMA, ``np.convolve`` is not directly suitable here because the
    weights are *not* symmetric (the kernel would need to be reversed).
    A manual loop with ``np.dot`` on slices keeps the intent clear and
    avoids off-by-one issues.
    """
    weights = np.arange(1, window + 1, dtype=np.float64)
    weight_sum = weights.sum()

    n = len(data)
    wma = np.full(n, np.nan)
    for i in range(window - 1, n):
        wma[i] = np.dot(data[i - window + 1 : i + 1], weights) / weight_sum
    return wma


def compute_rolling_std(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the rolling (sample) standard deviation using NumPy.

    Uses the cumulative-sum trick for efficiency:

    1. Compute the rolling sum and rolling sum-of-squares via
       ``np.cumsum``.
    2. Derive the variance as
       ``(sum_x2 - sum_x**2 / window) / (window - 1)`` (Bessel's
       correction).
    3. Take the square root.

    The first ``window - 1`` positions are ``NaN``.

    Parameters
    ----------
    data : np.ndarray
        1-D array of numeric values.
    window : int
        Number of periods in the rolling window.

    Returns
    -------
    np.ndarray
        Rolling standard deviation, same length as *data*.

    Notes
    -----
    This avoids pandas ``.rolling().std()`` by relying on cumulative
    sums, which is O(n) and numerically adequate for typical price data.
    """
    n = len(data)
    result = np.full(n, np.nan)

    cumsum = np.cumsum(data)
    cumsum2 = np.cumsum(data ** 2)

    # First full-window value
    sum_x = cumsum[window - 1]
    sum_x2 = cumsum2[window - 1]
    var = (sum_x2 - sum_x ** 2 / window) / (window - 1)
    result[window - 1] = np.sqrt(max(var, 0.0))

    # Remaining values via difference of cumulative sums
    for i in range(window, n):
        sum_x = cumsum[i] - cumsum[i - window]
        sum_x2 = cumsum2[i] - cumsum2[i - window]
        var = (sum_x2 - sum_x ** 2 / window) / (window - 1)
        result[i] = np.sqrt(max(var, 0.0))

    return result


def compute_daily_returns(data: np.ndarray) -> np.ndarray:
    """
    Compute day-over-day percentage returns.

    .. math::

        r_t = \\frac{x_t - x_{t-1}}{x_{t-1}}

    The first element is ``NaN`` (no prior day).

    Parameters
    ----------
    data : np.ndarray
        1-D array of prices.

    Returns
    -------
    np.ndarray
        Percentage returns with a leading NaN.
    """
    returns = np.empty_like(data, dtype=np.float64)
    returns[0] = np.nan
    returns[1:] = (data[1:] - data[:-1]) / data[:-1]
    return returns


def compute_cumulative_returns(data: np.ndarray) -> np.ndarray:
    """
    Compute the running cumulative return from daily returns.

    Given daily returns *r*, the cumulative return at time *t* is:

    .. math::

        C_t = \\prod_{i=1}^{t} (1 + r_i)

    The first element is ``NaN`` (matches the daily-return series).

    Parameters
    ----------
    data : np.ndarray
        1-D array of daily fractional returns (output of
        ``compute_daily_returns``).

    Returns
    -------
    np.ndarray
        Cumulative return series, same length as *data*.
    """
    cum = np.empty_like(data, dtype=np.float64)
    cum[0] = np.nan
    cum[1:] = np.cumprod(1 + data[1:])
    return cum


# =============================================================================
# Single-Stock Feature Engineering
# =============================================================================

@timer
def engineer_single_stock(filepath: str) -> pd.DataFrame:
    """
    Load a cleaned CSV and attach all engineered features.

    Features added
    ~~~~~~~~~~~~~~
    * **SMA_{w}** -- Simple Moving Average of Close for each *w* in
      ``config.SMA_WINDOWS``.
    * **EMA_{w}** -- Exponential Moving Average of Close for each *w* in
      ``config.EMA_WINDOWS``.
    * **WMA_{w}** -- Weighted Moving Average of Close for each *w* in
      ``config.WMA_WINDOWS``.
    * **Volatility_{v}** -- Rolling standard deviation of Close with
      window ``config.VOLATILITY_WINDOW``.
    * **Vol_SMA_{w}** -- Simple Moving Average of Volume for each *w* in
      ``config.SMA_WINDOWS``.
    * **Daily_Return** -- Day-over-day percentage change of Close.
    * **Cumulative_Return** -- Running cumulative product of (1 + daily
      return).

    Parameters
    ----------
    filepath : str
        Full path to a cleaned CSV file inside ``config.CLEANED_DATA_DIR``.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame saved to ``config.PROCESSED_DATA_DIR``.
    """
    ticker = get_ticker_from_filename(filepath)
    logger.info(f"Engineering features for {ticker}")

    df = load_dataframe(filepath)
    close = df["Close"].values.astype(np.float64)
    volume = df["Volume"].values.astype(np.float64)

    # --- Simple Moving Averages (Close) ---
    for w in config.SMA_WINDOWS:
        df[f"SMA_{w}"] = compute_sma(close, w)

    # --- Exponential Moving Averages (Close) ---
    for w in config.EMA_WINDOWS:
        df[f"EMA_{w}"] = compute_ema(close, w)

    # --- Weighted Moving Averages (Close) ---
    for w in config.WMA_WINDOWS:
        df[f"WMA_{w}"] = compute_wma(close, w)

    # --- Rolling Volatility (Close) ---
    v = config.VOLATILITY_WINDOW
    df[f"Volatility_{v}"] = compute_rolling_std(close, v)

    # --- Volume Moving Averages ---
    for w in config.SMA_WINDOWS:
        df[f"Vol_SMA_{w}"] = compute_sma(volume, w)

    # --- Daily & Cumulative Returns ---
    daily_ret = compute_daily_returns(close)
    df["Daily_Return"] = daily_ret
    df["Cumulative_Return"] = compute_cumulative_returns(daily_ret)

    # --- Save ---
    out_path = os.path.join(config.PROCESSED_DATA_DIR, os.path.basename(filepath))
    save_dataframe(df, out_path)
    logger.info(f"Saved processed data for {ticker} -> {out_path}")

    return df


# =============================================================================
# Batch Processing
# =============================================================================

@timer
def engineer_all_stocks() -> dict[str, pd.DataFrame]:
    """
    Apply feature engineering to every cleaned stock CSV.

    Iterates over all CSV files in ``config.CLEANED_DATA_DIR``, calls
    :func:`engineer_single_stock` on each, and collects the results.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker symbol -> enriched DataFrame.
    """
    files = list_csv_files(config.CLEANED_DATA_DIR)
    if not files:
        logger.warning("No cleaned CSV files found -- nothing to engineer.")
        return {}

    logger.info(f"Engineering features for {len(files)} stock(s)")
    results: dict[str, pd.DataFrame] = {}

    for fp in tqdm(files, desc="Feature engineering", unit="stock"):
        ticker = get_ticker_from_filename(fp)
        try:
            results[ticker] = engineer_single_stock(fp)
        except Exception as exc:
            logger.error(f"Failed to engineer {ticker}: {exc}")

    logger.info(f"Feature engineering complete -- {len(results)}/{len(files)} succeeded")
    return results


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    config.ensure_directories()
    engineer_all_stocks()
