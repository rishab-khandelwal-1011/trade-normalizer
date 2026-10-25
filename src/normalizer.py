"""
normalizer.py -- Stock Data Normalization
==========================================

Provides min-max and z-score normalization (and their inverses) for
preparing stock data for cross-stock comparison.  Includes convenience
functions that normalise every processed CSV in bulk or build a single
comparative DataFrame across all tickers.
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


# ==============================================================================
# Core Normalization Functions
# ==============================================================================
def minmax_normalize(
    data: np.ndarray,
    feature_range: tuple[float, float] = (0, 1),
) -> tuple[np.ndarray, dict]:
    """
    Min-max scale an array to a target feature range.

    Applies the formula::

        X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min

    Parameters
    ----------
    data : np.ndarray
        Input array of numeric values.
    feature_range : tuple of float, optional
        Desired (min, max) of the transformed data.  Default is ``(0, 1)``.

    Returns
    -------
    normalized : np.ndarray
        Scaled array with values in *feature_range*.
    params : dict
        Scaling parameters needed for inversion:
        ``{'min': float, 'max': float, 'data_min': float, 'data_max': float}``.

    Raises
    ------
    ValueError
        If *data* is empty or constant (data_max == data_min).
    """
    data = np.asarray(data, dtype=np.float64)

    if data.size == 0:
        raise ValueError("Cannot normalise an empty array.")

    data_min = float(np.nanmin(data))
    data_max = float(np.nanmax(data))

    if data_max == data_min:
        raise ValueError(
            "Cannot min-max normalise constant data (data_max == data_min)."
        )

    feat_min, feat_max = feature_range
    scale = (data - data_min) / (data_max - data_min)
    normalized = scale * (feat_max - feat_min) + feat_min

    params = {
        "min": feat_min,
        "max": feat_max,
        "data_min": data_min,
        "data_max": data_max,
    }
    return normalized, params


def zscore_normalize(data: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Z-score (standard-score) normalise an array.

    Applies the formula::

        X_scaled = (X - mean) / std

    Parameters
    ----------
    data : np.ndarray
        Input array of numeric values.

    Returns
    -------
    normalized : np.ndarray
        Array with zero mean and unit variance.
    params : dict
        Scaling parameters needed for inversion:
        ``{'mean': float, 'std': float}``.

    Raises
    ------
    ValueError
        If *data* is empty or has zero standard deviation.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.size == 0:
        raise ValueError("Cannot normalise an empty array.")

    mean = float(np.nanmean(data))
    std = float(np.nanstd(data, ddof=0))

    if std == 0:
        raise ValueError(
            "Cannot z-score normalise data with zero standard deviation."
        )

    normalized = (data - mean) / std

    params = {"mean": mean, "std": std}
    return normalized, params


# ==============================================================================
# Inverse Transforms
# ==============================================================================
def inverse_minmax(data: np.ndarray, params: dict) -> np.ndarray:
    """
    Reverse a min-max scaling transformation.

    Parameters
    ----------
    data : np.ndarray
        Previously scaled array.
    params : dict
        Parameters returned by :func:`minmax_normalize`.

    Returns
    -------
    np.ndarray
        Data restored to the original scale.
    """
    data = np.asarray(data, dtype=np.float64)
    feat_min = params["min"]
    feat_max = params["max"]
    data_min = params["data_min"]
    data_max = params["data_max"]

    # Reverse: scale back to [0, 1] then to original range
    unscaled = (data - feat_min) / (feat_max - feat_min)
    original = unscaled * (data_max - data_min) + data_min
    return original


def inverse_zscore(data: np.ndarray, params: dict) -> np.ndarray:
    """
    Reverse a z-score normalisation.

    Parameters
    ----------
    data : np.ndarray
        Previously scaled array.
    params : dict
        Parameters returned by :func:`zscore_normalize`.

    Returns
    -------
    np.ndarray
        Data restored to the original scale.
    """
    data = np.asarray(data, dtype=np.float64)
    return data * params["std"] + params["mean"]


# ==============================================================================
# Batch Normalization Helpers
# ==============================================================================
@timer
def normalize_all_stocks(method: str = "minmax") -> None:
    """
    Normalise *Adj Close* and *Volume* for every processed stock CSV.

    For each file in ``config.PROCESSED_DATA_DIR`` the function:

    1. Loads the CSV.
    2. Normalises the *Adj Close* and *Volume* columns using the chosen
       *method*.
    3. Appends ``Norm_AdjClose`` and ``Norm_Volume`` columns (existing
       columns are **not** overwritten).
    4. Saves the updated CSV back to ``config.PROCESSED_DATA_DIR``.
    5. Collects all per-ticker scaling parameters and writes them to
       ``config.REPORTS_DIR/normalization_params.csv``.

    Parameters
    ----------
    method : str, optional
        Normalization method -- ``'minmax'`` (default) or ``'zscore'``.

    Raises
    ------
    ValueError
        If *method* is not one of ``'minmax'`` or ``'zscore'``.
    """
    if method not in ("minmax", "zscore"):
        raise ValueError(f"Unknown normalization method: '{method}'. Use 'minmax' or 'zscore'.")

    csv_files = list_csv_files(config.PROCESSED_DATA_DIR)
    if not csv_files:
        logger.warning("No processed CSV files found -- nothing to normalise.")
        return

    normalize_fn = minmax_normalize if method == "minmax" else zscore_normalize

    all_params: list[dict] = []

    for filepath in csv_files:
        ticker = get_ticker_from_filename(filepath)
        logger.info(f"Normalising {ticker} using {method} ...")

        df = load_dataframe(filepath)

        for col, norm_col in [("Adj Close", "Norm_AdjClose"), ("Volume", "Norm_Volume")]:
            if col not in df.columns:
                logger.warning(f"  Column '{col}' not found in {ticker} -- skipping.")
                continue

            raw_values = df[col].values

            try:
                if method == "minmax":
                    normed, params = normalize_fn(raw_values, feature_range=config.MINMAX_FEATURE_RANGE)
                else:
                    normed, params = normalize_fn(raw_values)
            except ValueError as exc:
                logger.warning(f"  Could not normalise '{col}' for {ticker}: {exc}")
                continue

            df[norm_col] = normed

            # Record parameters
            param_record = {"Ticker": ticker, "Column": col, "Method": method}
            param_record.update(params)
            all_params.append(param_record)

        # Save back (preserves existing columns)
        save_dataframe(df, filepath)
        logger.info(f"  Saved normalised data for {ticker}.")

    # Persist scaling parameters
    if all_params:
        params_df = pd.DataFrame(all_params)
        params_path = os.path.join(config.REPORTS_DIR, "normalization_params.csv")
        save_dataframe(params_df, params_path, index=False)
        logger.info(f"Normalization parameters saved to {params_path}")

    logger.info(
        f"Batch normalisation complete -- {len(csv_files)} stocks processed "
        f"using '{method}'."
    )


@timer
def normalize_for_comparison(column: str = "Adj Close") -> pd.DataFrame:
    """
    Build a single min-max normalised DataFrame for cross-stock comparison.

    All processed stock CSVs are loaded, and the specified *column* from
    each is placed into a combined DataFrame (dates x tickers).  The
    entire combined frame is then normalised **across all stocks** so that
    relative magnitudes are preserved, making it suitable for overlay
    charts.

    Parameters
    ----------
    column : str, optional
        The column to extract and normalise.  Default is ``'Adj Close'``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``DatetimeIndex`` rows and ticker-symbol columns,
        values min-max normalised across the full matrix.

    Raises
    ------
    RuntimeError
        If no processed CSV files are found.
    """
    csv_files = list_csv_files(config.PROCESSED_DATA_DIR)
    if not csv_files:
        raise RuntimeError("No processed CSV files found in PROCESSED_DATA_DIR.")

    combined: dict[str, pd.Series] = {}

    for filepath in csv_files:
        ticker = get_ticker_from_filename(filepath)
        df = load_dataframe(filepath)

        if column not in df.columns:
            logger.warning(f"Column '{column}' missing in {ticker} -- skipped.")
            continue

        combined[ticker] = df[column]

    combined_df = pd.DataFrame(combined)
    combined_df.sort_index(inplace=True)

    logger.info(
        f"Combined DataFrame shape: {combined_df.shape} "
        f"({len(combined)} tickers x {len(combined_df)} dates)"
    )

    # Flatten, normalise, reshape
    values = combined_df.values  # may contain NaN at non-overlapping dates
    flat = values.flatten()

    # Use only non-NaN values to compute scaling parameters
    valid_mask = ~np.isnan(flat)
    data_min = float(np.nanmin(flat))
    data_max = float(np.nanmax(flat))

    if data_max == data_min:
        logger.warning("All values are identical -- returning zeros.")
        normalized_df = pd.DataFrame(
            np.zeros_like(values), index=combined_df.index, columns=combined_df.columns
        )
        return normalized_df

    normalized_values = (values - data_min) / (data_max - data_min)
    normalized_df = pd.DataFrame(
        normalized_values, index=combined_df.index, columns=combined_df.columns
    )

    logger.info(f"Cross-stock normalisation on '{column}' complete.")
    return normalized_df
