"""
data_downloader.py -- Historical Stock Data Downloader
======================================================

Downloads daily OHLCV data for all Nifty 50 tickers using the
``yfinance`` library and persists each ticker as a separate CSV
in the raw-data directory.
"""

import os
import time

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src import config
from src.utils import get_logger, save_dataframe, load_dataframe, timer

logger = get_logger(__name__)


# ==============================================================================
# Helper -- Load Ticker List
# ==============================================================================
def _load_ticker_list() -> pd.DataFrame:
    """
    Load the Nifty 50 ticker list from the configured CSV path.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Symbol``, ``Company``, ``Sector``.

    Raises
    ------
    FileNotFoundError
        If the ticker-list CSV does not exist.
    """
    if not os.path.exists(config.TICKER_LIST_PATH):
        raise FileNotFoundError(
            f"Ticker list not found at {config.TICKER_LIST_PATH}. "
            "Please create it first."
        )

    df = pd.read_csv(config.TICKER_LIST_PATH)

    required_cols = {"Symbol", "Company", "Sector"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Ticker CSV is missing required columns: {missing}"
        )

    logger.info(f"Loaded {len(df)} tickers from {config.TICKER_LIST_PATH}")
    return df


# ==============================================================================
# Single-Ticker Download
# ==============================================================================
def download_single_stock(ticker: str, force: bool = False) -> dict:
    """
    Download historical OHLCV data for a single ticker.

    Parameters
    ----------
    ticker : str
        Yahoo-Finance ticker symbol (e.g. ``"RELIANCE.NS"``).
    force : bool, optional
        If ``True``, re-download even when a local CSV already exists.
        Defaults to ``False``.

    Returns
    -------
    dict
        Summary dictionary with keys:

        - **ticker** (*str*) -- the ticker symbol.
        - **rows** (*int*) -- number of rows downloaded (0 on failure).
        - **date_range** (*tuple[str, str] | None*) -- earliest and latest
          dates in the downloaded data, or ``None`` on failure.
        - **status** (*str*) -- one of ``"downloaded"``, ``"skipped"``,
          or ``"failed"``.
        - **message** (*str*) -- human-readable status message.
    """
    filepath = os.path.join(config.RAW_DATA_DIR, f"{ticker}.csv")

    # Skip if file already exists and force is not set
    if os.path.exists(filepath) and not force:
        existing = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"[SKIP] {ticker}: CSV already exists ({len(existing)} rows) -- skipping")
        return {
            "ticker": ticker,
            "rows": len(existing),
            "date_range": (
                str(existing.index.min().date()),
                str(existing.index.max().date()),
            ),
            "status": "skipped",
            "message": "File already exists; use force=True to re-download.",
        }

    # Retry loop
    last_error: str = ""
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            logger.info(
                f"[DL] {ticker}: downloading (attempt {attempt}/{config.MAX_RETRIES})..."
            )

            df = yf.download(
                ticker,
                start=config.START_DATE,
                end=config.END_DATE,
                interval=config.DOWNLOAD_INTERVAL,
                progress=False,
            )

            # Handle multi-level columns returned by newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel("Ticker")

            if df.empty:
                last_error = "yfinance returned an empty DataFrame"
                logger.warning(f"[WARN] {ticker}: empty data on attempt {attempt}")
                if attempt < config.MAX_RETRIES:
                    time.sleep(config.RETRY_DELAY_SECONDS)
                continue

            # Ensure the expected columns are present
            expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            # yfinance >= 0.2.31 may label the column "Adj Close" or keep it
            # Some versions drop it; accept either set.
            for col in expected_cols:
                if col not in df.columns:
                    # "Adj Close" may be absent -- fill from "Close" as fallback
                    if col == "Adj Close" and "Close" in df.columns:
                        df["Adj Close"] = df["Close"]
                    else:
                        logger.warning(f"[WARN] {ticker}: missing expected column '{col}'")

            # Ensure Date index is named
            df.index.name = "Date"

            # Save
            save_dataframe(df, filepath, index=True)

            date_min = str(df.index.min().date())
            date_max = str(df.index.max().date())
            logger.info(
                f"[OK] {ticker}: saved {len(df)} rows "
                f"({date_min} -> {date_max}) to {filepath}"
            )

            return {
                "ticker": ticker,
                "rows": len(df),
                "date_range": (date_min, date_max),
                "status": "downloaded",
                "message": "Success",
            }

        except Exception as exc:
            last_error = str(exc)
            logger.error(
                f"[FAIL] {ticker}: error on attempt {attempt}/{config.MAX_RETRIES} -- {exc}"
            )
            if attempt < config.MAX_RETRIES:
                time.sleep(config.RETRY_DELAY_SECONDS)

    # All retries exhausted
    logger.error(f"[FAIL] {ticker}: all {config.MAX_RETRIES} attempts failed -- {last_error}")
    return {
        "ticker": ticker,
        "rows": 0,
        "date_range": None,
        "status": "failed",
        "message": f"Failed after {config.MAX_RETRIES} retries: {last_error}",
    }


# ==============================================================================
# Bulk Download
# ==============================================================================
@timer
def download_all_stocks(force: bool = False) -> dict:
    """
    Download historical OHLCV data for every ticker in the Nifty 50 list.

    Parameters
    ----------
    force : bool, optional
        If ``True``, re-download data even when a local CSV already exists.
        Defaults to ``False``.

    Returns
    -------
    dict
        Mapping of ``ticker -> summary_dict`` where each value has the
        same structure returned by :func:`download_single_stock`.
    """
    config.ensure_directories()
    ticker_df = _load_ticker_list()
    tickers = ticker_df["Symbol"].tolist()

    logger.info(
        f"Starting download for {len(tickers)} tickers "
        f"({config.START_DATE} -> {config.END_DATE}, "
        f"interval={config.DOWNLOAD_INTERVAL})"
    )

    results: dict = {}

    for ticker in tqdm(tickers, desc="Downloading stocks", unit="stock"):
        summary = download_single_stock(ticker, force=force)
        results[ticker] = summary

    # Log aggregate stats
    downloaded = sum(1 for v in results.values() if v["status"] == "downloaded")
    skipped = sum(1 for v in results.values() if v["status"] == "skipped")
    failed = sum(1 for v in results.values() if v["status"] == "failed")

    logger.info(
        f"Download complete -- "
        f"downloaded: {downloaded}, skipped: {skipped}, failed: {failed}"
    )

    if failed > 0:
        failed_tickers = [t for t, v in results.items() if v["status"] == "failed"]
        logger.warning(f"Failed tickers: {failed_tickers}")

    return results


# ==============================================================================
# Download Summary Report
# ==============================================================================
def generate_download_summary(results: dict | None = None) -> pd.DataFrame:
    """
    Create a summary report of downloaded raw data and save it as a CSV.

    If *results* is provided (from :func:`download_all_stocks`), the report
    is built directly from that dictionary.  Otherwise the function scans
    ``config.RAW_DATA_DIR`` for existing CSV files.

    Parameters
    ----------
    results : dict or None, optional
        Summary dictionary returned by :func:`download_all_stocks`.
        When ``None``, the function inspects files on disk.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with columns: ``Ticker``, ``Rows``,
        ``Start_Date``, ``End_Date``, ``Status``, ``Message``.
        Also saved to ``config.REPORTS_DIR/download_summary.csv``.
    """
    config.ensure_directories()

    if results is not None:
        rows = []
        for ticker, info in results.items():
            rows.append({
                "Ticker": ticker,
                "Rows": info["rows"],
                "Start_Date": info["date_range"][0] if info["date_range"] else None,
                "End_Date": info["date_range"][1] if info["date_range"] else None,
                "Status": info["status"],
                "Message": info["message"],
            })
        summary_df = pd.DataFrame(rows)
    else:
        # Build summary from files on disk
        from src.utils import list_csv_files, get_ticker_from_filename

        csv_files = list_csv_files(config.RAW_DATA_DIR)
        if not csv_files:
            logger.warning("No raw CSV files found to summarise.")
            summary_df = pd.DataFrame(
                columns=["Ticker", "Rows", "Start_Date", "End_Date", "Status", "Message"]
            )
        else:
            rows = []
            for fpath in csv_files:
                ticker = get_ticker_from_filename(fpath)
                try:
                    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                    rows.append({
                        "Ticker": ticker,
                        "Rows": len(df),
                        "Start_Date": str(df.index.min().date()),
                        "End_Date": str(df.index.max().date()),
                        "Status": "on_disk",
                        "Message": "Loaded from existing file.",
                    })
                except Exception as exc:
                    rows.append({
                        "Ticker": ticker,
                        "Rows": 0,
                        "Start_Date": None,
                        "End_Date": None,
                        "Status": "error",
                        "Message": str(exc),
                    })
            summary_df = pd.DataFrame(rows)

    # Persist report
    report_path = os.path.join(config.REPORTS_DIR, "download_summary.csv")
    save_dataframe(summary_df, report_path, index=False)
    logger.info(f"[INFO] Download summary saved to {report_path}")

    return summary_df


# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    results = download_all_stocks(force=False)
    generate_download_summary(results)
