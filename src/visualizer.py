"""
visualizer.py -- Charts & Visualizations
=========================================

Creates all charts and visualizations for the Stock Market Daily
Trade Normalizer project.  Every plotting function saves its output
to ``config.CHARTS_DIR`` and closes the figure to free memory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    PLOTLY_AVAILABLE = False

from src import config
from src.utils import (
    get_logger,
    load_dataframe,
    list_csv_files,
    get_ticker_from_filename,
    timer,
)

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Global style
# --------------------------------------------------------------------------- #
try:
    plt.style.use(config.CHART_STYLE)
except Exception:  # pragma: no cover - fall back if style unavailable
    logger.warning(
        "Chart style '%s' not available; using matplotlib defaults.",
        config.CHART_STYLE,
    )


# =========================================================================== #
# Helper utilities (private)
# =========================================================================== #
def _ensure_charts_dir() -> None:
    """Create the charts output directory if it does not exist."""
    os.makedirs(config.CHARTS_DIR, exist_ok=True)


def _get_top_n_tickers(n: int = config.TOP_N_STOCKS) -> list:
    """
    Return the top *n* ticker symbols ranked by average daily volume.

    Processed CSV files in ``config.PROCESSED_DATA_DIR`` are scanned.  Each
    file's mean ``Volume`` is computed and the *n* tickers with the highest
    average volume are returned.

    Parameters
    ----------
    n : int
        Number of top tickers to return.

    Returns
    -------
    list of str
        Ticker symbols sorted by descending average volume.
    """
    csv_files = list_csv_files(config.PROCESSED_DATA_DIR)
    if not csv_files:
        logger.warning("No processed CSV files found.")
        return []

    volumes = {}
    for fpath in csv_files:
        ticker = get_ticker_from_filename(fpath)
        try:
            df = load_dataframe(fpath)
            if "Volume" in df.columns:
                volumes[ticker] = df["Volume"].mean()
        except Exception as exc:
            logger.debug("Skipping %s: %s", ticker, exc)

    ranked = sorted(volumes, key=volumes.get, reverse=True)
    return ranked[:n]


# =========================================================================== #
# 1. Comparative normalised prices
# =========================================================================== #
@timer
def plot_comparative_prices(top_n: int = config.TOP_N_STOCKS) -> None:
    """
    Plot normalised Adj Close prices for the top *n* stocks.

    Each stock's Adjusted Close is min-max scaled to [0, 1] so that price
    trajectories can be compared on one chart regardless of absolute price.

    Parameters
    ----------
    top_n : int, optional
        Number of top stocks (by average volume) to include.
        Defaults to ``config.TOP_N_STOCKS``.
    """
    _ensure_charts_dir()
    tickers = _get_top_n_tickers(top_n)
    if not tickers:
        logger.warning("No tickers available for comparative price chart.")
        return

    fig, ax = plt.subplots(figsize=config.FIG_SIZE, dpi=config.FIG_DPI)

    for ticker in tickers:
        fpath = os.path.join(config.PROCESSED_DATA_DIR, f"{ticker}.csv")
        try:
            df = load_dataframe(fpath)
        except FileNotFoundError:
            logger.debug("File not found for %s; skipping.", ticker)
            continue

        if "Adj Close" not in df.columns:
            logger.debug("'Adj Close' column missing for %s; skipping.", ticker)
            continue

        series = df["Adj Close"].dropna()
        if series.empty:
            continue

        # Min-max normalisation to [0, 1]
        s_min, s_max = series.min(), series.max()
        if s_max - s_min == 0:
            normalised = series * 0.0
        else:
            normalised = (series - s_min) / (s_max - s_min)

        ax.plot(normalised.index, normalised.values, label=ticker, linewidth=1.0)

    ax.set_title("Comparative Normalised Prices (Top Stocks by Volume)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalised Adj Close [0, 1]")
    ax.legend(fontsize="small", loc="best")

    plt.tight_layout()
    save_path = os.path.join(config.CHARTS_DIR, "comparative_prices.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved comparative prices chart -> %s", save_path)


# =========================================================================== #
# 2. Moving averages for a single ticker
# =========================================================================== #
@timer
def plot_moving_averages(ticker: str) -> None:
    """
    Plot Close price with SMA overlays for a single stock.

    Overlays plotted: SMA_7, SMA_21, SMA_50, SMA_200.

    Parameters
    ----------
    ticker : str
        Ticker symbol whose processed CSV will be loaded.
    """
    _ensure_charts_dir()
    fpath = os.path.join(config.PROCESSED_DATA_DIR, f"{ticker}.csv")
    try:
        df = load_dataframe(fpath)
    except FileNotFoundError:
        logger.error("Processed data not found for %s.", ticker)
        return

    fig, ax = plt.subplots(figsize=config.FIG_SIZE, dpi=config.FIG_DPI)

    if "Close" in df.columns:
        ax.plot(df.index, df["Close"], label="Close", linewidth=1.2, alpha=0.9)

    sma_cols = ["SMA_7", "SMA_21", "SMA_50", "SMA_200"]
    for col in sma_cols:
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, linewidth=0.9, alpha=0.75)

    ax.set_title(f"{ticker} -- Close Price & Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(fontsize="small", loc="best")

    plt.tight_layout()
    save_path = os.path.join(config.CHARTS_DIR, f"{ticker}_moving_averages.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved moving averages chart -> %s", save_path)


# =========================================================================== #
# 3. Volume comparison bar chart
# =========================================================================== #
@timer
def plot_volume_comparison(top_n: int = config.TOP_N_STOCKS) -> None:
    """
    Bar chart comparing average daily volume across the top *n* stocks.

    Parameters
    ----------
    top_n : int, optional
        Number of top stocks to include.
        Defaults to ``config.TOP_N_STOCKS``.
    """
    _ensure_charts_dir()
    csv_files = list_csv_files(config.PROCESSED_DATA_DIR)
    if not csv_files:
        logger.warning("No processed CSV files found for volume comparison.")
        return

    volumes: dict[str, float] = {}
    for fpath in csv_files:
        ticker = get_ticker_from_filename(fpath)
        try:
            df = load_dataframe(fpath)
            if "Volume" in df.columns:
                volumes[ticker] = df["Volume"].mean()
        except Exception:
            continue

    if not volumes:
        logger.warning("No volume data available.")
        return

    # Sort and take top_n
    sorted_items = sorted(volumes.items(), key=lambda x: x[1], reverse=True)[:top_n]
    tickers = [t for t, _ in sorted_items]
    avg_vols = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=config.FIG_SIZE, dpi=config.FIG_DPI)
    colors = sns.color_palette(config.COLOR_PALETTE, n_colors=len(tickers))
    ax.bar(tickers, avg_vols, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_title("Average Daily Volume -- Top Stocks")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Average Volume")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    save_path = os.path.join(config.CHARTS_DIR, "volume_comparison.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved volume comparison chart -> %s", save_path)


# =========================================================================== #
# 4. Correlation heatmap of daily returns
# =========================================================================== #
@timer
def plot_correlation_heatmap() -> None:
    """
    Compute pairwise correlations of daily returns across all stocks
    and display a Seaborn heatmap.

    Daily returns are derived from the ``Adj Close`` column of each
    processed CSV file.
    """
    _ensure_charts_dir()
    csv_files = list_csv_files(config.PROCESSED_DATA_DIR)
    if not csv_files:
        logger.warning("No processed CSV files found for correlation heatmap.")
        return

    returns_dict: dict[str, pd.Series] = {}
    for fpath in csv_files:
        ticker = get_ticker_from_filename(fpath)
        try:
            df = load_dataframe(fpath)
            if "Adj Close" in df.columns:
                returns_dict[ticker] = df["Adj Close"].pct_change().dropna()
        except Exception:
            continue

    if len(returns_dict) < 2:
        logger.warning("Need >= 2 stocks for a correlation heatmap.")
        return

    returns_df = pd.DataFrame(returns_dict)
    corr = returns_df.corr()

    fig, ax = plt.subplots(figsize=config.FIG_SIZE, dpi=config.FIG_DPI)
    sns.heatmap(
        corr,
        annot=True if len(corr) <= 15 else False,
        fmt=".2f" if len(corr) <= 15 else "",
        cmap="coolwarm",
        linewidths=0.3,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Daily Return Correlations Across Stocks")

    plt.tight_layout()
    save_path = os.path.join(config.CHARTS_DIR, "correlation_heatmap.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved correlation heatmap -> %s", save_path)


# =========================================================================== #
# 5. Missing-data heatmap
# =========================================================================== #
@timer
def plot_missing_data_heatmap(stage: str = "raw") -> None:
    """
    Heatmap showing missing-data patterns for each stock and column.

    Parameters
    ----------
    stage : str, optional
        Data stage to inspect -- ``'raw'`` or ``'cleaned'``.
        Defaults to ``'raw'``.
    """
    _ensure_charts_dir()
    if stage == "cleaned":
        data_dir = config.CLEANED_DATA_DIR
    else:
        data_dir = config.RAW_DATA_DIR

    csv_files = list_csv_files(data_dir)
    if not csv_files:
        logger.warning("No CSV files found in %s.", data_dir)
        return

    missing_records: list[dict] = []
    for fpath in csv_files:
        ticker = get_ticker_from_filename(fpath)
        try:
            df = load_dataframe(fpath)
            missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
            missing_pct["Ticker"] = ticker
            missing_records.append(missing_pct)
        except Exception:
            continue

    if not missing_records:
        logger.warning("Could not compute missing-data statistics.")
        return

    missing_df = pd.DataFrame(missing_records).set_index("Ticker")
    missing_df = missing_df.sort_index()

    fig, ax = plt.subplots(figsize=config.FIG_SIZE, dpi=config.FIG_DPI)
    sns.heatmap(
        missing_df,
        annot=True if len(missing_df) <= 20 else False,
        fmt=".1f" if len(missing_df) <= 20 else "",
        cmap="YlOrRd",
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title(f"Missing Data (%) -- {stage.capitalize()} Stage")
    ax.set_xlabel("Column")
    ax.set_ylabel("Ticker")

    plt.tight_layout()
    save_path = os.path.join(config.CHARTS_DIR, f"missing_data_{stage}.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved missing-data heatmap -> %s", save_path)


# =========================================================================== #
# 6. Sector returns box plot
# =========================================================================== #
@timer
def plot_sector_returns() -> None:
    """
    Box plot of average daily returns grouped by sector.

    Sector mappings are read from ``config.TICKER_LIST_PATH``.  Processed
    data is used to compute each stock's mean daily return, then plotted
    by sector.
    """
    _ensure_charts_dir()

    # Load sector info
    try:
        tickers_df = pd.read_csv(config.TICKER_LIST_PATH)
    except FileNotFoundError:
        logger.error("Ticker list not found at %s.", config.TICKER_LIST_PATH)
        return

    if "Sector" not in tickers_df.columns or "Symbol" not in tickers_df.columns:
        logger.error("Ticker list must contain 'Symbol' and 'Sector' columns.")
        return

    sector_map = dict(zip(tickers_df["Symbol"], tickers_df["Sector"]))

    records: list[dict] = []
    csv_files = list_csv_files(config.PROCESSED_DATA_DIR)
    for fpath in csv_files:
        ticker = get_ticker_from_filename(fpath)
        sector = sector_map.get(ticker)
        if sector is None:
            continue
        try:
            df = load_dataframe(fpath)
            if "Adj Close" in df.columns:
                daily_returns = df["Adj Close"].pct_change().dropna()
                mean_return = daily_returns.mean() * 100  # percentage
                records.append(
                    {"Ticker": ticker, "Sector": sector, "Mean Daily Return (%)": mean_return}
                )
        except Exception:
            continue

    if not records:
        logger.warning("No sector-return data available.")
        return

    ret_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=config.FIG_SIZE, dpi=config.FIG_DPI)
    sns.boxplot(
        data=ret_df,
        x="Sector",
        y="Mean Daily Return (%)",
        hue="Sector",
        palette=config.COLOR_PALETTE,
        legend=False,
        ax=ax,
    )
    ax.set_title("Average Daily Returns by Sector")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    save_path = os.path.join(config.CHARTS_DIR, "sector_returns.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved sector returns chart -> %s", save_path)


# =========================================================================== #
# 7. Before / after cleaning comparison
# =========================================================================== #
@timer
def plot_before_after_cleaning(ticker: str) -> None:
    """
    Side-by-side subplots of Volume before and after cleaning.

    Parameters
    ----------
    ticker : str
        Ticker symbol to compare.
    """
    _ensure_charts_dir()
    raw_path = os.path.join(config.RAW_DATA_DIR, f"{ticker}.csv")
    clean_path = os.path.join(config.CLEANED_DATA_DIR, f"{ticker}.csv")

    try:
        raw_df = load_dataframe(raw_path)
    except FileNotFoundError:
        logger.error("Raw data not found for %s.", ticker)
        return

    try:
        clean_df = load_dataframe(clean_path)
    except FileNotFoundError:
        logger.error("Cleaned data not found for %s.", ticker)
        return

    fig, axes = plt.subplots(1, 2, figsize=config.FIG_SIZE, dpi=config.FIG_DPI)

    # Raw volume
    if "Volume" in raw_df.columns:
        axes[0].plot(raw_df.index, raw_df["Volume"], color="tomato", linewidth=0.8)
    axes[0].set_title(f"{ticker} -- Volume (Raw)")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Volume")

    # Cleaned volume
    if "Volume" in clean_df.columns:
        axes[1].plot(clean_df.index, clean_df["Volume"], color="steelblue", linewidth=0.8)
    axes[1].set_title(f"{ticker} -- Volume (Cleaned)")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Volume")

    plt.tight_layout()
    save_path = os.path.join(config.CHARTS_DIR, f"{ticker}_cleaning_comparison.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved cleaning comparison chart -> %s", save_path)


# =========================================================================== #
# 8. Interactive prices (Plotly)
# =========================================================================== #
@timer
def plot_interactive_prices(top_n: int = config.TOP_N_STOCKS) -> None:
    """
    Interactive Plotly line chart of normalised Adj Close prices for
    the top *n* stocks.  Saved as an HTML file.

    Parameters
    ----------
    top_n : int, optional
        Number of top stocks (by average volume) to include.
        Defaults to ``config.TOP_N_STOCKS``.
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("plotly is not installed; skipping interactive chart.")
        return

    _ensure_charts_dir()
    tickers = _get_top_n_tickers(top_n)
    if not tickers:
        logger.warning("No tickers available for interactive price chart.")
        return

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        fpath = os.path.join(config.PROCESSED_DATA_DIR, f"{ticker}.csv")
        try:
            df = load_dataframe(fpath)
        except FileNotFoundError:
            continue

        if "Adj Close" not in df.columns:
            continue

        series = df["Adj Close"].dropna()
        if series.empty:
            continue

        s_min, s_max = series.min(), series.max()
        if s_max - s_min == 0:
            normalised = series * 0.0
        else:
            normalised = (series - s_min) / (s_max - s_min)

        tmp = normalised.reset_index()
        tmp.columns = ["Date", "Normalised Adj Close"]
        tmp["Ticker"] = ticker
        frames.append(tmp)

    if not frames:
        logger.warning("No data available for interactive chart.")
        return

    all_data = pd.concat(frames, ignore_index=True)

    fig = px.line(
        all_data,
        x="Date",
        y="Normalised Adj Close",
        color="Ticker",
        title="Interactive Normalised Prices (Top Stocks by Volume)",
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Normalised Adj Close [0, 1]",
        legend_title="Ticker",
        hovermode="x unified",
    )

    save_path = os.path.join(config.CHARTS_DIR, "interactive_prices.html")
    fig.write_html(save_path)
    logger.info("Saved interactive prices chart -> %s", save_path)


# =========================================================================== #
# 9. Generate all charts
# =========================================================================== #
@timer
def generate_all_charts() -> None:
    """
    Run every visualisation function in sequence and log progress.

    The first ticker found in processed data is used for single-stock
    charts (moving averages, before/after cleaning).
    """
    logger.info("=" * 60)
    logger.info("Starting full chart generation run ...")
    logger.info("=" * 60)

    # Determine a sample ticker for single-stock charts
    csv_files = list_csv_files(config.PROCESSED_DATA_DIR)
    sample_ticker = get_ticker_from_filename(csv_files[0]) if csv_files else None

    # 1
    logger.info("[1/8] Comparative prices ...")
    plot_comparative_prices()

    # 2
    if sample_ticker:
        logger.info("[2/8] Moving averages for %s ...", sample_ticker)
        plot_moving_averages(sample_ticker)
    else:
        logger.warning("[2/8] Skipping moving averages -- no processed data.")

    # 3
    logger.info("[3/8] Volume comparison ...")
    plot_volume_comparison()

    # 4
    logger.info("[4/8] Correlation heatmap ...")
    plot_correlation_heatmap()

    # 5
    logger.info("[5/8] Missing-data heatmap (raw) ...")
    plot_missing_data_heatmap(stage="raw")

    # 6
    logger.info("[6/8] Sector returns ...")
    plot_sector_returns()

    # 7
    if sample_ticker:
        logger.info("[7/8] Before/after cleaning for %s ...", sample_ticker)
        plot_before_after_cleaning(sample_ticker)
    else:
        logger.warning("[7/8] Skipping cleaning comparison -- no processed data.")

    # 8
    logger.info("[8/8] Interactive prices (Plotly) ...")
    plot_interactive_prices()

    logger.info("=" * 60)
    logger.info("All charts generated successfully [OK]")
    logger.info("=" * 60)
