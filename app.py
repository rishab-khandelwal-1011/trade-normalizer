"""
Stock Market Daily Trade Normalizer (ML_6)
==========================================
Interactive Streamlit Web Dashboard

Mapping: Experiment 1, 2 -- Data Preprocessing
Team: Om Bijwani (24ESKCY036) + Partner (B240296)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add workspace to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src import config
from src.utils import load_dataframe, list_csv_files, get_ticker_from_filename
from src.feature_engineer import (
    compute_sma,
    compute_ema,
    compute_wma,
    compute_rolling_std,
    compute_daily_returns,
    compute_cumulative_returns,
)
from src.normalizer import minmax_normalize, zscore_normalize

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Stock Market Daily Trade Normalizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Data Loaders (Cached)
# ==============================================================================
@st.cache_data
def load_ticker_metadata():
    """Load Nifty 50 ticker list with company names and sectors."""
    if os.path.exists(config.TICKER_LIST_PATH):
        return pd.read_csv(config.TICKER_LIST_PATH)
    return pd.DataFrame(columns=["Symbol", "Company", "Sector"])


@st.cache_data
def load_stock_dataset(ticker: str, stage: str = "processed"):
    """Load dataset for a specific ticker and stage (raw, cleaned, processed)."""
    dir_map = {
        "raw": config.RAW_DATA_DIR,
        "cleaned": config.CLEANED_DATA_DIR,
        "processed": config.PROCESSED_DATA_DIR,
    }
    folder = dir_map.get(stage, config.PROCESSED_DATA_DIR)
    path = os.path.join(folder, f"{ticker}.csv")
    if os.path.exists(path):
        return load_dataframe(path)
    return None


@st.cache_data
def get_available_tickers():
    """Return all tickers available in processed data."""
    files = list_csv_files(config.PROCESSED_DATA_DIR)
    if not files:
        files = list_csv_files(config.RAW_DATA_DIR)
    return [get_ticker_from_filename(f) for f in files]


@st.cache_data
def load_reports():
    """Load audit reports if generated."""
    reports = {}
    cleaning_path = os.path.join(config.REPORTS_DIR, "cleaning_report.csv")
    download_path = os.path.join(config.REPORTS_DIR, "download_summary.csv")
    norm_path = os.path.join(config.REPORTS_DIR, "normalization_params.csv")

    if os.path.exists(cleaning_path):
        reports["cleaning"] = pd.read_csv(cleaning_path)
    if os.path.exists(download_path):
        reports["download"] = pd.read_csv(download_path)
    if os.path.exists(norm_path):
        reports["norm"] = pd.read_csv(norm_path)
    return reports


# ==============================================================================
# Header & Project Banner
# ==============================================================================
tickers_meta = load_ticker_metadata()
available_tickers = get_available_tickers()
reports_dict = load_reports()

col_title, col_info = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-header">📈 Stock Market Daily Trade Normalizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header"><b>Project ID:</b> ML_6 | <b>Mapping:</b> Experiment 1, 2 (Data Preprocessing) | <b>Universe:</b> Nifty 50 Historical Datasets</div>',
        unsafe_allow_html=True,
    )

with col_info:
    st.markdown(
        """
        <div style="text-align: right; font-size: 0.85rem; color: #64748B;">
            <b>Student 2:</b> Om Bijwani (24ESKCY036)<br>
            <b>Partner:</b> B240296<br>
            <b>Status:</b> <span style="color:#16A34A; font-weight:600;">Pipeline Active [OK]</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ==============================================================================
# Sidebar: Controls & Ticker Selection
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Dashboard Controls")

# Format ticker display
ticker_display_map = {}
for t in available_tickers:
    match = tickers_meta[tickers_meta["Symbol"] == t]
    if not match.empty:
        ticker_display_map[t] = f"{t.replace('.NS', '')} - {match.iloc[0]['Company']} ({match.iloc[0]['Sector']})"
    else:
        ticker_display_map[t] = t

selected_ticker = st.sidebar.selectbox(
    "Select Primary Stock",
    options=available_tickers,
    index=0 if available_tickers else None,
    format_func=lambda x: ticker_display_map.get(x, x),
)

# Global Date Range Selector
df_sample = load_stock_dataset(selected_ticker, "processed") if selected_ticker else None
if df_sample is not None and not df_sample.empty:
    min_date = df_sample.index.min().date()
    max_date = df_sample.index.max().date()
    date_range = st.sidebar.date_input(
        "Date Range Filter",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_filter, end_filter = date_range
    else:
        start_filter, end_filter = min_date, max_date
else:
    start_filter, end_filter = None, None

st.sidebar.divider()

# Pipeline Execution Triggers
st.sidebar.subheader("Pipeline Automation")
col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    if st.button("Run Cleaning", use_container_width=True):
        with st.spinner("Cleaning raw datasets..."):
            from src.data_cleaner import clean_all_stocks
            clean_all_stocks()
            st.cache_data.clear()
            st.success("Cleaned!")
with col_p2:
    if st.button("Re-Engineer", use_container_width=True):
        with st.spinner("Re-computing NumPy MAs..."):
            from src.feature_engineer import engineer_all_stocks
            engineer_all_stocks()
            st.cache_data.clear()
            st.success("Engineered!")

if st.sidebar.button("Run Full Pipeline (--all)", type="primary", use_container_width=True):
    with st.spinner("Executing full 5-stage pipeline..."):
        from src.data_cleaner import clean_all_stocks
        from src.feature_engineer import engineer_all_stocks
        from src.normalizer import normalize_all_stocks
        from src.visualizer import generate_all_charts
        clean_all_stocks()
        engineer_all_stocks()
        normalize_all_stocks()
        generate_all_charts()
        st.cache_data.clear()
        st.success("Full pipeline completed successfully!")

# Summary KPI Metrics
m1, m2, m3, m4 = st.columns(4)
total_stocks = len(available_tickers)
cleaning_df = reports_dict.get("cleaning", pd.DataFrame())
total_cleaned_vol = int(cleaning_df["Missing_Before_Volume"].sum()) if not cleaning_df.empty else 0
total_outliers = int(cleaning_df["Outliers_Flagged"].sum()) if not cleaning_df.empty else 0

m1.metric("Stocks Processed", f"{total_stocks} Stocks", "Nifty 50 Universe")
m2.metric("Historical Coverage", "5 Years", "2021 -> 2026 Daily OHLCV")
m3.metric("Missing Volumes Cleaned", f"{total_cleaned_vol:,} Records", "Imputed (Fwd/Bwd/Median)")
m4.metric("3-Sigma Outliers Flagged", f"{total_outliers} Detected", "Data Quality Audited")

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# Main Interactive Tabs
# ==============================================================================
tab_stock, tab_cleaning, tab_norm, tab_sector, tab_tables = st.tabs([
    "📈 Technical Analysis & NumPy Moving Averages",
    "🧹 Data Cleaning & Trade Volume Analysis",
    "⚖️ Multi-Stock Normalization & Comparison",
    "🌐 Sector Performance & Return Correlations",
    "📋 Dataset Explorer & Export",
])

# ------------------------------------------------------------------------------
# TAB 1: Technical Analysis & NumPy Moving Averages
# ------------------------------------------------------------------------------
with tab_stock:
    if df_sample is not None and not df_sample.empty:
        # Filter by date range
        mask = (df_sample.index.date >= start_filter) & (df_sample.index.date <= end_filter)
        df_stock = df_sample.loc[mask]

        st.subheader(f"{ticker_display_map.get(selected_ticker, selected_ticker)}")

        # Moving Average Toggle Controls
        col_ctrl1, col_ctrl2 = st.columns([3, 1])
        with col_ctrl1:
            ma_options = st.multiselect(
                "Overlay Moving Averages (Calculated with Pure NumPy):",
                options=["SMA_7", "SMA_14", "SMA_21", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "EMA_50", "WMA_7", "WMA_14", "WMA_21"],
                default=["SMA_21", "SMA_50", "SMA_200", "EMA_26"],
            )
        with col_ctrl2:
            chart_type = st.radio("Price Representation", ["Line Chart", "Candlestick"], horizontal=True)

        # Plotly Subplots: Price on top, Volume on bottom
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.7, 0.3],
            subplot_titles=[f"Price Action & Moving Averages ({selected_ticker})", "Trading Volume & Volume SMA"]
        )

        # Price Trace
        if chart_type == "Candlestick" and all(k in df_stock.columns for k in ["Open", "High", "Low", "Close"]):
            fig.add_trace(
                go.Candlestick(
                    x=df_stock.index,
                    open=df_stock["Open"],
                    high=df_stock["High"],
                    low=df_stock["Low"],
                    close=df_stock["Close"],
                    name="OHLC",
                ),
                row=1, col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df_stock.index,
                    y=df_stock["Close"],
                    mode="lines",
                    name="Close Price",
                    line=dict(color="#1E3A8A", width=1.5),
                ),
                row=1, col=1,
            )

        # Moving Average Overlays
        palette = px.colors.qualitative.Plotly
        for idx, ma in enumerate(ma_options):
            if ma in df_stock.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_stock.index,
                        y=df_stock[ma],
                        mode="lines",
                        name=ma,
                        line=dict(width=1.3, color=palette[idx % len(palette)]),
                    ),
                    row=1, col=1,
                )

        # Volume Subplot
        vol_colors = np.where(df_stock["Close"] >= df_stock["Open"], "#16A34A", "#DC2626") if "Open" in df_stock.columns else "#64748B"
        fig.add_trace(
            go.Bar(
                x=df_stock.index,
                y=df_stock["Volume"],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.7,
            ),
            row=2, col=1,
        )

        # Volume SMA overlay
        if "Vol_SMA_21" in df_stock.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_stock.index,
                    y=df_stock["Vol_SMA_21"],
                    mode="lines",
                    name="Vol SMA(21)",
                    line=dict(color="#D97706", width=1.4),
                ),
                row=2, col=1,
            )

        fig.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volatility & Returns Section
        col_ret1, col_ret2 = st.columns(2)
        with col_ret1:
            st.markdown("##### 21-Day Rolling Volatility (NumPy)")
            if "Volatility_21" in df_stock.columns:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=df_stock.index, y=df_stock["Volatility_21"],
                    mode="lines", fill="tozeroy",
                    line=dict(color="#9333EA", width=1.2),
                    name="Volatility (21d)",
                ))
                fig_vol.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=20), template="plotly_white")
                st.plotly_chart(fig_vol, use_container_width=True)

        with col_ret2:
            st.markdown("##### Cumulative Returns (%)")
            if "Cumulative_Return" in df_stock.columns:
                cum_pct = (df_stock["Cumulative_Return"] - 1.0) * 100
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=df_stock.index, y=cum_pct,
                    mode="lines",
                    line=dict(color="#059669", width=1.5),
                    name="Cumulative Return %",
                ))
                fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_cum.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=20), template="plotly_white")
                st.plotly_chart(fig_cum, use_container_width=True)

        with st.expander("ℹ️ Pure NumPy Implementation Details (Official Requirement)"):
            st.markdown(
                """
                - **SMA via `np.convolve`**: Computes running unweighted average with an `np.ones(w)/w` filter.
                - **EMA via Iterative Alpha**: Computes exponential decay $EMA_t = \\alpha P_t + (1-\\alpha) EMA_{t-1}$ where $\\alpha = 2/(w+1)$.
                - **WMA via Linear Weight Dot Products**: Computes linear ramp weights $\\frac{\\sum_{i=1}^{w} i \\cdot P_{t-w+i}}{\\sum i}$.
                - **Zero pandas `.rolling()`**: Strictly satisfies the core project requirement of pure NumPy feature engineering.
                """
            )
    else:
        st.warning("No data found for the selected stock. Please run the pipeline first.")

# ------------------------------------------------------------------------------
# TAB 2: Data Cleaning & Trade Volume Imputation
# ------------------------------------------------------------------------------
with tab_cleaning:
    st.subheader("Data Cleaning & Quality Assurance")
    st.markdown(
        "Demonstrates the treatment of **missing trade volumes**, zero-volume anomalies, linear OHLC interpolation, and $3\\sigma$ outlier detection."
    )

    df_raw = load_stock_dataset(selected_ticker, "raw")
    df_cleaned = load_stock_dataset(selected_ticker, "cleaned")

    if df_raw is not None and df_cleaned is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Raw Volume Series** ({selected_ticker})")
            zero_count = (df_raw["Volume"] <= config.MIN_VALID_VOLUME).sum()
            nan_count = df_raw["Volume"].isna().sum()
            st.caption(f"Raw issues: {nan_count} NaNs, {zero_count} zero/sub-threshold volume days.")

            fig_raw_v = go.Figure()
            fig_raw_v.add_trace(go.Scatter(
                x=df_raw.index, y=df_raw["Volume"],
                mode="lines", line=dict(color="#EF4444", width=0.8),
                name="Raw Volume"
            ))
            fig_raw_v.update_layout(height=320, template="plotly_white", margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_raw_v, use_container_width=True)

        with c2:
            st.markdown(f"**Cleaned Volume Series** ({selected_ticker})")
            st.caption("Cleaned with: Forward-Fill -> Backward-Fill -> Column Median Fill.")

            fig_clean_v = go.Figure()
            fig_clean_v.add_trace(go.Scatter(
                x=df_cleaned.index, y=df_cleaned["Volume"],
                mode="lines", line=dict(color="#10B981", width=0.8),
                name="Cleaned Volume"
            ))
            fig_clean_v.update_layout(height=320, template="plotly_white", margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_clean_v, use_container_width=True)

        # Outlier Detection Plot
        st.markdown(f"##### 3-Sigma Daily Return Outlier Detection ({selected_ticker})")
        if "Outlier_Flag" in df_cleaned.columns and "Daily_Return" in df_cleaned.columns:
            normal_mask = ~df_cleaned["Outlier_Flag"]
            outlier_mask = df_cleaned["Outlier_Flag"]

            fig_out = go.Figure()
            fig_out.add_trace(go.Scatter(
                x=df_cleaned.index[normal_mask],
                y=df_cleaned.loc[normal_mask, "Daily_Return"],
                mode="markers",
                marker=dict(color="#3B82F6", size=4, opacity=0.6),
                name="Normal Days"
            ))
            fig_out.add_trace(go.Scatter(
                x=df_cleaned.index[outlier_mask],
                y=df_cleaned.loc[outlier_mask, "Daily_Return"],
                mode="markers",
                marker=dict(color="#DC2626", size=8, symbol="x"),
                name=f"Outliers (>3σ): {outlier_mask.sum()} days"
            ))
            fig_out.update_layout(height=320, template="plotly_white", margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_out, use_container_width=True)

    # Full Cleaning Audit Table
    st.markdown("##### Cleaning Report Audit (All Stocks)")
    if "cleaning" in reports_dict:
        st.dataframe(reports_dict["cleaning"], use_container_width=True, height=280)
    else:
        st.info("Run the pipeline to generate the full cleaning report.")

# ------------------------------------------------------------------------------
# TAB 3: Multi-Stock Normalization & Comparison
# ------------------------------------------------------------------------------
with tab_norm:
    st.subheader("Multi-Stock Cross-Comparison & Normalization")
    st.markdown(
        "Normalizing stocks to a common scale ($[0, 1]$ Min-Max or $\\mu=0, \\sigma=1$ Z-Score) is essential because absolute share prices vary widely across blue-chips."
    )

    col_norm_ctrl1, col_norm_ctrl2 = st.columns([3, 1])
    with col_norm_ctrl1:
        comp_tickers = st.multiselect(
            "Select Stocks to Compare:",
            options=available_tickers,
            default=available_tickers[:6] if len(available_tickers) >= 6 else available_tickers,
            format_func=lambda x: ticker_display_map.get(x, x),
        )
    with col_norm_ctrl2:
        norm_type = st.radio("Normalization Technique", ["Min-Max Scaling [0, 1]", "Z-Score Standardization"], horizontal=True)

    if comp_tickers:
        fig_comp = go.Figure()

        for t in comp_tickers:
            df_t = load_stock_dataset(t, "processed")
            if df_t is not None and not df_t.empty:
                # Filter date
                m = (df_t.index.date >= start_filter) & (df_t.index.date <= end_filter)
                df_sub = df_t.loc[m]
                prices = df_sub["Close"].values

                try:
                    if "Min-Max" in norm_type:
                        scaled, _ = minmax_normalize(prices)
                    else:
                        scaled, _ = zscore_normalize(prices)

                    fig_comp.add_trace(go.Scatter(
                        x=df_sub.index,
                        y=scaled,
                        mode="lines",
                        name=t.replace(".NS", ""),
                        line=dict(width=1.5),
                    ))
                except Exception:
                    continue

        y_label = "Normalized Price [0, 1]" if "Min-Max" in norm_type else "Standardized Z-Score (σ)"
        fig_comp.update_layout(
            height=500,
            title=f"Comparative Stock Trajectory ({norm_type})",
            yaxis_title=y_label,
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Normalization parameters table
    if "norm" in reports_dict:
        with st.expander("📊 View Stored Normalization Parameters (Min, Max, Mean, Std)"):
            st.dataframe(reports_dict["norm"], use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: Sector Performance & Return Correlations
# ------------------------------------------------------------------------------
with tab_sector:
    st.subheader("Sector Performance & Cross-Stock Correlations")

    col_sec1, col_sec2 = st.columns([1, 1])

    # Sector Returns Boxplot
    with col_sec1:
        st.markdown("##### Average Daily Return Distribution by Sector")
        sector_records = []
        for t in available_tickers:
            meta_row = tickers_meta[tickers_meta["Symbol"] == t]
            if not meta_row.empty:
                sec = meta_row.iloc[0]["Sector"]
                df_t = load_stock_dataset(t, "processed")
                if df_t is not None and "Daily_Return" in df_t.columns:
                    sector_records.append({
                        "Ticker": t.replace(".NS", ""),
                        "Sector": sec,
                        "Mean_Return_Pct": df_t["Daily_Return"].mean() * 100,
                    })

        if sector_records:
            sec_df = pd.DataFrame(sector_records)
            fig_sec = px.box(
                sec_df, x="Sector", y="Mean_Return_Pct", color="Sector",
                title="Sector Mean Daily Return (%)",
                points="all",
            )
            fig_sec.update_layout(height=450, showlegend=False, template="plotly_white")
            st.plotly_chart(fig_sec, use_container_width=True)

    # Average Volume by Sector
    with col_sec2:
        st.markdown("##### Mean Trading Volume by Sector")
        vol_records = []
        for t in available_tickers:
            meta_row = tickers_meta[tickers_meta["Symbol"] == t]
            if not meta_row.empty:
                sec = meta_row.iloc[0]["Sector"]
                df_t = load_stock_dataset(t, "processed")
                if df_t is not None and "Volume" in df_t.columns:
                    vol_records.append({"Sector": sec, "Volume": df_t["Volume"].mean()})

        if vol_records:
            vol_df = pd.DataFrame(vol_records).groupby("Sector")["Volume"].mean().reset_index()
            vol_df["Volume_M"] = vol_df["Volume"] / 1e6
            fig_vol_sec = px.bar(
                vol_df.sort_values("Volume_M", ascending=False),
                x="Sector", y="Volume_M",
                color="Sector",
                title="Mean Daily Trading Volume (in Millions)",
            )
            fig_vol_sec.update_layout(height=450, showlegend=False, template="plotly_white")
            st.plotly_chart(fig_vol_sec, use_container_width=True)

    # Full Correlation Heatmap
    st.markdown("##### Interactive Daily Return Correlation Matrix (Nifty 50)")
    returns_matrix = {}
    for t in available_tickers[:25]:  # Top 25 for clear visualization
        df_t = load_stock_dataset(t, "processed")
        if df_t is not None and "Daily_Return" in df_t.columns:
            returns_matrix[t.replace(".NS", "")] = df_t["Daily_Return"]

    if returns_matrix:
        corr_df = pd.DataFrame(returns_matrix).corr()
        fig_corr = px.imshow(
            corr_df,
            color_continuous_scale="RdBu_r",
            zmin=-0.5, zmax=1.0,
            title="Correlation Heatmap (Daily Returns)",
        )
        fig_corr.update_layout(height=650, template="plotly_white")
        st.plotly_chart(fig_corr, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: Dataset Explorer & Export
# ------------------------------------------------------------------------------
with tab_tables:
    st.subheader("Dataset Explorer & Export")

    c_sel1, c_sel2 = st.columns(2)
    with c_sel1:
        tbl_ticker = st.selectbox("Select Stock to Inspect", available_tickers, format_func=lambda x: ticker_display_map.get(x, x), key="tbl_stock")
    with c_sel2:
        tbl_stage = st.selectbox("Select Pipeline Stage", ["processed (Feature-Enriched)", "cleaned (Missing Volumes Imputed)", "raw (Direct from Yahoo)"])

    stage_clean = tbl_stage.split()[0]
    df_inspect = load_stock_dataset(tbl_ticker, stage_clean)

    if df_inspect is not None and not df_inspect.empty:
        st.markdown(f"**Previewing:** `{tbl_ticker}` | **Stage:** `{stage_clean}` | **Shape:** `{df_inspect.shape[0]} rows x {df_inspect.shape[1]} columns`")

        st.dataframe(df_inspect.head(100), use_container_width=True, height=350)

        csv_bytes = df_inspect.to_csv().encode("utf-8")
        st.download_button(
            label=f"📥 Download {tbl_ticker}_{stage_clean}.csv",
            data=csv_bytes,
            file_name=f"{tbl_ticker}_{stage_clean}.csv",
            mime="text/csv",
        )
    else:
        st.info("No dataset available for this selection.")

st.divider()
st.caption("Stock Market Daily Trade Normalizer | SKIT Academic ML Project | Built with Python, NumPy, Pandas, Plotly & Streamlit")
