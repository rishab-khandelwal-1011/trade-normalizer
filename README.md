# 📈 Stock Market Daily Trade Normalizer

> **Project ID:** ML_6  
> **Mapping:** Experiment 1, 2 — Data Preprocessing  
> **Team:** Om Bijwani (24ESKCY036) + Partner (B240296)

## 📋 Description

Process multi-stock historical datasets, engineer rolling window moving averages with NumPy, clean missing trade volumes, and plot comparative stock charts.

This project implements a complete data preprocessing pipeline for **Nifty 50** stock data spanning **5 years** of daily trading history.

---

## 🏗️ Project Structure

```
trade-normalizer/
│
├── main.py                          # CLI entry-point (full pipeline)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── data/
│   ├── nifty50_tickers.csv          # Nifty 50 ticker list with sectors
│   ├── raw/                         # Downloaded raw OHLCV CSVs
│   ├── cleaned/                     # After missing-value cleaning
│   └── processed/                   # After feature engineering
│
├── notebooks/
│   └── walkthrough.ipynb            # Interactive Jupyter walkthrough
│
├── src/
│   ├── __init__.py                  # Package metadata
│   ├── config.py                    # Central configuration
│   ├── data_downloader.py           # Download data via yfinance
│   ├── data_cleaner.py              # Clean missing volumes & prices
│   ├── feature_engineer.py          # Rolling MAs with pure NumPy
│   ├── normalizer.py                # Min-Max / Z-Score normalization
│   ├── visualizer.py                # Comparative stock charts
│   └── utils.py                     # Logging, I/O, profiling
│
├── outputs/
│   ├── charts/                      # Saved PNG & HTML charts
│   └── reports/                     # Data quality reports (CSVs)
│
└── tests/
    └── test_pipeline.py             # Unit tests (pytest)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Dashboard (Streamlit)

```bash
streamlit run app.py
```
*(or `python -m streamlit run app.py`)*

This opens the interactive browser dashboard at `http://localhost:8501` featuring:
- **Interactive Stock Explorer:** Candlesticks, price action, and pure-NumPy moving average overlays (SMA, EMA, WMA, Volatility).
- **Data Cleaning Inspector:** Side-by-side raw vs. cleaned volume imputation and 3-sigma outlier detection.
- **Cross-Stock Normalizer:** Multi-stock comparison in Min-Max $[0, 1]$ or Z-score scales.
- **Sector & Correlation Heatmaps:** Interactive return correlations and sector distributions.
- **Pipeline Controls:** One-click buttons to run pipeline steps directly from your browser.

### 3. Run the CLI Pipeline (Batch Mode)

```bash
python main.py --all
```

This will:
1. **Download** 5 years of daily OHLCV data for all 50 Nifty stocks
2. **Clean** missing trade volumes and price gaps
3. **Engineer** rolling window moving averages (SMA, EMA, WMA) using pure NumPy
4. **Normalize** data for cross-stock comparison
5. **Generate** comparative stock charts (PNG + interactive HTML)

### 3. Run Individual Steps

```bash
python main.py --download              # Download only
python main.py --clean                 # Clean only
python main.py --engineer              # Feature engineering only
python main.py --normalize             # Normalization only
python main.py --visualize             # Charts only
python main.py --download --clean      # Combine steps
python main.py --all --force           # Force re-download
python main.py --all --norm-method zscore  # Use Z-Score instead of Min-Max
```

### 4. Run Tests

```bash
pytest tests/test_pipeline.py -v
```

### 5. Jupyter Notebook

```bash
jupyter notebook notebooks/walkthrough.ipynb
```

---

## 🔧 Pipeline Details

### Step 1: Data Download (`data_downloader.py`)
- Downloads daily OHLCV data for all Nifty 50 stocks from Yahoo Finance
- Saves one CSV per ticker in `data/raw/`
- Includes retry logic and progress tracking
- Generates `download_summary.csv` report

### Step 2: Data Cleaning (`data_cleaner.py`)
- **Missing Volume**: forward-fill → backward-fill → median fill
- **Missing Prices**: linear interpolation for gaps ≤ 5 days; removal for larger gaps
- **Zero Volumes**: replaced with NaN and filled
- **Outlier Detection**: flags daily returns > 3σ with an `Outlier_Flag` column
- Generates `cleaning_report.csv` with before/after statistics

### Step 3: Feature Engineering (`feature_engineer.py`)
All rolling window computations use **pure NumPy** (no pandas `.rolling()`):

| Feature | Windows | Method |
|---------|---------|--------|
| SMA (Simple Moving Average) | 7, 14, 21, 50, 200 | `np.convolve` |
| EMA (Exponential Moving Average) | 12, 26, 50 | Iterative α formula |
| WMA (Weighted Moving Average) | 7, 14, 21 | Linear weights via `np.dot` |
| Rolling Volatility | 21 | Cumsum² trick |
| Volume SMA | 7, 14, 21, 50, 200 | `np.convolve` |
| Daily Returns | — | Array slicing |
| Cumulative Returns | — | `np.cumprod` |

### Step 4: Normalization (`normalizer.py`)
- **Min-Max Scaling**: scales to [0, 1] for price comparison
- **Z-Score Standardization**: mean=0, std=1 for volume comparison
- Saves scaling parameters for invertibility
- Creates cross-stock comparison DataFrames

### Step 5: Visualization (`visualizer.py`)
| Chart | Format |
|-------|--------|
| Comparative normalized prices (top N) | PNG |
| Moving averages per stock | PNG |
| Volume comparison bar chart | PNG |
| Daily return correlation heatmap | PNG |
| Missing data heatmap (before/after) | PNG |
| Sector-wise return box plots | PNG |
| Before/after cleaning comparison | PNG |
| Interactive price explorer | HTML (Plotly) |

---

## ⚙️ Configuration

All settings are centralized in [`src/config.py`](src/config.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `START_DATE` | 2021-10-01 | Data start date |
| `END_DATE` | 2026-10-01 | Data end date |
| `SMA_WINDOWS` | [7, 14, 21, 50, 200] | SMA window sizes |
| `EMA_WINDOWS` | [12, 26, 50] | EMA window sizes |
| `WMA_WINDOWS` | [7, 14, 21] | WMA window sizes |
| `OUTLIER_STD_THRESHOLD` | 3.0 | Outlier detection threshold |
| `MAX_CONSECUTIVE_NAN_DAYS` | 5 | Max gap for interpolation |
| `TOP_N_STOCKS` | 10 | Stocks shown in charts |

---

## 📦 Dependencies

- **numpy** — Core numerical computations and rolling window implementations
- **pandas** — Data loading and DataFrame management
- **yfinance** — Yahoo Finance data download
- **matplotlib** — Static chart generation
- **seaborn** — Statistical visualizations (heatmaps, box plots)
- **plotly** — Interactive HTML charts
- **scikit-learn** — Reference normalization utilities
- **tqdm** — Progress bars
- **pytest** — Unit testing

---

## 📊 Sample Outputs

After running `python main.py --all`, check:

- `data/raw/` — 50 raw CSV files (one per stock)
- `data/cleaned/` — 50 cleaned CSV files
- `data/processed/` — 50 feature-enriched CSV files
- `outputs/charts/` — All charts (PNG + HTML)
- `outputs/reports/` — Cleaning report, normalization params, download summary

---

## 📄 License

Academic project — SKIT, Jaipur.
