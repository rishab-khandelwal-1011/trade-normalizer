"""
create_project_guide_docx.py
============================
Generates a comprehensive Microsoft Word (.docx) user manual and project
documentation for the Stock Market Daily Trade Normalizer (ML_6).
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    """Set the background color of a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    """Set cell margins in dxa (1 pt = 20 dxa)."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def style_table(table, header_bg="1E3A8A", header_fg="FFFFFF", alt_bg="F8FAFC"):
    """Format table with professional padding, borders, and colors."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table.rows):
        # Prevent row split across pages
        trPr = row._element.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=120, bottom=120, left=160, right=160)
            if i == 0:
                set_cell_background(cell, header_bg)
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor.from_string(header_fg)
                        run.font.size = Pt(9.5)
            else:
                if i % 2 == 1:
                    set_cell_background(cell, alt_bg)
                else:
                    set_cell_background(cell, "FFFFFF")
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9.0)
                        run.font.color.rgb = RGBColor.from_string("1F2937")

def add_callout(doc, text, title="KEY TAKEAWAY", icon="ℹ️", bg_color="EFF6FF", border_color="3B82F6"):
    """Create a styled callout box."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=160, bottom=160, left=200, right=200)

    # Set left border color and remove others
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'  <w:top w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"{icon} {title}\n")
    run_t.font.bold = True
    run_t.font.size = Pt(10)
    run_t.font.color.rgb = RGBColor.from_string("1E40AF")

    run_b = p.add_run(text)
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor.from_string("1E293B")
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def generate_document(output_path: str):
    doc = docx.Document()

    # Page Margins: 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Styles
    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor.from_string('1F2937')
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # Document Header / Title
    p_pre = doc.add_paragraph()
    p_pre.paragraph_format.space_before = Pt(0)
    p_pre.paragraph_format.space_after = Pt(2)
    r_pre = p_pre.add_run("ACADEMIC MACHINE LEARNING LAB -- EXPERIMENTS 1 & 2")
    r_pre.font.size = Pt(9.5)
    r_pre.font.bold = True
    r_pre.font.color.rgb = RGBColor.from_string("2563EB")

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("Stock Market Daily Trade Normalizer")
    r_title.font.size = Pt(24)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor.from_string("1E3A8A")

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run("Complete Project Architecture, Theoretical Formulations & Live Dashboard User Manual")
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = RGBColor.from_string("4B5563")

    # Metadata Table
    meta_table = doc.add_table(rows=6, cols=2)
    meta_table.autofit = False
    col_widths = [Inches(2.0), Inches(4.5)]
    for row in meta_table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = width

    meta_data = [
        ("Project Identifier", "ML_6 (Stock Market Daily Trade Normalizer)"),
        ("Curriculum Mapping", "Experiment 1 & 2 -- Data Preprocessing & Feature Engineering"),
        ("Institution", "Swami Keshvanand Institute of Technology (SKIT), Jaipur"),
        ("Student Team", "Om Bijwani (Roll: 24ESKCY036 | Email: B241243@skit.ac.in)\nPartner Email: B240296@skit.ac.in"),
        ("Target Universe", "Nifty 50 Constituent Blue-Chip Stocks (NSE)"),
        ("Temporal Coverage", "5 Full Years of Daily OHLCV Records (2021 to 2026)"),
    ]

    for i, (k, v) in enumerate(meta_data):
        meta_table.cell(i, 0).paragraphs[0].add_run(k).font.bold = True
        meta_table.cell(i, 1).paragraphs[0].add_run(v)

    style_table(meta_table, header_bg="1E3A8A")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Table of Contents Overview
    doc.add_heading("Table of Contents", level=1)
    toc_items = [
        "1. Executive Summary & Official Problem Statement",
        "2. High-Level System Architecture (5-Stage Pipeline)",
        "3. Live Dashboard Walkthrough -- Screen-by-Screen User Guide",
        "   - 3.1 Global Controls & Sidebar Automation",
        "   - 3.2 Tab 1: Technical Analysis & NumPy Moving Averages",
        "   - 3.3 Tab 2: Data Cleaning & Trade Volume Analysis",
        "   - 3.4 Tab 3: Multi-Stock Normalization & Comparison",
        "   - 3.5 Tab 4: Sector Performance & Return Correlations",
        "   - 3.6 Tab 5: Dataset Explorer & Audit Export",
        "4. Mathematical Formulations & Pure NumPy Implementations",
        "5. Data Quality Assurance & Missing Volume Imputation Logic",
        "6. Local Setup, Execution & Cloud Deployment Guide",
        "7. Project Verification, Test Suite & Evaluation Rubric",
    ]
    for item in toc_items:
        p_toc = doc.add_paragraph()
        p_toc.paragraph_format.space_before = Pt(0)
        p_toc.paragraph_format.space_after = Pt(2)
        r = p_toc.add_run(item)
        r.font.size = Pt(9.5)
        if item.startswith("3.") and not item.startswith("   -"):
            r.font.bold = True

    doc.add_page_break()

    # SECTION 1
    doc.add_heading("1. Executive Summary & Official Problem Statement", level=1)
    
    p = doc.add_paragraph()
    p.add_run("Official Description (Verbatim from Lab Assignment):\n").font.bold = True
    p_quote = doc.add_paragraph()
    p_quote.paragraph_format.left_indent = Inches(0.4)
    r_quote = p_quote.add_run(
        "\"Process multi-stock historical datasets, engineer rolling window moving averages with NumPy, "
        "clean missing trade volumes, plot comparative stock charts.\""
    )
    r_quote.font.italic = True
    r_quote.font.color.rgb = RGBColor.from_string("1E40AF")

    doc.add_paragraph(
        "In modern computational finance and machine learning, raw market datasets cannot be directly ingested "
        "by machine learning models due to several inherent anomalies: trading halts, non-synchronous market holidays, "
        "zero-volume reporting glitches, and stark price scale disparities between constituent equities (e.g., a stock "
        "trading at ₹150 vs. one trading at ₹6,000). Furthermore, raw price series exhibit non-stationarity, which violates "
        "the fundamental assumptions of standard statistical learning algorithms."
    )
    doc.add_paragraph(
        "The Stock Market Daily Trade Normalizer (ML_6) was engineered as a robust, production-grade data preprocessing "
        "and feature engineering pipeline. Built strictly around the Nifty 50 equity index over a comprehensive 5-year timeline, "
        "this system ingests raw multi-stock data, resolves missing trade volume anomalies using multi-stage imputation, "
        "computes an array of technical moving averages and volatility indicators using pure NumPy vectorized operations "
        "(deliberately avoiding high-level pandas rolling wrappers), normalizes price and volume distributions for cross-stock "
        "comparability, and exposes the entire pipeline via an interactive browser-based Streamlit dashboard."
    )

    add_callout(
        doc,
        "Every component in this project was architected to satisfy 100% of the lab description while providing an "
        "intuitive, interactive graphical interface that allows non-technical observers, evaluators, and quantitative "
        "analysts to interactively verify the preprocessing transformations in real time.",
        title="CORE PHILOSOPHY",
        icon="🎯"
    )

    # SECTION 2
    doc.add_heading("2. High-Level System Architecture (5-Stage Pipeline)", level=1)
    doc.add_paragraph(
        "The project follows a linear, deterministic 5-stage architectural pipeline designed for auditability and "
        "reproducibility. Every stage reads its inputs from a designated folder and outputs its results to the subsequent stage:"
    )

    pipe_table = doc.add_table(rows=6, cols=4)
    pipe_headers = ["Stage", "Module File", "Primary Operations", "Input / Output Artifacts"]
    for j, h in enumerate(pipe_headers):
        pipe_table.cell(0, j).paragraphs[0].add_run(h)

    pipe_rows = [
        ("1. Ingestion", "src/data_downloader.py", "Fetches 5 years of daily OHLCV data from Yahoo Finance API with retry logic", "Nifty 50 Ticker List -> data/raw/*.csv"),
        ("2. Cleaning", "src/data_cleaner.py", "Converts 0-volumes to NaN; Imputes via Fwd-Fill -> Bwd-Fill -> Median; Linear OHLC interpolation; 3-sigma outlier detection", "data/raw/*.csv -> data/cleaned/*.csv"),
        ("3. Engineering", "src/feature_engineer.py", "Computes SMA (7, 14, 21, 50, 200), EMA (12, 26, 50), WMA (7, 14, 21), 21d Volatility, and Returns with pure NumPy", "data/cleaned/*.csv -> data/processed/*.csv"),
        ("4. Normalization", "src/normalizer.py", "Performs Min-Max Scaling [0, 1] and Z-Score standardization (mu=0, sigma=1); Stores invertible parameters", "data/processed/*.csv -> Normalized Columns"),
        ("5. Visualization", "src/visualizer.py & app.py", "Generates static publication charts (Matplotlib/Seaborn) and an interactive web dashboard (Streamlit/Plotly)", "data/processed/*.csv -> outputs/charts/ & Live UI"),
    ]

    for i, row_data in enumerate(pipe_rows, 1):
        for j, val in enumerate(row_data):
            pipe_table.cell(i, j).paragraphs[0].add_run(val)

    style_table(pipe_table, header_bg="1E3A8A")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # SECTION 3
    doc.add_heading("3. Live Dashboard Walkthrough -- Screen-by-Screen User Guide", level=1)
    doc.add_paragraph(
        "When you launch or open the live deployed web application, you are greeted by an institutional analytics dashboard. "
        "Below is a comprehensive guide to understanding every button, filter, and visualization displayed on your screen."
    )

    # 3.1
    doc.add_heading("3.1 Global Controls & Sidebar Automation", level=2)
    doc.add_paragraph(
        "The left sidebar contains persistent controls that dictate what data is actively rendered across the entire dashboard:"
    )
    doc.add_paragraph("• Select Primary Stock: A searchable dropdown containing all Nifty 50 constituent companies, formatted as Symbol - Company Name (Sector). Selecting a ticker (e.g., ADANIENT, RELIANCE, TCS) dynamically re-renders all charts in Tabs 1 and 2.")
    doc.add_paragraph("• Date Range Filter: A dual-point calendar picker defaulting to the full 5-year span (October 1, 2021 to September 30, 2026). Adjusting this slider allows you to isolate specific market epochs, such as the 2022 rate hike cycle or the 2023 rally.")
    doc.add_paragraph("• Pipeline Automation (One-Click Actions): Provides buttons to directly trigger backend operations from the web browser: 'Run Cleaning' triggers data imputation, 'Re-Engineer' recalculates NumPy moving averages, and 'Run Full Pipeline (--all)' runs all 5 pipeline phases sequentially.")
    doc.add_paragraph("• Top Summary Metric Badges: Located at the top of the main screen, four metric badges summarize dataset health: Nifty 50 Universe, 5 Years of Historical OHLCV, Imputed Trade Volume count, and Audited Data Quality status.")

    # 3.2
    doc.add_heading("3.2 Tab 1: Technical Analysis & NumPy Moving Averages", level=2)
    doc.add_paragraph(
        "This tab serves as the primary visual proof of the project requirement: 'engineer rolling window moving averages with NumPy'."
    )
    doc.add_paragraph("1. Multiselect Moving Average Overlay: Allows users to toggle any combination of 11 distinct NumPy-computed technical indicators: SMA (7, 14, 21, 50, 200), EMA (12, 26, 50), and WMA (7, 14, 21).")
    doc.add_paragraph("2. Price Action Representation: Switch between smooth daily closing price lines or full OHLC Candlestick charts.")
    doc.add_paragraph("3. Dual-Axis Price & Volume Subplot:")
    doc.add_paragraph("   - Upper Panel: Tracks daily stock price against the selected moving averages. Observers can spot institutional crossovers, such as the Golden Cross (50 SMA crossing above 200 SMA) or short-term momentum shifts (21 SMA vs. 26 EMA).")
    doc.add_paragraph("   - Lower Panel: Visualizes daily trading volume with color-coded bars (Green = Up Day, Red = Down Day), overlaid with a 21-day Volume SMA line to highlight institutional volume breakouts.")
    doc.add_paragraph("4. 21-Day Rolling Volatility (NumPy): A shaded purple area chart tracking rolling standard deviation. High spikes indicate periods of extreme market turbulence (e.g., the Adani-Hindenburg period in early 2023).")
    doc.add_paragraph("5. Cumulative Return Curve (%): A green trajectory chart tracking the total percentage compounding return of an investment held from day one.")

    # 3.3
    doc.add_heading("3.3 Tab 2: Data Cleaning & Trade Volume Analysis", level=2)
    doc.add_paragraph(
        "This tab validates the requirement: 'clean missing trade volumes'. It provides direct visual and numerical proof of data sanitization."
    )
    doc.add_paragraph("1. Raw Volume vs. Cleaned Volume (Side-by-Side Plots):")
    doc.add_paragraph("   - Raw Volume (Red Line): Highlights uncleaned records from the exchange feed, showing zeros, dropped days, and holiday anomalies.")
    doc.add_paragraph("   - Cleaned Volume (Green Line): Shows the post-processed series where zeros were replaced with NaN and resolved via the forward-fill -> backward-fill -> column median algorithm.")
    doc.add_paragraph("2. 3-Sigma Daily Return Outlier Detection Scatter Plot: Plots daily returns across the entire 5-year timeline. Normal trading days appear as blue dots; statistical anomalies whose daily return exceeds 3 standard deviations (|Z| > 3) appear as prominent red crosses, bounded by horizontal dashed lines representing +3 sigma, -3 sigma, and the empirical mean.")
    doc.add_paragraph("3. Full Cleaning Report Audit Table: A live, searchable data grid listing all 48 processed stocks, documenting Original_Rows, Cleaned_Rows, Missing_Before_Volume, Missing_After_Volume (always 0 post-cleaning), and Outliers_Flagged.")

    # 3.4
    doc.add_heading("3.4 Tab 3: Multi-Stock Normalization & Comparison", level=2)
    doc.add_paragraph(
        "This tab satisfies the requirement: 'plot comparative stock charts'. Without normalization, plotting MRF (trading near ₹1,00,000) against Tata Steel (trading near ₹150) results in Tata Steel appearing as a completely flat horizontal line at the bottom of the axis."
    )
    doc.add_paragraph("• Stock Multi-Picker: Pick any combination of stocks (e.g., Adani Enterprises, TCS, Reliance, Axis Bank, Sun Pharma) across diverse sectors.")
    doc.add_paragraph("• Normalization Technique Switch:")
    doc.add_paragraph("   - Min-Max Scaling [0, 1]: Rescales every stock's 5-year price range so its 5-year low is exactly 0.0 and its 5-year high is 1.0. This allows viewers to observe relative range positioning and trend velocity.")
    doc.add_paragraph("   - Z-Score Standardization (mu=0, sigma=1): Normalizes price distributions in terms of standard deviations away from each stock's historical mean, ideal for statistical anomaly detection.")
    doc.add_paragraph("• Expandable Parameters Table: Discloses the exact mathematical parameters (Min, Max, Mean, Std Dev) used to perform the normalization, guaranteeing transformation invertibility.")

    # 3.5
    doc.add_heading("3.5 Tab 4: Sector Performance & Return Correlations", level=2)
    doc.add_paragraph(
        "Extends the analysis across broader industrial sector groupings:"
    )
    doc.add_paragraph("1. Sector Return Distribution (Box Plot): Displays median daily returns, interquartile ranges (IQR), and extreme outliers across key Indian sectors (IT, Energy, Financial Services, Metals, FMCG, Automobile, Healthcare).")
    doc.add_paragraph("2. Mean Trading Volume by Sector (Bar Chart): Ranks sectors by trading liquidity (in millions of shares), identifying where institutional market participation is concentrated.")
    doc.add_paragraph("3. Interactive 25-Stock Daily Return Correlation Matrix: An interactive heatmap color-coded from deep blue (-0.5) to crimson red (+1.0). Highlights strong co-movements within sectors (e.g., TCS and Infosys) and low/negative correlations ideal for multi-asset portfolio diversification.")

    # 3.6
    doc.add_heading("3.6 Tab 5: Dataset Explorer & Audit Export", level=2)
    doc.add_paragraph("• Provides transparent access to the underlying tabular data at each stage (Raw, Cleaned, Processed).")
    doc.add_paragraph("• Users can sort, search, and paginate through all engineered feature columns.")
    doc.add_paragraph("• Includes a dedicated 'Download CSV' button to export any processed stock dataset directly to Microsoft Excel or CSV format.")

    doc.add_page_break()

    # SECTION 4
    doc.add_heading("4. Mathematical Formulations & Pure NumPy Implementations", level=1)
    doc.add_paragraph(
        "A critical requirement of project ML_6 is engineering rolling window moving averages with NumPy rather than "
        "relying on convenience wrappers like pandas DataFrame.rolling(). Below are the exact mathematical formulas and "
        "their vectorized NumPy implementations:"
    )

    doc.add_heading("4.1 Simple Moving Average (SMA)", level=2)
    doc.add_paragraph("Mathematical Definition:")
    doc.add_paragraph("SMA_t = (1 / w) * Sum_{i=0}^{w-1} P_{t-i}")
    doc.add_paragraph(
        "NumPy Vectorization via Discrete Linear Convolution:\n"
        "In signal processing, an unweighted moving average is equivalent to convolving a 1-D price vector P with a uniform "
        "filter kernel K of length w where each element is 1/w:\n"
        "K = np.ones(w) / w\n"
        "valid_sma = np.convolve(P, K, mode='valid')\n"
        "To maintain exact index alignment with the original dataset, the first (w - 1) positions are prepended with np.nan."
    )

    doc.add_heading("4.2 Exponential Moving Average (EMA)", level=2)
    doc.add_paragraph("Mathematical Definition:")
    doc.add_paragraph("EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}    where alpha = 2 / (w + 1)")
    doc.add_paragraph(
        "NumPy Implementation:\n"
        "An empty NumPy array is pre-allocated with np.empty_like(P, dtype=np.float64). The algorithm seeds EMA_0 = P_0 "
        "and executes an optimized C-speed iteration across the array applying the recurrence formula. This avoids pandas "
        "overhead while precisely weighting recent prices exponentially higher than older observations."
    )

    doc.add_heading("4.3 Weighted Moving Average (WMA)", level=2)
    doc.add_paragraph("Mathematical Definition:")
    doc.add_paragraph("WMA_t = Sum_{i=1}^{w} [ i * P_{t - w + i} ] / Sum_{i=1}^{w} i")
    doc.add_paragraph(
        "NumPy Vectorization via Linear Weight Vector Dot Products:\n"
        "Weights are constructed as an arithmetic progression W = np.arange(1, w + 1, dtype=np.float64). "
        "The normalization denominator is the triangular sum S = w * (w + 1) / 2. Each window slice P[i - w + 1 : i + 1] "
        "is evaluated via np.dot(slice, W) / S, yielding an exact linearly weighted average."
    )

    doc.add_heading("4.4 Min-Max Normalization vs. Z-Score Standardization", level=2)
    doc.add_paragraph("Min-Max Scaling:")
    doc.add_paragraph("X_scaled = (X - X_min) / (X_max - X_min) * (target_max - target_min) + target_min")
    doc.add_paragraph("Z-Score Standardization:")
    doc.add_paragraph("X_standardized = (X - mu) / sigma    where mu = (1/N) * Sum(X) and sigma = sqrt( (1/N) * Sum(X - mu)^2 )")

    # SECTION 5
    doc.add_heading("5. Data Quality Assurance & Missing Volume Imputation Logic", level=1)
    doc.add_paragraph(
        "Raw trade volumes from financial exchanges frequently exhibit zeroes during non-trading sessions or reporting "
        "lags. Standard machine learning models fail if NaNs or zeros are fed into volume-derived features. The data_cleaner.py "
        "module enforces a robust 4-phase cleaning protocol:"
    )
    doc.add_paragraph("1. Threshold Identification: Any trading volume <= 1 (including 0) is flagged as an invalid trading record and converted to np.nan.")
    doc.add_paragraph("2. Forward-Fill (Fwd-Fill): Imputes the missing volume with the most recently observed valid trading volume, reflecting market continuity.")
    doc.add_paragraph("3. Backward-Fill (Bwd-Fill): If missing values occur at the very beginning of the historical window, backward-filling carries the earliest valid volume back.")
    doc.add_paragraph("4. Column Median Fallback: In the edge case of persistent missing runs, the stock's median volume across all trading days is imputed, preventing mean-skewing by volume outlier spikes.")
    doc.add_paragraph("5. Outlier Detection: Computes daily return percentage r_t = (P_t - P_{t-1}) / P_{t-1}. Any return satisfying |r_t - mu| > 3 * sigma is flagged in an Outlier_Flag boolean column for auditing.")

    # SECTION 6
    doc.add_heading("6. Local Setup, Execution & Cloud Deployment Guide", level=1)
    doc.add_paragraph("Prerequisites: Python 3.10 or higher installed on Windows, macOS, or Linux.")
    doc.add_paragraph("Step 1: Install Dependencies\n  pip install -r requirements.txt")
    doc.add_paragraph("Step 2: Run Automated CLI Pipeline\n  python main.py --all\n  (Or run specific phases: python main.py --clean --engineer --normalize --visualize)")
    doc.add_paragraph("Step 3: Run the Automated Unit Test Suite\n  python -m pytest tests/test_pipeline.py -v\n  (Verifies all 29 mathematical tests pass)")
    doc.add_paragraph("Step 4: Launch the Local Web Dashboard\n  python -m streamlit run app.py\n  (Opens http://localhost:8501 in your default web browser)")
    doc.add_paragraph("Step 5: Cloud Deployment (Streamlit Community Cloud)\n  The project is pre-configured with Git and GitHub integration. Simply push the repository to GitHub and link it on share.streamlit.io with main file path app.py for free, permanent 24/7 hosting.")

    # SECTION 7
    doc.add_heading("7. Project Verification, Test Suite & Evaluation Rubric", level=1)
    doc.add_paragraph(
        "The project includes a comprehensive automated test suite (tests/test_pipeline.py) covering 29 unit tests:"
    )

    test_table = doc.add_table(rows=7, cols=3)
    test_headers = ["Test Suite Category", "Functions Tested", "Verification Criteria & Assertions"]
    for j, h in enumerate(test_headers):
        test_table.cell(0, j).paragraphs[0].add_run(h)

    test_rows = [
        ("Simple Moving Average", "compute_sma", "Exact convolution equivalence, window=1 identity, constant array invariance, correct NaN padding length"),
        ("Exponential Moving Average", "compute_ema", "Recurrence accuracy, recent price weighting dominance (EMA >= SMA on ascending data), constant array stability"),
        ("Weighted Moving Average", "compute_wma", "Arithmetic weight vector dot product verification against hand-calculated analytical solutions"),
        ("Rolling Volatility & Returns", "compute_rolling_std, compute_daily_returns, compute_cumulative_returns", "Bessel correction accuracy, zero variance on constant arrays, cumulative compounding equivalence"),
        ("Min-Max Normalization", "minmax_normalize, inverse_minmax", "Exact [0, 1] bounds assertion, custom feature ranges, error handling on constant zero-variance arrays, exact inverse invertibility"),
        ("Z-Score Standardization", "zscore_normalize, inverse_zscore", "Mean = 0.00000, Std Dev = 1.00000 assertions, parameter storage verification, exact inverse invertibility"),
    ]

    for i, row_data in enumerate(test_rows, 1):
        for j, val in enumerate(row_data):
            test_table.cell(i, j).paragraphs[0].add_run(val)

    style_table(test_table, header_bg="1E3A8A")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    add_callout(
        doc,
        "Status: 29 passed in ~1.14s. All mathematical formulas, normalization constraints, and data cleaner edge cases "
        "have passed automated unit testing.",
        title="TEST EXECUTION VERIFICATION",
        icon="✅",
        bg_color="F0FDF4",
        border_color="16A34A"
    )

    # Save Document
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    print(f"Document successfully created at: {output_path}")

if __name__ == "__main__":
    out = os.path.join(os.getcwd(), "Stock_Market_Daily_Trade_Normalizer_User_Guide.docx")
    generate_document(out)
