#!/usr/bin/env python3
"""
main.py -- Stock Market Daily Trade Normalizer Pipeline
========================================================

CLI entry point that orchestrates the full data preprocessing pipeline.

Usage:
    python main.py --all                  # Run the complete pipeline
    python main.py --download             # Download data only
    python main.py --clean                # Clean data only
    python main.py --engineer             # Feature engineering only
    python main.py --normalize            # Normalize data only
    python main.py --visualize            # Generate charts only
    python main.py --download --clean     # Combine steps as needed
    python main.py --all --force          # Force re-download of data

Project: ML_6 -- Stock Market Daily Trade Normalizer
Mapping: Experiment 1, 2 -- Data Preprocessing
"""

import argparse
import sys
import time

from src import config
from src.utils import get_logger, timer

logger = get_logger("pipeline")


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with boolean flags for each pipeline step.
    """
    parser = argparse.ArgumentParser(
        description="Stock Market Daily Trade Normalizer -- ML_6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                Run full pipeline
  python main.py --all --force        Re-download and reprocess everything
  python main.py --download --clean   Download and clean only
  python main.py --visualize          Generate charts from existing data
        """,
    )

    parser.add_argument(
        "--all", action="store_true",
        help="Run the complete pipeline (download -> clean -> engineer -> normalize -> visualize)"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Step 1: Download raw stock data from Yahoo Finance"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Step 2: Clean missing values and handle outliers"
    )
    parser.add_argument(
        "--engineer", action="store_true",
        help="Step 3: Engineer rolling window features (SMA, EMA, WMA)"
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Step 4: Normalize data for cross-stock comparison"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Step 5: Generate comparative stock charts"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download of data even if files exist"
    )
    parser.add_argument(
        "--norm-method", type=str, default="minmax",
        choices=["minmax", "zscore"],
        help="Normalization method (default: minmax)"
    )

    args = parser.parse_args()

    # If no step is specified, show help
    if not any([args.all, args.download, args.clean, args.engineer,
                args.normalize, args.visualize]):
        parser.print_help()
        sys.exit(0)

    return args


@timer
def run_pipeline(args):
    """
    Execute the data preprocessing pipeline based on CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    config.ensure_directories()

    steps_run = []

    # -- Step 1: Download --------------------------------------------------
    if args.all or args.download:
        logger.info("=" * 60)
        logger.info("STEP 1/5: Downloading stock data from Yahoo Finance")
        logger.info("=" * 60)
        from src.data_downloader import download_all_stocks, generate_download_summary

        results = download_all_stocks(force=args.force)
        generate_download_summary(results)
        steps_run.append("download")
        logger.info("")

    # -- Step 2: Clean -----------------------------------------------------
    if args.all or args.clean:
        logger.info("=" * 60)
        logger.info("STEP 2/5: Cleaning missing values and handling outliers")
        logger.info("=" * 60)
        from src.data_cleaner import clean_all_stocks

        clean_all_stocks()
        steps_run.append("clean")
        logger.info("")

    # -- Step 3: Feature Engineering ---------------------------------------
    if args.all or args.engineer:
        logger.info("=" * 60)
        logger.info("STEP 3/5: Engineering rolling window features (NumPy)")
        logger.info("=" * 60)
        from src.feature_engineer import engineer_all_stocks

        engineer_all_stocks()
        steps_run.append("engineer")
        logger.info("")

    # -- Step 4: Normalize -------------------------------------------------
    if args.all or args.normalize:
        logger.info("=" * 60)
        logger.info("STEP 4/5: Normalizing data for comparison")
        logger.info("=" * 60)
        from src.normalizer import normalize_all_stocks

        normalize_all_stocks(method=args.norm_method)
        steps_run.append("normalize")
        logger.info("")

    # -- Step 5: Visualize -------------------------------------------------
    if args.all or args.visualize:
        logger.info("=" * 60)
        logger.info("STEP 5/5: Generating comparative stock charts")
        logger.info("=" * 60)
        from src.visualizer import generate_all_charts

        generate_all_charts()
        steps_run.append("visualize")
        logger.info("")

    return steps_run


def main():
    """Main entry point."""
    print()
    print("=" * 62)
    print("   Stock Market Daily Trade Normalizer (ML_6)")
    print("   Experiment 1, 2 -- Data Preprocessing")
    print("=" * 62)
    print()

    args = parse_args()

    logger.info(f"Pipeline started | Project: {config.PROJECT_ID}")
    logger.info(f"Data range: {config.START_DATE} -> {config.END_DATE}")
    logger.info(f"Force re-download: {args.force}")
    logger.info(f"Normalization method: {args.norm_method}")
    logger.info("")

    try:
        steps_run = run_pipeline(args)

        print()
        print("=" * 62)
        print("   PIPELINE COMPLETE")
        print("=" * 62)
        print(f"  Steps executed: {', '.join(steps_run)}")
        print(f"  Raw data:       {config.RAW_DATA_DIR}")
        print(f"  Cleaned data:   {config.CLEANED_DATA_DIR}")
        print(f"  Processed data: {config.PROCESSED_DATA_DIR}")
        print(f"  Charts:         {config.CHARTS_DIR}")
        print(f"  Reports:        {config.REPORTS_DIR}")
        print()

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
