"""
test_pipeline.py -- Unit Tests for the Trade Normalizer Pipeline
================================================================

Tests cover:
    - NumPy moving average computations (SMA, EMA, WMA)
    - Data cleaning logic
    - Normalization correctness and invertibility
    - Utility functions

Run with:
    pytest tests/test_pipeline.py -v
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineer import (
    compute_sma,
    compute_ema,
    compute_wma,
    compute_rolling_std,
    compute_daily_returns,
    compute_cumulative_returns,
)
from src.normalizer import (
    minmax_normalize,
    zscore_normalize,
    inverse_minmax,
    inverse_zscore,
)
from src.utils import profile_dataframe, get_ticker_from_filename


# ==============================================================================
# Test Data Fixtures
# ==============================================================================
@pytest.fixture
def simple_series():
    """A simple ascending series for testing."""
    return np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])


@pytest.fixture
def constant_series():
    """A constant series -- edge case."""
    return np.array([50.0] * 10)


@pytest.fixture
def series_with_nans():
    """A series containing NaN values."""
    return np.array([10.0, np.nan, 30.0, 40.0, np.nan, 60.0, 70.0, 80.0, 90.0, 100.0])


@pytest.fixture
def sample_dataframe():
    """A sample stock DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    df = pd.DataFrame(
        {
            "Open": close + np.random.randn(100),
            "High": close + abs(np.random.randn(100)) * 2,
            "Low": close - abs(np.random.randn(100)) * 2,
            "Close": close,
            "Adj Close": close,
            "Volume": np.random.randint(100000, 10000000, size=100).astype(float),
        },
        index=dates,
    )
    return df


# ==============================================================================
# Tests: Simple Moving Average (SMA)
# ==============================================================================
class TestSMA:
    """Tests for compute_sma()."""

    def test_sma_basic(self, simple_series):
        """SMA of [10, 20, 30, 40, 50] with window=3 should give known values."""
        result = compute_sma(simple_series, window=3)
        assert len(result) == len(simple_series)
        # First 2 values should be NaN (window-1)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # SMA at index 2 = (10+20+30)/3 = 20
        np.testing.assert_almost_equal(result[2], 20.0)
        # SMA at index 3 = (20+30+40)/3 = 30
        np.testing.assert_almost_equal(result[3], 30.0)

    def test_sma_window_1(self, simple_series):
        """SMA with window=1 should return the original series."""
        result = compute_sma(simple_series, window=1)
        np.testing.assert_array_almost_equal(result, simple_series)

    def test_sma_full_window(self, simple_series):
        """SMA with window = len(series) should have only last value non-NaN."""
        result = compute_sma(simple_series, window=len(simple_series))
        assert np.isnan(result[:-1]).all()
        np.testing.assert_almost_equal(result[-1], simple_series.mean())

    def test_sma_constant(self, constant_series):
        """SMA of a constant series should equal that constant."""
        result = compute_sma(constant_series, window=3)
        valid = result[~np.isnan(result)]
        np.testing.assert_array_almost_equal(valid, np.full_like(valid, 50.0))


# ==============================================================================
# Tests: Exponential Moving Average (EMA)
# ==============================================================================
class TestEMA:
    """Tests for compute_ema()."""

    def test_ema_basic(self, simple_series):
        """EMA should return array of same length with no NaN except possibly first."""
        result = compute_ema(simple_series, window=3)
        assert len(result) == len(simple_series)
        # EMA should exist for all values (first value is seeded)
        assert not np.isnan(result[-1])

    def test_ema_responds_to_recent(self, simple_series):
        """EMA should weight recent values more -- last EMA > SMA for ascending."""
        ema = compute_ema(simple_series, window=3)
        sma = compute_sma(simple_series, window=3)
        # For an ascending series, EMA should be >= SMA at the end
        assert ema[-1] >= sma[-1]

    def test_ema_constant(self, constant_series):
        """EMA of a constant series should equal that constant."""
        result = compute_ema(constant_series, window=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_array_almost_equal(valid, np.full_like(valid, 50.0))


# ==============================================================================
# Tests: Weighted Moving Average (WMA)
# ==============================================================================
class TestWMA:
    """Tests for compute_wma()."""

    def test_wma_basic(self, simple_series):
        """WMA should return correct length and have NaN padding."""
        result = compute_wma(simple_series, window=3)
        assert len(result) == len(simple_series)
        # First (window-1) values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])

    def test_wma_manual_calculation(self):
        """Verify WMA against hand calculation."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = compute_wma(data, window=3)
        # WMA at index 2: (1*10 + 2*20 + 3*30) / (1+2+3) = (10+40+90)/6 = 23.33...
        expected = (1 * 10 + 2 * 20 + 3 * 30) / 6.0
        np.testing.assert_almost_equal(result[2], expected, decimal=2)

    def test_wma_constant(self, constant_series):
        """WMA of a constant series should equal that constant."""
        result = compute_wma(constant_series, window=4)
        valid = result[~np.isnan(result)]
        np.testing.assert_array_almost_equal(valid, np.full_like(valid, 50.0))


# ==============================================================================
# Tests: Rolling Standard Deviation
# ==============================================================================
class TestRollingStd:
    """Tests for compute_rolling_std()."""

    def test_rolling_std_constant(self, constant_series):
        """Rolling std of a constant series should be 0."""
        result = compute_rolling_std(constant_series, window=3)
        valid = result[~np.isnan(result)]
        np.testing.assert_array_almost_equal(valid, np.zeros_like(valid))

    def test_rolling_std_positive(self, simple_series):
        """Rolling std of a non-constant series should be positive."""
        result = compute_rolling_std(simple_series, window=3)
        valid = result[~np.isnan(result)]
        assert (valid > 0).all()

    def test_rolling_std_length(self, simple_series):
        """Output length should match input length."""
        result = compute_rolling_std(simple_series, window=5)
        assert len(result) == len(simple_series)


# ==============================================================================
# Tests: Daily Returns
# ==============================================================================
class TestDailyReturns:
    """Tests for compute_daily_returns()."""

    def test_daily_returns_basic(self):
        """Known daily returns calculation."""
        data = np.array([100.0, 110.0, 99.0, 115.0])
        result = compute_daily_returns(data)
        assert len(result) == len(data)
        assert np.isnan(result[0])  # First value should be NaN
        np.testing.assert_almost_equal(result[1], 0.10)  # 10% increase
        np.testing.assert_almost_equal(result[2], -0.1)  # 10% decrease
        np.testing.assert_almost_equal(result[3], 115.0 / 99.0 - 1)

    def test_daily_returns_constant(self, constant_series):
        """Daily returns of constant series should be 0."""
        result = compute_daily_returns(constant_series)
        valid = result[~np.isnan(result)]
        np.testing.assert_array_almost_equal(valid, np.zeros_like(valid))


# ==============================================================================
# Tests: Cumulative Returns
# ==============================================================================
class TestCumulativeReturns:
    """Tests for compute_cumulative_returns()."""

    def test_cumulative_returns_basic(self):
        """Cumulative return at end should equal total return."""
        prices = np.array([100.0, 110.0, 121.0])  # +10%, +10%
        daily_ret = compute_daily_returns(prices)   # [NaN, 0.10, 0.10]
        result = compute_cumulative_returns(daily_ret)
        # cumprod([1.10, 1.10]) = [1.10, 1.21]  -> subtract 1 -> 0.21 (21%)
        np.testing.assert_almost_equal(result[-1] - 1, 0.21, decimal=2)

    def test_cumulative_returns_length(self, simple_series):
        """Output length should match input length."""
        result = compute_cumulative_returns(simple_series)
        assert len(result) == len(simple_series)


# ==============================================================================
# Tests: Min-Max Normalization
# ==============================================================================
class TestMinMaxNormalization:
    """Tests for minmax_normalize() and inverse_minmax()."""

    def test_minmax_range(self, simple_series):
        """Normalized values should be in [0, 1]."""
        result, params = minmax_normalize(simple_series)
        assert np.nanmin(result) >= 0.0
        assert np.nanmax(result) <= 1.0

    def test_minmax_custom_range(self, simple_series):
        """Normalized values with custom range should respect that range."""
        result, params = minmax_normalize(simple_series, feature_range=(-1, 1))
        np.testing.assert_almost_equal(np.nanmin(result), -1.0)
        np.testing.assert_almost_equal(np.nanmax(result), 1.0)

    def test_minmax_inverse(self, simple_series):
        """Inverse should recover the original data."""
        normalized, params = minmax_normalize(simple_series)
        recovered = inverse_minmax(normalized, params)
        np.testing.assert_array_almost_equal(recovered, simple_series)

    def test_minmax_constant(self, constant_series):
        """Min-Max of constant series should raise ValueError (zero range)."""
        with pytest.raises(ValueError, match="constant data"):
            minmax_normalize(constant_series)


# ==============================================================================
# Tests: Z-Score Normalization
# ==============================================================================
class TestZScoreNormalization:
    """Tests for zscore_normalize() and inverse_zscore()."""

    def test_zscore_mean_std(self, simple_series):
        """Z-score normalized data should have mean~=0 and std~=1."""
        result, params = zscore_normalize(simple_series)
        np.testing.assert_almost_equal(np.nanmean(result), 0.0, decimal=5)
        np.testing.assert_almost_equal(np.nanstd(result, ddof=0), 1.0, decimal=5)

    def test_zscore_inverse(self, simple_series):
        """Inverse should recover the original data."""
        normalized, params = zscore_normalize(simple_series)
        recovered = inverse_zscore(normalized, params)
        np.testing.assert_array_almost_equal(recovered, simple_series)

    def test_zscore_params(self, simple_series):
        """Params should contain mean and std."""
        _, params = zscore_normalize(simple_series)
        assert "mean" in params
        assert "std" in params
        np.testing.assert_almost_equal(params["mean"], simple_series.mean())


# ==============================================================================
# Tests: Utility Functions
# ==============================================================================
class TestUtils:
    """Tests for utility functions."""

    def test_get_ticker_from_filename(self):
        """Should extract ticker from a CSV filename."""
        assert get_ticker_from_filename("data/raw/TCS.NS.csv") == "TCS.NS"
        assert get_ticker_from_filename("RELIANCE.NS.csv") == "RELIANCE.NS"

    def test_profile_dataframe(self, sample_dataframe):
        """Profile should return valid structure."""
        profile = profile_dataframe(sample_dataframe, name="test")
        assert profile["name"] == "test"
        assert profile["shape"] == (100, 6)
        assert "Close" in profile["columns"]
        assert isinstance(profile["missing_counts"], dict)
        assert isinstance(profile["numeric_summary"], dict)

    def test_profile_missing_counts(self):
        """Profile should correctly count missing values."""
        df = pd.DataFrame({"A": [1, np.nan, 3], "B": [np.nan, np.nan, 6]})
        profile = profile_dataframe(df)
        assert profile["missing_counts"]["A"] == 1
        assert profile["missing_counts"]["B"] == 2


# ==============================================================================
# Tests: Integration -- Feature Engineering on a DataFrame
# ==============================================================================
class TestFeatureEngineering:
    """Integration tests for feature engineering on realistic data."""

    def test_sma_on_stock_data(self, sample_dataframe):
        """SMA should work on realistic stock close prices."""
        close = sample_dataframe["Close"].values
        for window in [7, 14, 21, 50]:
            result = compute_sma(close, window)
            assert len(result) == len(close)
            # First (window-1) should be NaN
            assert np.isnan(result[: window - 1]).all()
            # Rest should not be NaN
            assert not np.isnan(result[window - 1 :]).any()

    def test_all_features_no_crash(self, sample_dataframe):
        """All feature functions should run without errors on realistic data."""
        close = sample_dataframe["Close"].values
        volume = sample_dataframe["Volume"].values

        sma = compute_sma(close, 7)
        ema = compute_ema(close, 12)
        wma = compute_wma(close, 7)
        std = compute_rolling_std(close, 21)
        ret = compute_daily_returns(close)
        cum = compute_cumulative_returns(close)

        # All should have correct length
        for arr in [sma, ema, wma, std, ret, cum]:
            assert len(arr) == len(close)


# ==============================================================================
# Run Tests
# ==============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
