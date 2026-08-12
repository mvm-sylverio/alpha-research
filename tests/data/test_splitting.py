import pytest
import warnings
import pandas as pd
import polars as pl
from alpha_research.data.splitting import research_test_split


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def large_df_pandas():
    """
    5 years of daily data, single asset.
    ~1260 unique dates, ~60 months.
    Easily satisfies all defaults.
    """
    dates = pd.date_range('2019-01-01', '2023-12-31', freq='B')
    return pd.DataFrame({'time': dates, 'symbol': 'AAPL', 'close': 1.0})

@pytest.fixture
def large_df_polars(large_df_pandas):
    return pl.from_pandas(large_df_pandas)

@pytest.fixture
def multi_asset_df_pandas():
    """
    5 years, 3 assets — same dates, multiple rows per date.
    Tests that split is consistent across assets.
    """
    dates = pd.date_range('2019-01-01', '2023-12-31', freq='B')
    dfs = [pd.DataFrame({'time': dates, 'symbol': s, 'close': 1.0})
           for s in ['AAPL', 'MSFT', 'GOOGL']]
    return pd.concat(dfs, ignore_index=True)

@pytest.fixture
def too_small_df_pandas():
    """
    Only 3 months of data — insufficient for test block minimum (6 months).
    """
    dates = pd.date_range('2023-01-01', '2023-03-31', freq='B')
    return pd.DataFrame({'time': dates, 'symbol': 'AAPL', 'close': 1.0})

@pytest.fixture
def borderline_research_df_pandas():
    """
    ~26 months total. After purge zone (7.2 months), research gets ~18.8 months.
    Below 24 month minimum but above 12 month half-minimum → expect UserWarning.
    """
    dates = pd.date_range('2021-11-01', '2023-12-31', freq='B')
    return pd.DataFrame({'time': dates, 'symbol': 'AAPL', 'close': 1.0})

@pytest.fixture
def critical_research_df_pandas():
    """
    ~16 months total. After purge zone (7.2 months), research gets ~8.8 months.
    Below 12 month half-minimum → expect ValueError.
    """
    dates = pd.date_range('2022-09-01', '2023-12-31', freq='B')
    return pd.DataFrame({'time': dates, 'symbol': 'AAPL', 'close': 1.0})


# ------------------------------------------------------
# research_test_split
# ------------------------------------------------------
# raises
def test_split_purging_coefficient_below_one_raises(large_df_pandas):
    """Should raise ValueError when purging_coefficient < 1."""
    with pytest.raises(ValueError, match="purging_coefficient"):
        research_test_split(large_df_pandas, purging_coefficient=0.9)

def test_split_insufficient_data_for_test_raises(too_small_df_pandas):
    """Should raise ValueError when total data cannot satisfy test minimum."""
    with pytest.raises(ValueError, match="Insufficient data"):
        research_test_split(too_small_df_pandas)

def test_split_research_below_half_minimum_raises(critical_research_df_pandas):
    """Should raise ValueError when research block is below half minimum after purging."""
    with pytest.raises(ValueError, match="Research block too small"):
        research_test_split(
            critical_research_df_pandas,
            min_research_period=24,
            min_test_period=6,
            min_research_unique_pts=500,
            min_test_unique_pts=120,
        )

# warnings
def test_split_research_below_minimum_warns(borderline_research_df_pandas):
    """Should warn when research block is below recommended minimum but above half."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        research_test_split(
            borderline_research_df_pandas,
            min_research_period=24,
            min_test_period=6,
            min_research_unique_pts=500,
            min_test_unique_pts=120,
        )
        assert any(issubclass(x.category, UserWarning) for x in w)
        assert any('smaller than recommended' in str(x.message) for x in w)

# output structure
def test_split_returns_tuple(large_df_pandas):
    """Should return a tuple of two DataFrames."""
    result = research_test_split(large_df_pandas)
    assert isinstance(result, tuple)
    assert len(result) == 2

def test_split_research_is_before_test(large_df_pandas):
    """Research block must end before test block starts."""
    df_research, df_test = research_test_split(large_df_pandas)
    assert df_research['time'].max() < df_test['time'].min()

def test_split_no_date_overlap(large_df_pandas):
    """No date should appear in both blocks."""
    df_research, df_test = research_test_split(large_df_pandas)
    research_dates = set(df_research['time'].unique())
    test_dates = set(df_test['time'].unique())
    assert len(research_dates & test_dates) == 0

def test_split_preserves_all_columns(large_df_pandas):
    """Both returned DataFrames should preserve all original columns."""
    df_research, df_test = research_test_split(large_df_pandas)
    assert set(df_research.columns) == set(large_df_pandas.columns)
    assert set(df_test.columns) == set(large_df_pandas.columns)

def test_split_research_larger_than_test(large_df_pandas):
    """Research block should be larger than test block."""
    df_research, df_test = research_test_split(large_df_pandas)
    assert len(df_research['time'].unique()) > len(df_test['time'].unique())

# purge zone
def test_split_purge_zone_exists(large_df_pandas):
    """There should be a gap between research end and test start."""
    df_research, df_test = research_test_split(large_df_pandas, purging_coefficient=1.2)
    research_end = df_research['time'].max()
    test_start = df_test['time'].min()
    assert test_start > research_end

def test_split_no_purge_zone_when_coefficient_is_one(large_df_pandas):
    """With purging_coefficient=1.0, research end and test start should be adjacent."""
    df_research, df_test = research_test_split(large_df_pandas, purging_coefficient=1.0)
    research_end = df_research['time'].max()
    test_start = df_test['time'].min()
    # no dates in between — difference should be minimal (next business day)
    assert (test_start - research_end).days <= 5

# multi asset
def test_split_consistent_across_assets(multi_asset_df_pandas):
    """All assets should have the same cutoff date in both blocks."""
    df_research, df_test = research_test_split(multi_asset_df_pandas)
    assert df_research['time'].max() == df_research.groupby('symbol')['time'].max().max()
    assert df_test['time'].min() == df_test.groupby('symbol')['time'].min().min()

# pandas / polars consistency
def test_split_pandas_polars_consistency(large_df_pandas, large_df_polars):
    """Should return identical cutoff dates for pandas and polars input."""
    res_pd, test_pd = research_test_split(large_df_pandas)
    res_pl, test_pl = research_test_split(large_df_polars)

    assert res_pd['time'].max().to_pydatetime() == res_pl['time'].max()
    assert test_pd['time'].min().to_pydatetime() == test_pl['time'].min()

def test_split_returns_pandas_for_pandas_input(large_df_pandas):
    """Should return pd.DataFrames when input is pandas."""
    df_research, df_test = research_test_split(large_df_pandas)
    assert isinstance(df_research, pd.DataFrame)
    assert isinstance(df_test, pd.DataFrame)

def test_split_returns_polars_for_polars_input(large_df_polars):
    """Should return pl.DataFrames when input is polars."""
    df_research, df_test = research_test_split(large_df_polars)
    assert isinstance(df_research, pl.DataFrame)
    assert isinstance(df_test, pl.DataFrame)


# ── extra columns ─────────────────────────────────────────────
def test_split_works_with_extra_columns(large_df_pandas):
    """Should split correctly regardless of extra columns in DataFrame."""
    df = large_df_pandas.copy()
    df['feature_a'] = 0.5
    df['feature_b'] = 1.0
    df_research, df_test = research_test_split(df)
    assert 'feature_a' in df_research.columns
    assert 'feature_b' in df_test.columns
