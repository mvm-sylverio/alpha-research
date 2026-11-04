import pytest
import warnings
import pandas as pd
import polars as pl
from alpha_research.data.splitting import research_test_split
from alpha_research.data.splitting import _split_dates
import numpy as np


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
    assert research_end < test_start
    assert sorted(pd.concat([df_research, df_test])['time'].tolist()) == sorted(large_df_pandas['time'].tolist())

# multi asset
def test_split_consistent_across_assets(multi_asset_df_pandas):
    """All assets should have the same cutoff date in both blocks."""
    df_research, df_test = research_test_split(multi_asset_df_pandas)
    assert df_research.groupby('symbol')['time'].max().nunique() == 1
    assert df_test.groupby('symbol')['time'].min().nunique() == 1


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('months,research_ids,test_ids', [(0, [0, 1, 2, 3, 4], [7, 8, 9]),
                                                       (2, [0, 1, 2, 3, 4, 5], [7, 8, 9])])
def test_split_exact_irregular_calendar(backend, months, research_ids, test_ids):
    """Should retain exactly the hand-calculated dates under point and month cutoffs."""
    # Last date Aug 1; three test points begin Jun 1. Purging six points
    # ends research Mar 3. With two test months, four purged months end Apr 1.
    dates = pd.to_datetime(['2024-01-01', '2024-01-07', '2024-02-01', '2024-03-01',
                            '2024-03-03', '2024-04-01', '2024-05-01', '2024-06-01',
                            '2024-07-01', '2024-08-01'])
    frame = pd.DataFrame({'time': dates, 'id': range(10), 'value': [None] * 10})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    research, test = research_test_split(frame, purging_coefficient=2,
        min_research_period=0, min_test_period=months, min_research_unique_pts=1,
        min_test_unique_pts=3 if months == 0 else 2)
    assert research['id'].to_list() == research_ids
    assert test['id'].to_list() == test_ids


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_split_incomplete_assets_unsorted_duplicates_and_schema(backend):
    """Should apply one global cutoff while retaining row order, duplicates, and schema."""
    frame = pd.DataFrame({'time': pd.to_datetime(['2024-01-05', '2024-01-01', '2024-01-04',
                                                  '2024-01-02', '2024-01-03', '2024-01-01']),
                          'symbol': ['A', 'A', 'B', 'B', 'A', 'A'], 'id': range(6)})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    before = frame.copy(deep=True) if backend == 'pandas' else frame.clone()
    research, test = research_test_split(frame, purging_coefficient=1, min_research_period=0,
        min_test_period=0, min_research_unique_pts=1, min_test_unique_pts=2)
    assert research['id'].to_list() == [1, 3, 4, 5]
    assert test['id'].to_list() == [0, 2]
    assert list(research.columns) == list(frame.columns)
    if backend == 'pandas':
        pd.testing.assert_frame_equal(frame, before)
        assert research.dtypes.equals(frame.dtypes)
    else:
        assert frame.equals(before) and research.schema == frame.schema


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_split_purges_forward_label_information_intervals(backend):
    """Should exclude research labels that reach the first test observation."""
    frame = pd.DataFrame({'time': pd.date_range('2024-01-01', periods=8),
                          'label_end': pd.date_range('2024-01-03', periods=8), 'id': range(8)})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    research, test = research_test_split(frame, purging_coefficient=1, min_research_period=0,
        min_test_period=0, min_research_unique_pts=1, min_test_unique_pts=2, label_end_col='label_end')
    assert research['id'].to_list() == [0, 1, 2, 3]
    assert test['id'].to_list() == [6, 7]
    assert research['label_end'].max() < test['time'].min()


@pytest.mark.parametrize('name', ['min_research_period', 'min_test_period', 'min_research_unique_pts', 'min_test_unique_pts'])
@pytest.mark.parametrize('value', [-1, True, 1.5, np.nan, np.inf])
def test_split_rejects_invalid_minimum_parameters(large_df_pandas, name, value):
    """Should reject invalid minimum parameters before selecting cutoffs."""
    with pytest.raises(ValueError, match=name):
        research_test_split(large_df_pandas, **{name: value})


@pytest.mark.parametrize('value', [True, np.nan, np.inf, -np.inf, 1e308])
def test_split_rejects_invalid_or_oversized_purge(large_df_pandas, value):
    """Should reject nonfinite, boolean, and unrepresentable purge sizes clearly."""
    expected_error = TypeError if value is True else ValueError
    with pytest.raises(expected_error):
        research_test_split(large_df_pandas, purging_coefficient=value)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_split_date_helper_validation(backend):
    """Should reject absent, missing, numeric, and textual date keys."""
    make = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    for values in ([1, 2], ['2024-01-01', '2024-01-02']):
        with pytest.raises(ValueError, match='datetime'):
            _split_dates(make({'time': values}), 'time')
    frame = pd.DataFrame({'time': pd.to_datetime(['2024-01-01', None])})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    with pytest.raises(ValueError, match='missing'):
        _split_dates(frame, 'time')
    with pytest.raises(KeyError):
        _split_dates(frame, 'absent')


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('timezone', [None, 'UTC', 'America/New_York'])
def test_split_dst_and_timezone_preservation(backend, timezone):
    """Should split by observed dates across DST without changing timezone or order."""
    dates = pd.date_range('2024-03-08 12:00', periods=6, tz=timezone)
    frame = pd.DataFrame({'time': dates, 'id': range(6)})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    research, test = research_test_split(frame, purging_coefficient=1, min_research_period=0,
        min_test_period=0, min_research_unique_pts=1, min_test_unique_pts=2)
    assert research['id'].to_list() == [0, 1, 2, 3]
    assert test['id'].to_list() == [4, 5]
    if backend == 'pandas':
        assert research['time'].dtype == frame['time'].dtype
    else:
        assert research.schema == frame.schema


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_split_leap_year_month_end_exact_rows(backend):
    """Should place leap-day month-end observations in the test block exactly."""
    frame = pd.DataFrame({'time': pd.to_datetime(['2023-12-31', '2024-01-31', '2024-02-28',
                                                  '2024-02-29', '2024-03-31']), 'id': range(5)})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    research, test = research_test_split(frame, purging_coefficient=1, min_research_period=0,
        min_test_period=1, min_research_unique_pts=1, min_test_unique_pts=1)
    assert research['id'].to_list() == [0, 1, 2]
    assert test['id'].to_list() == [3, 4]


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('invalid', ['missing', 'backward', 'timezone'])
def test_split_rejects_invalid_label_intervals(backend, invalid):
    """Should reject unknown, backward, and timezone-inconsistent label endpoints."""
    frame = pd.DataFrame({'time': pd.date_range('2024-01-01', periods=5),
                          'end': pd.date_range('2024-01-02', periods=5)})
    if invalid == 'missing':
        frame.loc[1, 'end'] = pd.NaT
    elif invalid == 'backward':
        frame.loc[1, 'end'] = pd.Timestamp('2023-01-01')
    else:
        frame['end'] = frame['end'].dt.tz_localize('UTC')
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    with pytest.raises(ValueError):
        research_test_split(frame, min_research_period=0, min_test_period=0,
                            min_research_unique_pts=1, min_test_unique_pts=1, label_end_col='end')


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_split_rejects_zero_point_minima_and_excessive_purge(backend):
    """Should reject zero unique-date minima and purges exceeding available history."""
    make = pd.DataFrame({'time': pd.date_range('2024-01-01', periods=5)})
    frame = make if backend == 'pandas' else pl.from_pandas(make)
    for key in ('min_research_unique_pts', 'min_test_unique_pts'):
        with pytest.raises(ValueError, match=key):
            research_test_split(frame, **{key: 0})
    with pytest.raises(ValueError, match='Research block too small'):
        research_test_split(frame, purging_coefficient=10, min_research_period=0,
                            min_test_period=0, min_research_unique_pts=1, min_test_unique_pts=1)


@pytest.mark.parametrize('minimum,outcome', [(4, 'ok'), (8, 'warn'), (10, 'error')])
def test_split_exact_research_minimum_thresholds(minimum, outcome):
    """Should accept the recommendation, warn at half, and reject below half."""
    frame = pd.DataFrame({'time': pd.date_range('2024-01-01', periods=6)})
    kwargs = dict(purging_coefficient=1, min_research_period=0, min_test_period=0,
                  min_research_unique_pts=minimum, min_test_unique_pts=2)
    if outcome == 'error':
        with pytest.raises(ValueError, match='Research block too small'):
            research_test_split(frame, **kwargs)
    elif outcome == 'warn':
        with pytest.warns(UserWarning, match='smaller than recommended'):
            research, _ = research_test_split(frame, **kwargs)
        assert len(research) == 4
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            research, _ = research_test_split(frame, **kwargs)
        assert len(research) == 4


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('research_count', [2, 3])
def test_split_odd_point_minimum_uses_exact_half(backend, research_count):
    """Should reject two research dates and allow three when the recommendation is five."""
    frame = pd.DataFrame({'time': pd.date_range('2024-01-01', periods=research_count + 2)})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    kwargs = dict(purging_coefficient=1, min_research_period=0, min_test_period=0,
                  min_research_unique_pts=5, min_test_unique_pts=2)
    if research_count == 2:
        with pytest.raises(ValueError, match='Minimum allowed: 3 dates'):
            research_test_split(frame, **kwargs)
    else:
        with pytest.warns(UserWarning, match='smaller than recommended'):
            research, test = research_test_split(frame, **kwargs)
        assert len(research) == 3 and len(test) == 2


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('research_months', [2, 3])
def test_split_odd_month_minimum_uses_exact_half(backend, research_months):
    """Should reject two full months and allow three when the recommendation is five."""
    # The final observation is test-only; Jan 1 through Mar 1 spans two
    # full months, and Jan 1 through Apr 1 spans three full months.
    frame = pd.DataFrame({'time': pd.date_range('2024-01-01', periods=research_months + 2, freq='MS')})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    kwargs = dict(purging_coefficient=1, min_research_period=5, min_test_period=0,
                  min_research_unique_pts=1, min_test_unique_pts=1)
    if research_months == 2:
        with pytest.raises(ValueError, match=r'Minimum allowed: 1 dates \(3 months\)'):
            research_test_split(frame, **kwargs)
    else:
        with pytest.warns(UserWarning, match='smaller than recommended'):
            research, test = research_test_split(frame, **kwargs)
        assert len(research) == 4 and len(test) == 1


def test_split_date_helper_preserves_valid_precision():
    """Should return timezone-aware nanosecond keys without rounding their values."""
    dates = pd.date_range('2024-01-01', periods=3, freq='ns', tz='UTC')
    for frame in (pd.DataFrame({'time': dates}), pl.from_pandas(pd.DataFrame({'time': dates}))):
        pd.testing.assert_index_equal(_split_dates(frame, 'time'), pd.DatetimeIndex(dates, name='time'))

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
