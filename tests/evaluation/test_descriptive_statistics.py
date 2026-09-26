import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.evaluation import (
    distribution_summary,
    rolling_distribution_summary,
)
from alpha_research.evaluation.descriptive_statistics import (
    _prepare_value_frame,
    _summarize_values,
)
from alpha_research.evaluation.timeseries import rolling_temporal_association


@pytest.fixture
def distribution_frame_pandas():
    """Create two assets with hand-verifiable values at five dates."""
    return pd.DataFrame({
        'time': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'symbol': ['A', 'B'] * 5,
        'feature': [0.0, 10.0, 1.0, 20.0, 2.0, 30.0, 3.0, 40.0, 4.0, 50.0],
    })


def test_summarize_values_uses_independent_known_statistics():
    """Should report variance 2.5 and corrected excess kurtosis -1.2 for 0..4."""
    result = _summarize_values(pl.Series([0.0, 1.0, 2.0, 3.0, 4.0]))

    assert (result['n_total'], result['n_valid'], result['n_missing'], result['n_infinite']) == (5, 5, 0, 0)
    assert result['mean'] == result['median'] == 2.0
    assert result['std'] == pytest.approx(np.sqrt(2.5))
    assert result['min'] == 0.0
    assert result['max'] == 4.0
    assert result['q01'] == pytest.approx(0.04)
    assert result['q05'] == pytest.approx(0.2)
    assert result['q25'] == 1.0
    assert result['q75'] == 3.0
    assert result['q95'] == pytest.approx(3.8)
    assert result['q99'] == pytest.approx(3.96)
    assert result['iqr'] == 2.0
    assert result['skewness'] == pytest.approx(0.0)
    assert result['excess_kurtosis'] == pytest.approx(-1.2)


def test_summarize_values_corrects_asymmetric_sample_shape():
    """Should calculate corrected skewness 2 and excess kurtosis 4 for 0,0,0,2."""
    result = _summarize_values(pl.Series([0.0, 0.0, 0.0, 2.0]))

    assert result['skewness'] == pytest.approx(2.0)
    assert result['excess_kurtosis'] == pytest.approx(4.0)


def test_summarize_values_separates_missing_infinite_and_finite():
    """Should use only 2 and 4 for quantiles and count NaN and infinity."""
    result = _summarize_values(pl.Series([np.nan, np.inf, -np.inf, 2.0, 4.0]))

    assert (result['n_total'], result['n_valid'], result['n_missing'], result['n_infinite']) == (5, 2, 1, 2)
    assert result['median'] == 3.0
    assert result['q25'] == 2.5
    assert result['q75'] == 3.5
    assert result['iqr'] == 1.0
    assert np.isnan(result['skewness'])
    assert np.isnan(result['excess_kurtosis'])


@pytest.mark.parametrize('values', [
    pl.Series([np.nan, np.nan]),
    pl.Series([np.inf, -np.inf]),
])
def test_summarize_values_all_invalid_is_explicit(values):
    """Should retain counts and undefined statistics for groups without finite values."""
    result = _summarize_values(values)

    assert result['n_valid'] == 0
    assert np.isnan(result['median'])
    assert np.isnan(result['iqr'])
    assert np.isnan(result['excess_kurtosis'])


def test_summarize_values_accepts_an_untyped_all_missing_group():
    """Should count all-null groups even when no numeric dtype can be inferred."""
    result = _summarize_values(pl.Series([None, None]))

    assert result['n_total'] == result['n_missing'] == 2
    assert result['n_valid'] == result['n_infinite'] == 0
    assert np.isnan(result['median'])


def test_summarize_values_constant_and_small_samples():
    """Should leave shape statistics undefined for constant or undersized samples."""
    one = _summarize_values(pl.Series([7.0]))
    constant = _summarize_values(pl.Series([7.0] * 5))

    assert np.isnan(one['std'])
    assert one['median'] == 7.0
    assert constant['std'] == 0.0
    assert np.isnan(constant['skewness'])
    assert np.isnan(constant['excess_kurtosis'])


@pytest.mark.parametrize('values', [
    [1.0, 2.0],
    pd.Series(['a', 'b']),
    pl.Series(['a', 'b']),
    pl.Series([True, False]),
])
def test_summarize_values_rejects_unsupported_input(values):
    """Should accept only numeric, non-boolean Series data."""
    with pytest.raises(TypeError, match='numeric, non-boolean'):
        _summarize_values(values)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_prepare_value_frame_infers_or_selects_a_value(distribution_frame_pandas, backend):
    """Should preserve keys while accepting a wider paired frame."""
    frame = distribution_frame_pandas if backend == 'pandas' else pl.from_pandas(distribution_frame_pandas)
    prepared, name = _prepare_value_frame(frame, None, 'time', 'symbol')
    assert name == 'feature'
    assert isinstance(prepared, type(frame))
    assert list(prepared.columns) == ['time', 'symbol', 'feature']

    wider = frame.assign(target=1.0) if backend == 'pandas' else frame.with_columns(pl.lit(1.0).alias('target'))
    prepared, name = _prepare_value_frame(wider, 'feature', 'time', 'symbol')
    assert name == 'feature'
    assert isinstance(prepared, type(frame))
    assert list(prepared.columns) == ['time', 'symbol', 'feature']


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_prepare_value_frame_rejects_bad_keys_and_values(distribution_frame_pandas, backend):
    """Should reject missing or repeated keys and nonnumeric values."""
    frame = distribution_frame_pandas.copy()
    frame.loc[0, 'symbol'] = None
    candidate = frame if backend == 'pandas' else pl.from_pandas(frame)
    with pytest.raises(ValueError, match='symbol must not contain missing'):
        _prepare_value_frame(candidate, None, 'time', 'symbol')

    frame = distribution_frame_pandas.copy()
    frame.loc[1, 'symbol'] = 'A'
    candidate = frame if backend == 'pandas' else pl.from_pandas(frame)
    with pytest.raises(ValueError, match='unique'):
        _prepare_value_frame(candidate, None, 'time', 'symbol')

    frame = distribution_frame_pandas.assign(feature='text')
    candidate = frame if backend == 'pandas' else pl.from_pandas(frame)
    with pytest.raises(TypeError, match='numeric'):
        _prepare_value_frame(candidate, None, 'time', 'symbol')


def test_prepare_value_frame_rejects_missing_columns_and_overlaps(distribution_frame_pandas):
    """Should require column inference and explicit selection to identify one distinct value."""
    with pytest.raises(KeyError, match='missing required columns'):
        _prepare_value_frame(distribution_frame_pandas, 'absent', 'time', 'symbol')
    with pytest.raises(ValueError, match='distinct'):
        _prepare_value_frame(distribution_frame_pandas, 'feature', 'time', 'time')
    with pytest.raises(ValueError, match='differ'):
        _prepare_value_frame(distribution_frame_pandas, 'time', 'time', 'symbol')
    with pytest.raises(ValueError, match='exactly one'):
        _prepare_value_frame(distribution_frame_pandas.assign(target=1.0), None, 'time', 'symbol')
    with pytest.raises(ValueError, match='must not be empty'):
        _prepare_value_frame(distribution_frame_pandas.iloc[0:0], None, 'time', 'symbol')
    with pytest.raises(TypeError, match='Pandas or Polars'):
        _prepare_value_frame([], None, 'time', 'symbol')
    with pytest.raises(TypeError, match='numeric'):
        _prepare_value_frame(
            distribution_frame_pandas.assign(feature=True),
            None,
            'time',
            'symbol',
        )


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_distribution_summary_returns_requested_group_rows(distribution_frame_pandas, backend):
    """Should summarize each requested scope without mixing dates or assets."""
    frame = distribution_frame_pandas if backend == 'pandas' else pl.from_pandas(distribution_frame_pandas)
    overall = distribution_summary(frame)
    by_symbol = distribution_summary(frame, group_by='symbol')
    by_time = distribution_summary(frame, group_by='time')
    assert isinstance(overall, type(frame))
    assert isinstance(by_symbol, type(frame))
    assert isinstance(by_time, type(frame))
    if backend == 'polars':
        overall, by_symbol, by_time = overall.to_pandas(), by_symbol.to_pandas(), by_time.to_pandas()

    assert len(overall) == 1
    assert overall.loc[0, 'value_name'] == 'feature'
    assert overall.loc[0, 'n_valid'] == 10
    assert by_symbol['symbol'].tolist() == ['A', 'B']
    assert by_symbol['median'].tolist() == [2.0, 30.0]
    assert by_time['time'].tolist() == [1, 2, 3, 4, 5]
    assert by_time['median'].tolist() == [5.0, 10.5, 16.0, 21.5, 27.0]
    assert by_time['n_total'].tolist() == [2] * 5


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_distribution_summary_keeps_all_missing_groups(distribution_frame_pandas, backend):
    """Should retain fully missing dates with counts and undefined quantiles."""
    frame = distribution_frame_pandas.copy()
    frame.loc[frame['time'] == 2, 'feature'] = np.nan
    frame = frame if backend == 'pandas' else pl.from_pandas(frame)
    result = distribution_summary(frame, group_by='time')
    result = result if backend == 'pandas' else result.to_pandas()

    missing_date = result.loc[result['time'] == 2].iloc[0]
    assert missing_date['n_total'] == 2
    assert missing_date['n_missing'] == 2
    assert missing_date['n_valid'] == 0
    assert np.isnan(missing_date['median'])


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_distribution_summary_accepts_an_all_null_value_column(backend):
    """Should retain counts for a wholly null feature in both backends."""
    frame = pd.DataFrame({
        'time': [1, 2],
        'symbol': ['A', 'A'],
        'feature': [None, None],
    })
    frame = frame if backend == 'pandas' else pl.from_pandas(frame)
    result = distribution_summary(frame)
    result = result if backend == 'pandas' else result.to_pandas()

    assert result.loc[0, 'n_total'] == 2
    assert result.loc[0, 'n_missing'] == 2
    assert result.loc[0, 'n_valid'] == 0
    assert np.isnan(result.loc[0, 'median'])


def test_distribution_summary_rejects_unsupported_grouping(distribution_frame_pandas):
    """Should restrict grouping to overall, symbol, and date scopes."""
    with pytest.raises(ValueError, match='group_by'):
        distribution_summary(distribution_frame_pandas, group_by='month')


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_distribution_summary_respects_custom_key_names(distribution_frame_pandas, backend):
    """Should group by configured key names instead of default names."""
    renamed = distribution_frame_pandas.rename(columns={'time': 'date', 'symbol': 'ticker'})
    frame = renamed if backend == 'pandas' else pl.from_pandas(renamed)

    by_ticker = distribution_summary(
        frame, group_by='ticker', time_col='date', symbol_col='ticker',
    )
    by_date = distribution_summary(
        frame, group_by='date', time_col='date', symbol_col='ticker',
    )
    with pytest.raises(ValueError, match='group_by'):
        distribution_summary(frame, group_by='symbol', time_col='date', symbol_col='ticker')
    if backend == 'polars':
        by_ticker, by_date = by_ticker.to_pandas(), by_date.to_pandas()

    assert by_ticker['ticker'].tolist() == ['A', 'B']
    assert by_ticker['median'].tolist() == [2.0, 30.0]
    assert by_date['date'].tolist() == [1, 2, 3, 4, 5]
    assert by_date['median'].tolist() == [5.0, 10.5, 16.0, 21.5, 27.0]


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_rolling_distribution_summary_uses_full_single_asset_windows(distribution_frame_pandas, backend):
    """Should return the three known medians for five observations and three-row windows."""
    single = distribution_frame_pandas.loc[distribution_frame_pandas['symbol'] == 'A']
    frame = single if backend == 'pandas' else pl.from_pandas(single)
    result = rolling_distribution_summary(frame, window_size=3)
    assert isinstance(result, type(frame))
    result = result if backend == 'pandas' else result.to_pandas()

    assert result['window_start'].tolist() == [1, 2, 3]
    assert result['window_end'].tolist() == [3, 4, 5]
    assert result['symbol'].tolist() == ['A'] * 3
    assert result['median'].tolist() == [1.0, 2.0, 3.0]
    assert result['q25'].tolist() == [0.5, 1.5, 2.5]
    assert result['iqr'].tolist() == [1.0] * 3
    assert result['n_total'].tolist() == [3] * 3


def test_rolling_distribution_summary_counts_missing_without_shortening_windows(distribution_frame_pandas):
    """Should keep an internal missing value in the window denominator."""
    single = distribution_frame_pandas.loc[distribution_frame_pandas['symbol'] == 'A'].copy()
    single.loc[single['time'] == 3, 'feature'] = np.nan
    result = rolling_distribution_summary(single, window_size=3, window_step=2)

    assert result['window_end'].tolist() == [3, 5]
    assert result['n_total'].tolist() == [3, 3]
    assert result['n_valid'].tolist() == [2, 2]
    assert result['n_missing'].tolist() == [1, 1]
    assert result['median'].tolist() == [0.5, 3.5]


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_rolling_distribution_summary_respects_custom_keys_and_missing_values(
        distribution_frame_pandas,
        backend,
):
    """Should preserve window bounds with custom keys and missing rows."""
    single = distribution_frame_pandas.loc[distribution_frame_pandas['symbol'] == 'A'].copy()
    single.loc[single['time'] == 3, 'feature'] = np.nan
    renamed = single.rename(columns={'time': 'date', 'symbol': 'ticker'})
    frame = renamed if backend == 'pandas' else pl.from_pandas(renamed)
    result = rolling_distribution_summary(
        frame, window_size=3, window_step=2, time_col='date', symbol_col='ticker',
    )
    assert isinstance(result, type(frame))
    result = result if backend == 'pandas' else result.to_pandas()

    assert result['ticker'].tolist() == ['A', 'A']
    assert result['window_start'].tolist() == [1, 3]
    assert result['window_end'].tolist() == [3, 5]
    assert result['n_valid'].tolist() == [2, 2]
    assert result['n_missing'].tolist() == [1, 1]
    assert result['median'].tolist() == [0.5, 3.5]


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('window_size', [1, 2])
def test_rolling_distribution_summary_handles_windows_too_small_for_shape(
        distribution_frame_pandas,
        backend,
        window_size,
):
    """Should preserve quantiles and leave shape undefined for short windows."""
    single = distribution_frame_pandas.loc[distribution_frame_pandas['symbol'] == 'A']
    frame = single if backend == 'pandas' else pl.from_pandas(single)
    result = rolling_distribution_summary(frame, window_size=window_size)
    result = result if backend == 'pandas' else result.to_pandas()

    assert result['n_total'].tolist() == [window_size] * (6 - window_size)
    assert result.loc[0, 'median'] == pytest.approx((window_size - 1) / 2)
    assert result['skewness'].isna().all()
    assert result['excess_kurtosis'].isna().all()


def test_polars_distribution_paths_do_not_convert_to_pandas(distribution_frame_pandas, monkeypatch):
    """Should keep public Polars calculations in Polars through their results."""
    def reject_conversion(*args, **kwargs):
        """Reject an unexpected DataFrame conversion.

        Parameters
        ----------
        *args, **kwargs
            Arguments from a possible DataFrame conversion call.

        Returns
        -------
        None
            This function never returns.

        Raises
        ------
        AssertionError
            Always, because conversion would defeat the native Polars path.
        """
        raise AssertionError('Polars frame converted to Pandas.')

    frame = pl.from_pandas(distribution_frame_pandas)
    monkeypatch.setattr(pl.DataFrame, 'to_pandas', reject_conversion)
    assert distribution_summary(frame).height == 1
    assert distribution_summary(frame, group_by='symbol').height == 2
    assert distribution_summary(frame, group_by='time').height == 5
    single = frame.filter(pl.col('symbol') == 'A')
    assert rolling_distribution_summary(single, window_size=3).height == 3


def test_pandas_and_polars_match_for_nonfinite_values_and_rolling_shape():
    """Should return matching backend schemas and metrics for finite and invalid rows."""
    frame = pd.DataFrame({
        'time': list(range(1, 9)),
        'symbol': ['A'] * 8,
        'feature': [0.0, 0.0, 0.0, 2.0, np.nan, 3.0, np.inf, 4.0],
    })
    polars_frame = pl.from_pandas(frame)
    for pandas_result, polars_result in [
        (distribution_summary(frame), distribution_summary(polars_frame)),
        (
            rolling_distribution_summary(frame, window_size=4),
            rolling_distribution_summary(polars_frame, window_size=4),
        ),
    ]:
        assert pandas_result.columns.tolist() == polars_result.columns
        converted = polars_result.to_pandas()
        assert pandas_result['n_total'].tolist() == converted['n_total'].tolist()
        assert pandas_result['n_valid'].tolist() == converted['n_valid'].tolist()
        assert pandas_result['n_missing'].tolist() == converted['n_missing'].tolist()
        assert pandas_result['n_infinite'].tolist() == converted['n_infinite'].tolist()
        for metric in ['mean', 'std', 'min', 'q01', 'q05', 'q25', 'median',
                       'q75', 'q95', 'q99', 'max', 'iqr', 'skewness', 'excess_kurtosis']:
            np.testing.assert_allclose(
                pandas_result[metric].to_numpy(dtype=float),
                converted[metric].to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_full_rolling_window_matches_overall_hand_calculated_shape(backend):
    """Should match the known q75 and corrected moments for a full 0,0,0,2 window."""
    frame = pd.DataFrame({
        'time': [1, 2, 3, 4],
        'symbol': ['A'] * 4,
        'feature': [0.0, 0.0, 0.0, 2.0],
    })
    frame = frame if backend == 'pandas' else pl.from_pandas(frame)
    overall = distribution_summary(frame)
    rolling = rolling_distribution_summary(frame, window_size=4)
    overall = overall if backend == 'pandas' else overall.to_pandas()
    rolling = rolling if backend == 'pandas' else rolling.to_pandas()

    for metric, expected in [('q75', 0.5), ('iqr', 0.5),
                             ('skewness', 2.0), ('excess_kurtosis', 4.0)]:
        assert overall.loc[0, metric] == pytest.approx(expected)
        assert rolling.loc[0, metric] == pytest.approx(expected)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_rolling_distribution_keeps_an_all_null_window(backend):
    """Should report counts and undefined quantiles for all-null windows in both backends."""
    frame = pd.DataFrame({
        'time': [1, 2, 3],
        'symbol': ['A'] * 3,
        'feature': [None, None, None],
    })
    frame = frame if backend == 'pandas' else pl.from_pandas(frame)
    result = rolling_distribution_summary(frame, window_size=2)
    result = result if backend == 'pandas' else result.to_pandas()

    assert result['n_total'].tolist() == [2, 2]
    assert result['n_missing'].tolist() == [2, 2]
    assert result['n_valid'].tolist() == [0, 0]
    assert result['median'].isna().all()


def test_rolling_distribution_aligns_with_strict_temporal_association(distribution_frame_pandas):
    """Should align shared raw dates when association rejects incomplete pairs."""
    single = distribution_frame_pandas.loc[distribution_frame_pandas['symbol'] == 'A'].copy()
    single.loc[single['time'] == 3, 'feature'] = np.nan
    single['target'] = [0.0, 2.0, 4.0, 6.0, 8.0]
    distribution = rolling_distribution_summary(
        single,
        window_size=3,
        window_step=2,
        value_col='feature',
    )
    association = rolling_temporal_association(
        single,
        feature='feature',
        target='target',
        window_size=3,
        window_step=2,
        block_length=2,
        n_bootstraps=2,
    ).rolling_frame

    assert distribution['window_start'].tolist() == association['window_start'].tolist()
    assert distribution['window_end'].tolist() == association['window_end'].tolist()
    assert distribution['n_valid'].tolist() == [2, 2]
    assert association['status'].tolist() == ['missing_pairs', 'missing_pairs']


@pytest.mark.parametrize('argument', [
    {'window_size': 0},
    {'window_size': True},
    {'window_size': 6},
    {'window_size': 2, 'window_step': 0},
])
def test_rolling_distribution_summary_rejects_bad_window_arguments(distribution_frame_pandas, argument):
    """Should reject window sizes and steps outside the valid input range."""
    single = distribution_frame_pandas.loc[distribution_frame_pandas['symbol'] == 'A']
    with pytest.raises(ValueError):
        rolling_distribution_summary(single, **argument)


def test_rolling_distribution_summary_rejects_multiple_or_unordered_assets(distribution_frame_pandas):
    """Should reject multiple symbols and unordered observations in rolling windows."""
    with pytest.raises(ValueError, match='one symbol'):
        rolling_distribution_summary(distribution_frame_pandas, window_size=2)
    single = distribution_frame_pandas.loc[distribution_frame_pandas['symbol'] == 'A'].iloc[::-1]
    with pytest.raises(ValueError, match='increasingly ordered'):
        rolling_distribution_summary(single, window_size=2)
