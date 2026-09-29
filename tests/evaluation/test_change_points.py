from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.evaluation import detect_change_points
from alpha_research.evaluation.change_points import PeltResult


@pytest.fixture
def piecewise_frame():
    """Should provide three hand-verifiable four-observation segments."""
    return pd.DataFrame({
        'window_end': pd.date_range('2024-01-01', periods=12, freq='D'),
        'metric': [0, 1, 0, 1, 10, 11, 10, 11, 0, 1, 0, 1],
    }, index=range(100, 112))


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_detect_change_points_finds_two_exact_boundaries(piecewise_frame, backend):
    """Should choose the two known mean shifts under a hand-checked penalty."""
    frame = piecewise_frame if backend == 'pandas' else pl.from_pandas(piecewise_frame)

    result = detect_change_points(
        frame, value_col='metric', time_col='window_end', model='l2', penalty=20,
    )

    # Each intended segment has SSE=1; its total cost is 3 + 2*20 = 43.
    # Moving either boundary one step mixes 1 and 10, adding more than 20
    # to SSE. An extra boundary can save at most one unit of within-segment SSE.
    assert isinstance(result, PeltResult)
    assert [point.position for point in result.breakpoints] == [4, 8]
    assert [point.time for point in result.breakpoints] == [
        piecewise_frame['window_end'].iloc[4],
        piecewise_frame['window_end'].iloc[8],
    ]
    assert [(part.start, part.stop, part.n_obs) for part in result.segments] == [
        (0, 4, 4), (4, 8, 4), (8, 12, 4),
    ]
    assert [part.mean for part in result.segments] == [0.5, 10.5, 0.5]
    assert [part.variance for part in result.segments] == pytest.approx([1 / 3] * 3)
    assert result.segments[0].first_time == piecewise_frame['window_end'].iloc[0]
    assert result.segments[0].last_time == piecewise_frame['window_end'].iloc[3]
    assert result.segments[-1].last_time == piecewise_frame['window_end'].iloc[-1]
    assert (result.value_col, result.time_col, result.model) == (
        'metric', 'window_end', 'l2',
    )
    assert (result.penalty, result.min_size, result.jump, result.n_obs) == (
        20.0, 2, 1, 12,
    )


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_detect_change_points_returns_one_segment_when_no_break(backend):
    """Should distinguish an evaluated no-break result from no result."""
    pandas_frame = pd.DataFrame({'time': range(8), 'value': [2.0] * 8})
    frame = pandas_frame if backend == 'pandas' else pl.from_pandas(pandas_frame)

    result = detect_change_points(frame, 'value', 'l2', penalty=1)

    assert result.breakpoints == ()
    assert len(result.segments) == 1
    assert (result.segments[0].start, result.segments[0].stop) == (0, 8)
    assert result.segments[0].mean == 2.0
    assert result.segments[0].variance == 0.0


def test_detect_change_points_uses_positional_time_not_pandas_index(piecewise_frame):
    """Should map the first new observation to its time rather than index label."""
    result = detect_change_points(
        piecewise_frame, 'metric', 'l2', penalty=20, time_col='window_end',
    )

    assert result.breakpoints[0].position == 4
    assert result.breakpoints[0].time == pd.Timestamp('2024-01-05')
    assert result.breakpoints[0].position != piecewise_frame.index[4]


def test_detect_change_points_respects_candidate_grid():
    """Should consider only grid boundaries and retain the original time key."""
    frame = pd.DataFrame({
        'time': range(12),
        'value': [0.0] * 4 + [10.0] * 8,
    })

    result = detect_change_points(
        frame, 'value', 'l2', penalty=30, jump=3,
    )

    # Among candidates 3, 6, and 9, position 3 has the smallest SSE:
    # (0-80/9)^2 + 8*(10-80/9)^2 = 800/9 < 400/3 at 6.
    # An additional boundary at 6 lowers SSE to 200/3, saving 200/9 < 30.
    assert [point.position for point in result.breakpoints] == [3]
    assert [point.time for point in result.breakpoints] == [3]


def test_detect_change_points_records_single_observation_variance():
    """Should mark one-observation sample variance undefined when allowed."""
    frame = pd.DataFrame({'time': [0, 1, 2], 'value': [0.0, 0.0, 100.0]})

    result = detect_change_points(frame, 'value', 'l2', penalty=10, min_size=1)

    assert [point.position for point in result.breakpoints] == [2]
    assert result.segments[-1].n_obs == 1
    assert np.isnan(result.segments[-1].variance)


@pytest.mark.parametrize('model,penalty', [
    ('l1', 5),
    ('l2', 5),
    pytest.param(
        'normal', 5,
        marks=pytest.mark.filterwarnings(
            r'ignore:^New behaviour in v1\.1\.5:UserWarning:ruptures\.costs\.costnormal'
        ),
    ),
    ('rbf', 1),
])
def test_detect_change_points_supports_documented_costs(model, penalty):
    """Should locate a unique boundary between two constant blocks for each cost."""
    frame = pd.DataFrame({
        'time': range(20),
        'value': [0.0] * 10 + [10.0] * 10,
    })

    result = detect_change_points(
        frame, 'value', model, penalty=penalty, min_size=3,
    )

    # Homogeneous blocks have zero dispersion and an internal split adds
    # a penalty without reducing their cost. A mixed block has nonzero cost.
    assert [point.position for point in result.breakpoints] == [10]


def test_detect_change_points_respects_penalty_and_minimum_size():
    """Should retain one segment when a break is too costly or infeasible."""
    frame = pd.DataFrame({'time': range(6), 'value': [0.0] * 3 + [10.0] * 3})

    expensive = detect_change_points(frame, 'value', 'l2', penalty=200)
    infeasible = detect_change_points(
        frame, 'value', 'l2', penalty=1, min_size=4,
    )

    # A split at 3 reduces SSE from 150 to zero, so penalty 200 prevents it.
    # min_size=4 makes two segments impossible among only six observations.
    assert expensive.breakpoints == ()
    assert infeasible.breakpoints == ()


def test_detect_change_points_result_is_immutable(piecewise_frame):
    """Should keep segmentation metadata immutable after calculation."""
    result = detect_change_points(
        piecewise_frame, 'metric', 'l2', penalty=20, time_col='window_end',
    )

    with pytest.raises(FrozenInstanceError):
        result.penalty = 100.0
    with pytest.raises(FrozenInstanceError):
        result.breakpoints[0].position = 5


@pytest.mark.parametrize('name,value', [
    ('value_col', ''),
    ('value_col', None),
    ('time_col', ''),
    ('time_col', 3),
    ('model', 'unknown'),
    ('model', None),
    ('model', ['l2']),
    ('penalty', 0),
    ('penalty', -1),
    ('penalty', float('nan')),
    ('penalty', float('inf')),
    ('min_size', 0),
    ('min_size', True),
    ('min_size', 1.5),
    ('jump', 0),
    ('jump', True),
    ('jump', 1.5),
])
def test_detect_change_points_rejects_invalid_arguments(name, value):
    """Should reject invalid identifiers and segmentation settings early."""
    frame = pd.DataFrame({'time': range(6), 'value': [0, 0, 0, 5, 5, 5]})
    arguments = {'value_col': 'value', 'time_col': 'time', 'model': 'l2', 'penalty': 5}
    arguments[name] = value

    with pytest.raises(ValueError):
        detect_change_points(frame, **arguments)


@pytest.mark.parametrize('penalty', [True, '10', None])
def test_detect_change_points_rejects_non_numeric_penalty(penalty):
    """Should reject penalty types that could be silently coerced."""
    frame = pd.DataFrame({'time': range(4), 'value': [0.0, 0.0, 5.0, 5.0]})

    with pytest.raises(TypeError):
        detect_change_points(frame, 'value', 'l2', penalty=penalty)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('times', [
    [0, 1, 1, 3],
    [0, 2, 1, 3],
    [0, None, 2, 3],
    [0, float('nan'), 2, 3],
])
def test_detect_change_points_rejects_invalid_time_order(backend, times):
    """Should reject missing, repeated, and unordered observation keys."""
    pandas_frame = pd.DataFrame({'time': times, 'value': [0.0, 0.0, 5.0, 5.0]})
    frame = pandas_frame if backend == 'pandas' else pl.from_pandas(pandas_frame)

    with pytest.raises(ValueError):
        detect_change_points(frame, 'value', 'l2', penalty=1)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_detect_change_points_rejects_infinite_numeric_time(backend):
    """Should reject an infinite numeric key even if the values are ordered."""
    pandas_frame = pd.DataFrame({
        'time': [0.0, 1.0, 2.0, float('inf')],
        'value': [0.0, 0.0, 5.0, 5.0],
    })
    frame = pandas_frame if backend == 'pandas' else pl.from_pandas(pandas_frame)

    with pytest.raises(ValueError, match='time must contain only finite values'):
        detect_change_points(frame, 'value', 'l2', penalty=1)


def test_detect_change_points_rejects_nullable_pandas_missing_data():
    """Should reject Pandas nullable numeric gaps without coercion errors."""
    frame = pd.DataFrame({
        'time': pd.Series([0, 1, 2, 3], dtype='Int64'),
        'value': pd.Series([0, 0, pd.NA, 5], dtype='Int64'),
    })

    with pytest.raises(ValueError, match='value must contain only finite values'):
        detect_change_points(frame, 'value', 'l2', penalty=1)

    frame['value'] = pd.Series([0, 0, 5, 5], dtype='Int64')
    frame.loc[2, 'time'] = pd.NA
    with pytest.raises(ValueError, match='time must not contain missing values'):
        detect_change_points(frame, 'value', 'l2', penalty=1)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('bad_value', [None, float('nan'), float('inf'), -float('inf')])
def test_detect_change_points_rejects_nonfinite_values(backend, bad_value):
    """Should never compress missing observations or pass infinities to PELT."""
    pandas_frame = pd.DataFrame({'time': range(4), 'value': [0.0, 1.0, bad_value, 2.0]})
    frame = pandas_frame if backend == 'pandas' else pl.from_pandas(pandas_frame)

    with pytest.raises(ValueError):
        detect_change_points(frame, 'value', 'l2', penalty=1)


@pytest.mark.parametrize('values', [
    ['0', '0', '5', '5'],
    [False, False, True, True],
])
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_detect_change_points_rejects_non_numeric_values(backend, values):
    """Should reject strings and booleans rather than silently coerce them."""
    pandas_frame = pd.DataFrame({'time': range(4), 'value': values})
    frame = pandas_frame if backend == 'pandas' else pl.from_pandas(pandas_frame)

    with pytest.raises(TypeError):
        detect_change_points(frame, 'value', 'l2', penalty=1)


def test_detect_change_points_rejects_invalid_frame_and_columns():
    """Should require a nonempty frame with two distinct existing columns."""
    frame = pd.DataFrame({'time': [0, 1, 2], 'value': [0.0, 1.0, 2.0]})

    with pytest.raises(TypeError):
        detect_change_points([0, 1, 2], 'value', 'l2', penalty=1)
    with pytest.raises(ValueError):
        detect_change_points(frame.iloc[:0], 'value', 'l2', penalty=1)
    with pytest.raises(KeyError):
        detect_change_points(frame, 'missing', 'l2', penalty=1)
    with pytest.raises(ValueError):
        detect_change_points(frame, 'time', 'l2', penalty=1)
    with pytest.raises(ValueError):
        detect_change_points(frame, 'value', 'l2', penalty=1, min_size=4)


def test_detect_change_points_explains_missing_optional_dependency(monkeypatch):
    """Should provide the install extra when ruptures is unavailable."""
    frame = pd.DataFrame({'time': range(4), 'value': [0.0, 0.0, 5.0, 5.0]})
    monkeypatch.setitem(__import__('sys').modules, 'ruptures', None)

    with pytest.raises(ImportError, match=r'alpha-research\[changepoint\]'):
        detect_change_points(frame, 'value', 'l2', penalty=1)
