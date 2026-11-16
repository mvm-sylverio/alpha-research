import pandas as pd
import polars as pl
import pytest

from alpha_research.features.encoding import (
    _map_frame,
    binary_mapping,
    ordinal_mapping,
    validate_mapping,
)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_mapping_values_and_originals(backend):
    """Should encode explicit label positions and retain labels, row order and missing values."""
    frame = pd.DataFrame({'id': ['003', '001', '002', '004'], 'x': ['High', 'Low', 'Medium', None]})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    result = ordinal_mapping(frame, 'x', ['Low', 'Medium', 'High'], [-1, 5, 20])
    actual = result['x_ordinal'].to_list()
    assert actual[:3] == [20, -1, 5]
    assert pd.isna(actual[3])
    assert result['id'].to_list() == ['003', '001', '002', '004']
    assert result['x'].to_list()[:3] == ['High', 'Low', 'Medium']
    assert 'x_ordinal' not in frame.columns
    default = ordinal_mapping(frame, ['x'], ['High', 'Low', 'Medium'])
    assert default['x_ordinal'].to_list()[:3] == [0, 1, 2]


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_binary_unknown_and_collisions(backend):
    """Should encode chosen binary labels, reject unknowns and protect existing columns."""
    frame = pd.DataFrame({'x': ['Yes', 'No', None]})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    assert binary_mapping(frame, 'x', ['No', 'Yes'])['x_binary'].to_list()[:2] == [1, 0]
    with pytest.raises(ValueError, match='Unknown category'):
        ordinal_mapping(frame, 'x', ['No', 'Other'])
    with pytest.raises(ValueError):
        ordinal_mapping(frame, [], ['No', 'Yes'])
    with pytest.raises(ValueError):
        ordinal_mapping(frame, ['absent'], ['No', 'Yes'])
    with pytest.raises(TypeError):
        ordinal_mapping(frame, 1, ['No', 'Yes'])
    existing = binary_mapping(frame, 'x', ['No', 'Yes'])
    with pytest.raises(ValueError, match='already exists'):
        binary_mapping(existing, 'x', ['No', 'Yes'])
    existing = ordinal_mapping(frame, 'x', ['No', 'Yes'])
    with pytest.raises(ValueError, match='already exists'):
        ordinal_mapping(existing, 'x', ['No', 'Yes'])


@pytest.mark.parametrize('categories,codes', [
    ([], []), (['a', 'a'], [0, 1]), (['a', None], [0, 1]), ([float('nan')], [0]),
    ([[]], [0]), (['a'], [True]), (['a'], [float('inf')]), (['a'], []),
    (['a', 'b'], [1, 0]), (['a', 'b'], [1, 1]), ('ab', [0, 1]),
])
def test_invalid_mapping(categories, codes):
    """Should reject missing or ambiguous labels and invalid or nonincreasing codes."""
    with pytest.raises(ValueError):
        validate_mapping(categories, codes)


def test_mapping_validation_and_invalid_frames():
    """Should enforce binary zero/one codes and reject unsupported frames."""
    validate_mapping(['No', 'Yes'], [0, 1], binary=True)
    with pytest.raises(ValueError):
        validate_mapping(['No', 'Yes'], [1, 2], binary=True)
    with pytest.raises(ValueError):
        binary_mapping(pd.DataFrame({'x': ['a']}), 'x', ['a'])
    for function in (ordinal_mapping, binary_mapping):
        with pytest.raises(TypeError):
            function({}, 'x', ['a', 'b'])


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_shared_mapper_missing_numeric_labels_and_independent_outputs(backend):
    """Should map numeric labels and missing values without reserving other output suffixes."""
    frame = pd.DataFrame({'x': [20., float('nan'), 5.], 'x_ordinal': ['keep', 'all', 'labels']})
    if backend == 'polars':
        frame = pl.from_pandas(frame, nan_to_null=False)
    result = _map_frame(frame, 'x', [5, 20], [-1, 3], '_code')
    assert result['x_code'].to_list()[0] == 3
    assert pd.isna(result['x_code'].to_list()[1])
    assert result['x_code'].to_list()[2] == -1
    assert binary_mapping(frame, 'x', [5, 20])['x_binary'].to_list()[0] == 1
    with pytest.raises(ValueError):
        _map_frame(frame, 'x', [5, 20], [2, 1], '_code')
    with pytest.raises(TypeError):
        _map_frame([], 'x', [5, 20], [0, 1], '_code')
    empty = pd.DataFrame({'x': [None, None]}) if backend == 'pandas' else pl.DataFrame({'x': [None, None]})
    assert all(pd.isna(value) for value in ordinal_mapping(empty, 'x', ['a'])['x_ordinal'].to_list())
    with pytest.raises(ValueError):
        validate_mapping(['a'], [2**1000])
