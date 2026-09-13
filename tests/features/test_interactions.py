import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.features.interactions import (
    _validate_interaction_inputs,
    feature_difference,
    feature_product,
    feature_ratio,
    feature_where,
)


@pytest.fixture
def interaction_frame_pandas():
    """Create aligned observations for generic pairwise feature operations."""
    return pd.DataFrame({
        'time': [1, 2, 3, 4],
        'symbol': ['A', 'A', 'B', 'B'],
        'x': [2.0, 4.0, -3.0, np.nan],
        'z': [1.0, 2.0, 3.0, 4.0],
        'y': [1.0, 0.0, -1.0, np.nan],
    })


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_feature_ratio_computes_values_and_nulls_zero_denominators(
        interaction_frame_pandas,
        backend,
):
    """Should compute X/Y row-wise and avoid infinite zero-denominator output."""
    frame = (
        interaction_frame_pandas
        if backend == 'pandas'
        else pl.from_pandas(interaction_frame_pandas)
    )
    result = feature_ratio(frame, 'x', 'y')
    np.testing.assert_allclose(
        result['x_div_y'].to_numpy(),
        [2.0, np.nan, 3.0, np.nan],
        equal_nan=True,
    )


def test_feature_ratio_can_raise_on_zero_denominators(interaction_frame_pandas):
    """Should support a strict zero-denominator research policy."""
    with pytest.raises(ValueError, match='must not contain zero'):
        feature_ratio(interaction_frame_pandas, 'x', 'y', zero_policy='raise')


def test_feature_ratio_validates_zero_policy(interaction_frame_pandas):
    """Should reject unknown division-by-zero policies."""
    with pytest.raises(ValueError, match='zero_policy'):
        feature_ratio(interaction_frame_pandas, 'x', 'y', zero_policy='ignore')


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_feature_product_and_difference_are_directional_and_backend_equivalent(
        interaction_frame_pandas,
        backend,
):
    """Should calculate product and explicit X-minus-Y interactions row-wise."""
    frame = (
        interaction_frame_pandas
        if backend == 'pandas'
        else pl.from_pandas(interaction_frame_pandas)
    )
    product = feature_product(frame, 'x', 'y')
    difference = feature_difference(frame, 'x', 'y')

    np.testing.assert_allclose(
        product['x_mul_y'].to_numpy(),
        [2.0, 0.0, 3.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        difference['x_minus_y'].to_numpy(),
        [1.0, 4.0, -2.0, np.nan],
        equal_nan=True,
    )


def test_interactions_support_multiple_features_without_mutating_input(
        interaction_frame_pandas,
):
    """Should append one output per left feature and preserve the source frame."""
    original = interaction_frame_pandas.copy(deep=True)
    result = feature_product(interaction_frame_pandas, ['x', 'z'], 'y')

    assert {'x_mul_y', 'z_mul_y'} <= set(result.columns)
    pd.testing.assert_frame_equal(interaction_frame_pandas, original)


@pytest.mark.parametrize(
    'operator, expected',
    [
        ('greater', [0.0, 0.0, 0.0, 0.0]),
        ('greater_equal', [2.0, 0.0, 0.0, 0.0]),
        ('less', [0.0, 4.0, -3.0, 0.0]),
        ('less_equal', [2.0, 4.0, -3.0, 0.0]),
        ('equal', [2.0, 0.0, 0.0, 0.0]),
        ('not_equal', [0.0, 4.0, -3.0, 0.0]),
    ],
)
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_feature_where_supports_all_comparison_operators(
        interaction_frame_pandas,
        operator,
        expected,
        backend,
):
    """Should gate a feature with every supported fixed-threshold comparison."""
    frame = (
        interaction_frame_pandas
        if backend == 'pandas'
        else pl.from_pandas(interaction_frame_pandas)
    )
    result = feature_where(frame, 'x', 'y', operator, threshold=1)
    np.testing.assert_allclose(
        result[f'x_where_y_{operator}_1'].to_numpy(),
        expected,
        equal_nan=True,
    )


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_feature_where_can_use_missing_fallback(interaction_frame_pandas, backend):
    """Should emit missing values when otherwise is explicitly None."""
    frame = (
        interaction_frame_pandas
        if backend == 'pandas'
        else pl.from_pandas(interaction_frame_pandas)
    )
    result = feature_where(
        frame,
        'x',
        'y',
        'greater',
        threshold=0,
        otherwise=None,
    )
    expected = [2.0, np.nan, np.nan, np.nan]
    np.testing.assert_allclose(
        result['x_where_y_greater_0'].to_numpy(),
        expected,
        equal_nan=True,
    )


def test_feature_where_validates_operator_and_scalars(interaction_frame_pandas):
    """Should reject unknown conditions and invalid fixed scalar values."""
    with pytest.raises(ValueError, match='operator'):
        feature_where(interaction_frame_pandas, 'x', 'y', 'between', 0)
    with pytest.raises(TypeError, match='threshold must be numeric'):
        feature_where(interaction_frame_pandas, 'x', 'y', 'greater', True)
    with pytest.raises(ValueError, match='otherwise must be finite'):
        feature_where(
            interaction_frame_pandas,
            'x',
            'y',
            'greater',
            0,
            otherwise=np.inf,
        )


def test_validate_interaction_inputs_accepts_mixed_feature_provenance():
    """Should validate aligned rows without prohibiting intentional scope mixes."""
    frame = pd.DataFrame({
        'time': [1, 2],
        'symbol': ['A', 'A'],
        'x_cs_rank': [1.0, 2.0],
        'y_rolling_zscore': [-1.0, 1.0],
    })
    assert _validate_interaction_inputs(
        frame,
        'x_cs_rank',
        'y_rolling_zscore',
        'symbol',
        'time',
    ) == ['x_cs_rank']


@pytest.mark.parametrize(
    'mutator, error_type, message',
    [
        (lambda frame: frame.drop(columns='y'), KeyError, 'missing required'),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), ValueError, 'unique'),
        (lambda frame: frame.assign(time=[1, 2, 3, np.nan]), ValueError, 'time'),
        (lambda frame: frame.assign(y=['a', 'b', 'c', 'd']), ValueError, 'numeric'),
        (lambda frame: frame.assign(y=[1.0, 2.0, np.inf, 4.0]), ValueError, 'finite'),
    ],
)
def test_validate_interaction_inputs_rejects_invalid_frames(
        interaction_frame_pandas,
        mutator,
        error_type,
        message,
):
    """Should enforce schema, keys, numeric types, and finite-or-missing values."""
    with pytest.raises(error_type, match=message):
        _validate_interaction_inputs(
            mutator(interaction_frame_pandas),
            'x',
            'y',
            'symbol',
            'time',
        )


@pytest.mark.parametrize(
    'features, other, error_type, message',
    [
        (['x', 'x'], 'y', ValueError, 'duplicate'),
        ('x', 'x', ValueError, 'must differ'),
        (['x', ''], 'y', TypeError, 'non-empty'),
        ('x', '', TypeError, 'other_feature'),
    ],
)
def test_validate_interaction_inputs_rejects_invalid_names(
        interaction_frame_pandas,
        features,
        other,
        error_type,
        message,
):
    """Should reject ambiguous or malformed feature specifications."""
    with pytest.raises(error_type, match=message):
        _validate_interaction_inputs(
            interaction_frame_pandas,
            features,
            other,
            'symbol',
            'time',
        )


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_interaction_algebraic_invariants_hold_for_finite_rows(
        interaction_frame_pandas,
        backend,
):
    """Should preserve product commutativity and difference antisymmetry."""
    finite = interaction_frame_pandas.dropna().loc[
        interaction_frame_pandas['y'] != 0,
    ]
    frame = finite if backend == 'pandas' else pl.from_pandas(finite)

    xy_product = feature_product(frame, 'x', 'y')['x_mul_y'].to_numpy()
    yx_product = feature_product(frame, 'y', 'x')['y_mul_x'].to_numpy()
    xy_difference = feature_difference(frame, 'x', 'y')['x_minus_y'].to_numpy()
    yx_difference = feature_difference(frame, 'y', 'x')['y_minus_x'].to_numpy()
    ratio = feature_ratio(frame, 'x', 'y')['x_div_y'].to_numpy()

    np.testing.assert_allclose(xy_product, yx_product)
    np.testing.assert_allclose(xy_difference, -yx_difference)
    np.testing.assert_allclose(ratio * frame['y'].to_numpy(), frame['x'].to_numpy())


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_interactions_preserve_row_order_keys_and_pandas_index(
        interaction_frame_pandas,
        backend,
):
    """Should append values without sorting or changing observation identity."""
    shuffled = interaction_frame_pandas.iloc[[2, 0, 3, 1]].copy()
    frame = shuffled if backend == 'pandas' else pl.from_pandas(shuffled)
    result = feature_difference(frame, 'x', 'y')

    assert result['time'].to_list() == shuffled['time'].tolist()
    assert result['symbol'].to_list() == shuffled['symbol'].tolist()
    if backend == 'pandas':
        assert result.index.tolist() == shuffled.index.tolist()


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_interactions_preserve_missing_operands(backend):
    """Should propagate a missing arithmetic operand in both backends."""
    pandas_frame = pd.DataFrame({
        'time': [1, 2],
        'symbol': ['A', 'A'],
        'x': [np.nan, 2.0],
        'y': [1.0, np.nan],
    })
    frame = pandas_frame if backend == 'pandas' else pl.from_pandas(pandas_frame)

    product = feature_product(frame, 'x', 'y')['x_mul_y'].to_numpy()
    difference = feature_difference(frame, 'x', 'y')['x_minus_y'].to_numpy()
    ratio = feature_ratio(frame, 'x', 'y')['x_div_y'].to_numpy()
    for values in [product, difference, ratio]:
        assert np.isnan(values).all()


def test_feature_ratio_strict_policy_ignores_missing_denominators():
    """Should reserve strict failures for observed zeros, not missing values."""
    frame = pd.DataFrame({
        'time': [1, 2],
        'symbol': ['A', 'A'],
        'x': [1.0, 2.0],
        'y': [np.nan, 2.0],
    })
    result = feature_ratio(frame, 'x', 'y', zero_policy='raise')
    np.testing.assert_allclose(result['x_div_y'], [np.nan, 1.0], equal_nan=True)


def test_feature_where_does_not_depend_on_other_rows(
        interaction_frame_pandas,
):
    """Should leave an observation unchanged when only later rows are modified."""
    original = feature_where(
        interaction_frame_pandas,
        'x',
        'y',
        'greater',
        0,
    )
    changed_frame = interaction_frame_pandas.copy()
    changed_frame.loc[1:, 'y'] = 10_000.0
    changed = feature_where(changed_frame, 'x', 'y', 'greater', 0)

    assert original.loc[0, 'x_where_y_greater_0'] == changed.loc[
        0,
        'x_where_y_greater_0',
    ]


@pytest.mark.parametrize(
    'mutator, message',
    [
        (lambda frame: frame.assign(x=True), 'real numeric'),
        (lambda frame: frame.assign(x=1 + 2j), 'real numeric'),
        (
            lambda frame: pd.concat(
                [frame, frame['x']],
                axis=1,
            ).rename(columns={'x': 'y'}),
            'duplicate column',
        ),
    ],
)
def test_interactions_reject_ambiguous_or_non_real_columns(
        interaction_frame_pandas,
        mutator,
        message,
):
    """Should reject data that cannot satisfy a real-valued feature contract."""
    with pytest.raises(ValueError, match=message):
        feature_product(mutator(interaction_frame_pandas), 'x', 'y')


@pytest.mark.parametrize(
    'features, other, symbol_col, time_col, message',
    [
        ('x', 'y', 'symbol', 'symbol', 'different'),
        ('time', 'y', 'symbol', 'time', 'key columns'),
        ('x', 'symbol', 'symbol', 'time', 'key columns'),
    ],
)
def test_interactions_reject_key_columns_as_feature_operands(
        interaction_frame_pandas,
        features,
        other,
        symbol_col,
        time_col,
        message,
):
    """Should keep observation keys outside arithmetic feature operations."""
    with pytest.raises(ValueError, match=message):
        feature_product(
            interaction_frame_pandas,
            features,
            other,
            symbol_col=symbol_col,
            time_col=time_col,
        )


def test_interactions_validate_duplicate_polars_observation_keys(
        interaction_frame_pandas,
):
    """Should reject duplicate time-symbol observations natively in Polars."""
    duplicate = pd.concat([
        interaction_frame_pandas,
        interaction_frame_pandas.iloc[[0]],
    ])
    with pytest.raises(ValueError, match='unique'):
        feature_product(pl.from_pandas(duplicate), 'x', 'y')


def test_interactions_validate_boolean_polars_features(
        interaction_frame_pandas,
):
    """Should reject Boolean operands consistently in Polars."""
    frame = pl.from_pandas(interaction_frame_pandas).with_columns(
        pl.lit(True).alias('x'),
    )
    with pytest.raises(ValueError, match='numeric'):
        feature_product(frame, 'x', 'y')
