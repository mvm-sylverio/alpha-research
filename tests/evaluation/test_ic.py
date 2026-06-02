import pytest
import numpy as np
import pandas as pd
import polars as pl
from alpha_research.evaluation.ic import information_coefficient, compute_ic


# ── fixtures ────────────────────────────────────────────────
@pytest.fixture
def perfect_corr():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    return x, x.copy()

@pytest.fixture
def perfect_inverse():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
    return x, y

@pytest.fixture
def cross_section_df_pandas():
    """
    Two dates, with 5 assets each.
    date 1: perfect correlation
    date 2: perfect inverse correlation
    """
    return pd.DataFrame({
        'time': ['2024-01-01'] * 5 + ['2024-01-02'] * 5,
        'feature': [1.0, 2.0, 3.0, 4.0, 5.0,
                    1.0, 2.0, 3.0, 4.0, 5.0],
        'target':  [1.0, 2.0, 3.0, 4.0, 5.0,
                    5.0, 4.0, 3.0, 2.0, 1.0],
    })

@pytest.fixture
def cross_section_df_polars():
    return pl.DataFrame({
        'time': [
            '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01',
            '2024-01-02', '2024-01-02', '2024-01-02', '2024-01-02', '2024-01-02',
        ],
        'feature': [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        'target':  [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    }).with_columns(pl.col('time').str.to_datetime())


# ── information_coefficient ──────────────────────────────────
def test_ic_perfect_positive(perfect_corr):
    """Should return 1.0 for perfect positive correlation with spearman corr_method."""
    x, y = perfect_corr
    assert information_coefficient(x, y) == pytest.approx(1.0)

def test_ic_perfect_inverse(perfect_inverse):
    """Should return -1.0 for perfect inverse correlation with spearman corr_method."""
    x, y = perfect_inverse
    assert information_coefficient(x, y) == pytest.approx(-1.0)

def test_ic_pearson(perfect_corr):
    """Should return 1.0 for perfect positive correlation with pearson corr_method."""
    x, y = perfect_corr
    assert information_coefficient(x, y, corr_method='pearson') == pytest.approx(1.0)

def test_ic_length_mismatch_raises():
    """Should raise ValueError on length mismatch."""
    with pytest.raises(ValueError, match="len"):
        information_coefficient(pd.Series([1, 2]), pd.Series([1]))

def test_ic_invalid_method_raises():
    """Should raise ValueError for unsupported corr_method."""
    x = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="corr_method"):
        information_coefficient(x, x, corr_method='kendall')


# ── compute_ic ───────────────────────────────────────────────
def test_compute_ic_pandas_shape(cross_section_df_pandas):
    """Should return one IC value per date and ic should be one of the two columns."""
    result = compute_ic(cross_section_df_pandas, 'feature', 'target')
    assert result.shape == (2, 2)
    assert 'ic' in result.columns

def test_compute_ic_pandas_values(cross_section_df_pandas):
    """Should return perfect positive correlation in first date and perfect inverse correlation in second date."""
    result = compute_ic(cross_section_df_pandas, 'feature', 'target')
    ic_values = result.set_index('time')['ic']
    assert ic_values['2024-01-01'] == pytest.approx(1.0)
    assert ic_values['2024-01-02'] == pytest.approx(-1.0)

def test_compute_ic_polars_shape(cross_section_df_polars):
    """Should return shape (2,2) because cross_section_df_polars has two dates."""
    result = compute_ic(cross_section_df_polars, 'feature', 'target')
    assert result.shape == (2, 2)

def test_compute_ic_polars_values(cross_section_df_polars):
    """Should return perfect positive correlation in first date and perfect inverse correlation in second date."""
    result = compute_ic(cross_section_df_polars, 'feature', 'target')
    result_sorted = result.sort('time')
    assert result_sorted['ic'][0] == pytest.approx(1.0)
    assert result_sorted['ic'][1] == pytest.approx(-1.0)

def test_compute_ic_pandas_polars_consistency(cross_section_df_pandas, cross_section_df_polars):
    """Backends should return numerically identical IC values."""
    res_pd = compute_ic(cross_section_df_pandas, 'feature', 'target').sort_values('time')
    res_pl = compute_ic(cross_section_df_polars, 'feature', 'target').sort('time')
    np.testing.assert_allclose(
        res_pd['ic'].values,
        res_pl['ic'].to_numpy(),
        rtol=1e-6
    )

def test_compute_ic_missing_column_raises(cross_section_df_pandas):
    """Should raise KeyError for missing columns."""
    with pytest.raises(KeyError):
        compute_ic(cross_section_df_pandas, 'nonexistent', 'target')

def test_compute_ic_invalid_df_raises():
    """Should raise ValueError with unsupported input types."""
    with pytest.raises(ValueError, match="Pandas or Polars"):
        compute_ic([[1, 2], [3, 4]], 'feature', 'target')

def test_compute_ic_custom_ic_column(cross_section_df_pandas):
    """Should return custom ic column name correctly."""
    result = compute_ic(cross_section_df_pandas, 'feature', 'target', ic_column='my_ic')
    assert 'my_ic' in result.columns