"""Check that public requirements describe supported observation layouts."""

from inspect import isfunction

import pandas as pd
import polars as pl
import pytest

from alpha_research.evaluation import (
    change_points,
    descriptive_statistics,
    economic_significance,
    ic,
    partial,
    relationship,
    statistical_tests,
    timeseries,
)
from alpha_research.evaluation.ic import compute_ic, information_coefficient
from alpha_research.evaluation.requirements import EVALUATION_REQUIREMENTS
from alpha_research.evaluation.timeseries import temporal_association
from alpha_research.resampling import block_bootstrap, convergence, null_models


def test_all_public_evaluation_and_resampling_functions_have_requirements():
    """Should require an explicit declaration for every exported scientific function."""
    modules = (change_points, descriptive_statistics, economic_significance, ic, partial,
               relationship, statistical_tests, timeseries, block_bootstrap, convergence, null_models)
    functions = {name for module in modules for name in module.__all__ if isfunction(getattr(module, name))}
    assert set(EVALUATION_REQUIREMENTS) == functions


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
def test_cross_sectional_numeric_axis(backend):
    """Should group exact numeric coordinates without requiring presorted input."""
    frame = backend({'temperature': [120, 100, 120, 100],
                     'curve': ['B', 'B', 'A', 'A'],
                     'signal': [3, 1, 1, 3], 'target': [1, 1, 3, 3]})
    requirement = EVALUATION_REQUIREMENTS['compute_ic']
    assert requirement.axis_order == 'sorted_by_calculation'
    result = compute_ic(frame, 'signal', 'target', date_column='temperature', symbol_column='curve')
    assert result['temperature'].to_list() == [100, 120]
    assert result['ic'].to_list() == pytest.approx([1, -1])
    duplicate = backend({'temperature': [100, 100], 'curve': ['A', 'A'],
                         'signal': [1, 3], 'target': [1, 3]})
    with pytest.raises(ValueError, match='unique|duplicate'):
        compute_ic(duplicate, 'signal', 'target', date_column='temperature', symbol_column='curve')


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
def test_axis_free_and_ordered_association(backend):
    """Should accept repeated measurement values while requiring a unique ordered sequence axis."""
    frame = backend({'temperature': [100, 100, 120, 120], 'curve': ['A'] * 4,
                     'position': [1, 2, 3, 4], 'pressure': [2, 3, 4, 6]})
    for method, expected in [('pearson', 5 / 35**0.5), ('spearman', 2 / 5**0.5)]:
        assert EVALUATION_REQUIREMENTS['information_coefficient'].axis_order == 'unused'
        assert information_coefficient(frame['temperature'], frame['pressure'], method) == pytest.approx(expected)
        assert temporal_association(frame, 'temperature', 'pressure', method,
                                    time_col='position', symbol_col='curve') == pytest.approx(expected)
    assert EVALUATION_REQUIREMENTS['temporal_association'].axis_order == 'strictly_increasing'
    with pytest.raises(ValueError, match='unique|duplicate'):
        temporal_association(frame, 'temperature', 'pressure', time_col='temperature', symbol_col='curve')
