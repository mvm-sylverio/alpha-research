# alpha-research

`alpha-research` is a reusable Python library for quantitative signal research.
It provides feature and target construction, cross-sectional IC analysis,
single-asset temporal association diagnostics, and dependency-aware bootstrap
utilities. It is intentionally independent of databases, orchestration, and
backtesting applications.

The library supports both Pandas and Polars. Public functions preserve the
input DataFrame backend whenever practical.

## What it provides

- Feature and target utilities: simple/log returns, forward returns, trend
  features, volatility measures including realized volatility and ATR, ADX,
  and cross-sectional ranking.
- Research data utilities: feature/target schema helpers and purged
  train/test splitting.
- Cross-sectional information coefficient analysis with Pearson or Spearman
  estimators, IC metrics, Newey-West statistics, FDR correction, and IC decay.
- Single-asset temporal association with Moving Block Bootstrap (MBB), Wald
  diagnostics, and bootstrap directional stability.
- Rolling temporal association with percentile bootstrap bands.
- Visualization utilities for rolling temporal association, time-series feature
  values, and cross-sectional feature summaries.
- Resampling utilities for moving-block bootstrap summaries and bootstrap
  Monte Carlo convergence diagnostics.
- ADF stationarity testing.

## Installation

Install the library and its core dependencies:

```bash
pip install -e .
```

## Cross-sectional IC

Cross-sectional IC measures the association between a feature and a forward
target across assets at each timestamp. The example below creates one feature,
joins it to a forward-return target, calculates IC diagnostics, and applies FDR
correction to the resulting feature family.

```python
from alpha_research.evaluation.ic import ic_summary_table
from alpha_research.evaluation.statistical_tests import fdr_correction
from alpha_research.features.targets import fwd_returns
from alpha_research.features.trend import price_to_sma_ratio

# ohlcv contains: time, symbol, close
feature_frame = price_to_sma_ratio(ohlcv, window=20)
target_frame = fwd_returns(ohlcv, horizon=10)

research_frame = (
    feature_frame
    .merge(target_frame, on=['time', 'symbol'])
    .dropna()
)

ic_result = ic_summary_table(
    research_frame,
    feature_list=['price_to_sma_ratio_20'],
    target='fwd_ret_10',
    feature_groups={'price_to_sma_ratio_20': 'trend'},
)
ic_table = fdr_correction(ic_result.table, method='bh')
```

`ic_result.ic_frames` retains the per-date IC series used by the summary.

## Single-asset temporal association

Temporal association measures feature-target correlation through time for one
asset. It is distinct from cross-sectional IC: each observation is a paired
feature and aligned target at one point in time.

The global summary uses MBB, so feature and target are resampled together in
contiguous blocks rather than as IID observations.

```python
from alpha_research.evaluation.timeseries import temporal_association_summary_table

# single_asset_frame contains one symbol and: time, symbol, feature, fwd_ret_20
summary = temporal_association_summary_table(
    single_asset_frame,
    feature_list=['feature'],
    target='fwd_ret_20',
    block_length=20,
    n_bootstraps=2_000,
    corr_method='spearman',
    random_state=42,
)
```

The bootstrap sign proportion is a directional-stability diagnostic. It is not
a p-value. The Wald statistics in this global summary are separate from the
percentile bootstrap intervals used by the rolling analysis below.

## Rolling temporal association

`rolling_temporal_association()` evaluates local temporal association in full,
strict windows. A window with a missing feature-target pair is recorded with a
status instead of compressing the time axis before block bootstrap.

```python
from alpha_research.evaluation.timeseries import (
    plot_rolling_temporal_association,
    rolling_temporal_association,
)

rolling_result = rolling_temporal_association(
    single_asset_frame,
    feature='feature',
    target='fwd_ret_20',
    window_size=252,
    window_step=20,
    block_length=20,
    n_bootstraps=2_000,
    bootstrap_method='moving_block',
    confidence_level=0.95,
    random_state=42,
)

rolling_frame = rolling_result.rolling_frame
rolling_summary = rolling_result.summary_table
ax = plot_rolling_temporal_association(rolling_frame)
```

Each rolling row contains the observed association, percentile bootstrap bounds,
bootstrap directional stability, effective bootstrap count, and a computation
status. The plot function accepts `ax=...`, allowing an application to compose
the association panel with its own shared-time context panels.

## Scope

This project documents implemented research capabilities rather than a fixed
roadmap. New methods are added when they fit the library's statistical and
architectural boundaries.

## References

- *Active Portfolio Management* — Grinold & Kahn
- *Advances in Financial Machine Learning* — Marcos López de Prado
- *Machine Learning for Asset Managers* — Marcos López de Prado
