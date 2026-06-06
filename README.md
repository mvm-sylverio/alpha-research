# alpha-research
Systematic alpha research framework focused on signal discovery, validation, and robustness.

Inspired by concepts from:

- *Active Portfolio Management* — Grinold & Kahn
- *Advances in Financial Machine Learning* — Marcos López de Prado
- *Machine Learning for Asset Managers* — Marcos López de Prado

## Stack

Python - Pandas - Polars - NumPy - Scikit-learn

Both Pandas and Polars are accept as backends. Pandas is accepted because of its widespread usage in the community and Polars for its performance for bigger tasks. Pandas is adopted without index, like polars.

## Pipeline
Completed modules are marked; remaining items reflect the planned roadmap.

### Pre-Research
- [ ] Data ingestion and cleaning
- [ ] Volume bars / dollar bars
- [ ] Exploratory data analysis (EDA)
- [ ] Feature engineering
- [ ] Outlier winsorization
- [ ] Stationarity testing (ADF)
- [ ] Fractional differentiation for non-stationary features
- [ ] Triple Barrier Method (labeling)
- [ ] Event-based sampling (CUSUM filter)

### Research — "Does the signal exist?"

**Causality — "Is the signal real or spurious?"**

Fully exploratory. Runs on the full dataset before any fold construction.
- [ ] Granger causality — linear filter
- [ ] PCMCI — causal discovery with multiple confounders
- [ ] Transfer entropy — non-linear relationships

**Purged K-Fold construction**
- [ ] K-fold split with purging and embargo (López de Prado)
- [ ] Embargo length defined by IC decay horizon
- [ ] All decision-informing analyses restricted to train folds

**Foundation — IC Analysis**
- [x] Cross-sectional Information Coefficient (IC) — Spearman and Pearson
- [x] IC metrics: mean, stability (IC IR), pct positive, quantiles, signal alignment
- [x] Newey-West HAC t-statistic for IC significance
- [x] IC summary table across feature universe with feature group classification
- [ ] Rolling IC stability
- [ ] IC decay analysis

**Multiple Testing Correction**
- [ ] Benjamini-Hochberg correction per signal family

**Structure — "Are features redundant?"**
- [ ] Feature correlation matrix
- [ ] Feature clustering (hierarchical, distance-based)
- [ ] PCA / orthogonalization (optional, for multicollinearity)

**Importance — "Which features really matter?"**
- [ ] Single Feature Importance (SFI)
- [ ] Mean Decrease Impurity (MDI)
- [ ] Mean Decrease Accuracy / Permutation Importance (MDA)

**Interpretability — "Why do they matter and when?"**
- [ ] SHAP values
- [ ] Rolling SHAP for regime detection

**Regime Detection**
- [ ] Hidden Markov Model (HMM) — latent volatility/return states
- [ ] Changepoint detection — structural distribution shifts
- [ ] Regime clustering (K-means on volatility and correlation features)
- [ ] Regime as meta-feature entering Purged K-Fold

**Advanced**
- [ ] Signature features (Rough Path Theory)
- [ ] Topological Data Analysis (TDA) — persistent homology
- [ ] Hawkes processes — temporal event clustering

### Backtesting
- [ ] Cross-sectional portfolio construction
- [ ] Metalabeling
- [ ] Position sizing
- [ ] Simulated P&L
- [ ] Turnover analysis
- [ ] Alpha evaluation
- [ ] Sharpe ratio
- [ ] Deflated Sharpe Ratio (DSR)

## Usage

```python
import pandas as pd
from alpha_research.evaluation.ic import compute_ic, ic_summary_table

df = pd.DataFrame()  # or pl.DataFrame

# Compute IC time series for a single feature
ic_series = compute_ic(df, feature="mom_5_rank", target="fwd_ret_10")

# Summarize IC across multiple features with group classification
feature_groups = {
    "mom_5_rank": "momentum",
    "mom_21_rank": "momentum",
    "rsi_14_rank": "mean_reversion",
}

table = ic_summary_table(
    df,
    feature_list=["mom_5_rank", "mom_21_rank", "rsi_14_rank"],
    target="fwd_ret_10",
    feature_groups=feature_groups,
)
```