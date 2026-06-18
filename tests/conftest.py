import pytest
import pandas as pd


# ------------------------------------------------------
# shared fixtures
# ------------------------------------------------------
@pytest.fixture
def single_asset_ohlcv_pandas():
    """
    Single asset, 5 bars.
    close = [100, 110, 121, 133.1, 146.41]

    Precomputed simple_ret_1:
        bar 0: NaN (no previous)
        bar 1: 110/100 - 1 = 0.10
        bar 2: 121/110 - 1 = 0.10
        bar 3: 133.1/121 - 1 = 0.10
        bar 4: 146.41/133.1 - 1 = 0.10

    Precomputed simple_ret_2:
        bar 0: NaN
        bar 1: NaN
        bar 2: 121/100 - 1 = 0.21
        bar 3: 133.1/110 - 1 = 0.21
        bar 4: 146.41/121 - 1 = 0.21

    Precomputed log_ret_1:
        bar 0: NaN
        bar 1: ln(110/100) = ln(1.10) ≈ 0.09531
        bar 2: ln(121/110) = ln(1.10) ≈ 0.09531
        bar 3: ln(133.1/121) = ln(1.10) ≈ 0.09531
        bar 4: ln(146.41/133.1) = ln(1.10) ≈ 0.09531

    Precomputed log_ret_2:
        bar 0: NaN
        bar 1: NaN
        bar 2: ln(121/100) = ln(1.21) ≈ 0.19062
        bar 3: ln(133.1/110) ≈ 0.19062
        bar 4: ln(146.41/121) ≈ 0.19062

    fwd_ret_1:
        bar 0: 110/100 - 1 = 0.10
        bar 1: 121/110 - 1 = 0.10
        bar 2: 133.1/121 - 1 = 0.10
        bar 3: 146.41/133.1 - 1 = 0.10
        bar 4: NaN (no future return)

    fwd_ret_2:
        bar 0: 121/100 - 1 = 0.21
        bar 1: 133.1/110 - 1 = 0.21
        bar 2: 146.41/121 - 1 = 0.21
        bar 3: NaN
        bar 4: NaN

    sma_3:
        bar 0: NaN
        bar 1: NaN
        bar 2: (100+110+121)/3 = 110.333
        bar 3: (110+121+133.1)/3 = 121.367
        bar 4: (121+133.1+146.41)/3 = 133.503

    price_to_sma_ratio_3:
        bar 0: NaN
        bar 1: NaN
        bar 2: 121/110.333 - 1 ≈ 0.09667
        bar 3: 133.1/121.367 - 1 ≈ 0.09667
        bar 4: 146.41/133.503 - 1 ≈ 0.09667

    sma_2: [NaN, 105.0, 115.5, 127.05, 139.755]
    sma_3: [NaN, NaN, 110.333, 121.367, 133.503]
    sma_2_crossover_sma_3:
        bar 0: NaN
        bar 1: NaN
        bar 2: 115.5/110.333 - 1 ≈ 0.046827
        bar 3: 127.05/121.367 - 1 ≈ 0.046827
        bar 4: 139.755/133.503 - 1 ≈ 0.046827
    """
    return pd.DataFrame({
        'time': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'symbol': ['AAPL'] * 5,
        'close': [100.0, 110.0, 121.0, 133.1, 146.41],
    })


@pytest.fixture
def multi_asset_ohlcv_pandas():
    """
    Two assets, 3 bars each — interleaved.
    AAPL close = [100, 110, 121]
    MSFT close = [200, 210, 220]

    Precomputed ret_1:
        AAPL: [NaN, 0.10, 0.10]
        MSFT: [NaN, 0.05, 0.04762]

    Precomputed ret_2:
        AAPL: [NaN, NaN, 0.21]
        MSFT: [NaN, NaN, 0.10]

    Precomputed log_ret_1:
        AAPL: [NaN, ln(1.10)≈0.09531, ln(1.10)≈0.09531]
        MSFT: [NaN, ln(210/200)≈0.04879, ln(220/210)≈0.04652]

    Precomputed log_ret_2:
        AAPL: [NaN, NaN, ln(121/100)≈0.19062]
        MSFT: [NaN, NaN, ln(220/200)≈0.09531]

    Precomputed fwd_ret_1:
        AAPL: [0.10, 0.10, NaN]
        MSFT: [0.05, 0.047619, NaN]

    AAPL sma and ratio:
        sma_2:    [NaN, 105.0, 115.5]
        ratio_2:  [NaN, 110/105-1≈0.047619, 121/115.5-1≈0.047619]

    MSFT sma and ratio:
        sma_2:    [NaN, 205.0, 215.0]
        ratio_2:  [NaN, 210/205-1≈0.024390, 220/215-1≈0.023256]

    AAPL sma_2: [NaN, 105.0, 115.5]
    AAPL sma_3: [NaN, NaN, 110.333]
    AAPL crossover: [NaN, NaN, 115.5/110.333-1 ≈ 0.046827]

    MSFT sma_2: [NaN, 205.0, 215.0]
    MSFT sma_3: [NaN, NaN, 210.0]
    MSFT crossover: [NaN, NaN, 215.0/210.0-1 ≈ 0.023810]

    Critical: groupby must not mix assets.
    """
    return pd.DataFrame({
        'time': ['2024-01-01', '2024-01-01',
                 '2024-01-02', '2024-01-02',
                 '2024-01-03', '2024-01-03'],
        'symbol': ['AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL', 'MSFT'],
        'close': [100.0, 200.0, 110.0, 210.0, 121.0, 220.0],
    })
