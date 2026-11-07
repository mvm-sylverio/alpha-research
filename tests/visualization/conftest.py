import pytest


@pytest.fixture
def plot_axis():
    """Should provide an Agg axis and release its figure after each test."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    try:
        yield axis
    finally:
        plt.close(figure)
