import numpy as np
import pandas as pd
import pytest
from markowitz.data import synthetic_prices, validate_prices, load_prices
from markowitz.forecast import supervised_windows, ridge_forecast, neural_forecast


@pytest.mark.parametrize('kind', ['missing', 'negative', 'unsorted', 'duplicate'])
def test_invalid_prices(kind):
    prices = synthetic_prices(observations=30, assets=3)
    if kind == 'missing': prices.iloc[3, 0] = np.nan
    if kind == 'negative': prices.iloc[3, 0] = -1
    if kind == 'unsorted': prices = prices.iloc[::-1]
    if kind == 'duplicate': prices.index = pd.DatetimeIndex([prices.index[0]] * len(prices))
    with pytest.raises(ValueError): validate_prices(prices)


def test_duplicate_csv_header_rejected(tmp_path):
    p = tmp_path / 'bad.csv'
    p.write_text('date,A,A\n2020-01-01,1,1\n')
    with pytest.raises(ValueError): load_prices(p)


def test_windows_align_features_before_targets():
    x = np.arange(200).reshape(100, 2)
    features, targets = supervised_windows(x, lookback=5, horizon=3)
    np.testing.assert_equal(features[0], x[:5])
    np.testing.assert_equal(targets[0], x[5:8].mean(axis=0))
    np.testing.assert_equal(features[-1], x[-8:-3])
    np.testing.assert_equal(targets[-1], x[-3:].mean(axis=0))


def test_ridge_constant_returns_and_repeatability():
    x = np.tile([0.001, 0.002], (130, 1))
    np.testing.assert_allclose(ridge_forecast(x), [0.001, 0.002])
    np.testing.assert_array_equal(ridge_forecast(x), ridge_forecast(x))


@pytest.mark.parametrize('architecture', ['lstm', 'bilstm', 'cnn_bilstm'])
def test_neural_forecast_is_finite_and_reproducible(architecture):
    pytest.importorskip('torch')
    x = np.random.default_rng(1).normal(0, 0.01, (150, 3))
    first = neural_forecast(x, horizon=5, epochs=2, architecture=architecture)
    second = neural_forecast(x, horizon=5, epochs=2, architecture=architecture)
    assert first.shape == (3,)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, atol=1e-10)
