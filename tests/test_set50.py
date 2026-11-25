import numpy as np
import pandas as pd
import pytest
from markowitz.market_data import clean_panel, chart_to_prices
from markowitz.data import synthetic_prices
from markowitz.price_forecast import price_windows, predict_prices, predict_from_checkpoint, build_price_network
from markowitz.backtest import Config, run_backtest, performance


def test_cleaner_fills_only_short_internal_gaps_and_logs_exclusions():
    p = synthetic_prices(observations=200, assets=13)
    p.iloc[40:42, 0] = np.nan
    p.iloc[:2, 1] = np.nan
    p.iloc[40:45, 2] = np.nan
    clean, audit = clean_panel(p)
    assert clean.shape[1] == 11
    assert clean.iloc[40, 0] == clean.iloc[39, 0]
    assert audit.set_index('symbol').loc['Asset_01', 'forward_filled'] == 2
    assert 'Asset_02' not in clean and 'Asset_03' not in clean
    assert np.isfinite(clean.to_numpy()).all()


def test_cleaner_does_not_backfill_or_extend_a_delisted_series():
    p = synthetic_prices(observations=200, assets=12)
    p.iloc[-1, 0] = np.nan
    p.iloc[0, 1] = np.nan
    clean, audit = clean_panel(p)
    assert len(clean.columns) == 10
    assert (audit.exclusion_reason.iloc[:2] == 'missing_start_or_end').all()


def test_price_windows_use_terminal_price_after_execution_delay():
    p = np.exp(np.arange(240).reshape(120, 2) / 100)
    x, y, anchors = price_windows(p, lookback=5, horizon=3, execution_delay=1)
    assert anchors[0] == 4
    np.testing.assert_allclose(x[0], np.log(p[:5] / p[4]))
    np.testing.assert_allclose(y[0], np.log(p[8] / p[4]))
    assert anchors[-1] + 4 == len(p) - 1


@pytest.mark.parametrize('architecture', ['lstm', 'bilstm', 'rcnn_bilstm'])
def test_price_network_checkpoint_replays_and_validation_is_purged(architecture):
    pytest.importorskip('torch')
    p = synthetic_prices(observations=180, assets=4).to_numpy()
    fitted = predict_prices(p, architecture, epochs=2, keep_checkpoint=True)
    assert fitted.audit['last_train_target_index'] < fitted.audit['first_validation_anchor_index']
    assert np.all(fitted.prices > 0)
    np.testing.assert_allclose(predict_from_checkpoint(fitted.checkpoint, p), fitted.prices)


def test_recurrent_convolution_reuses_weights_and_receives_gradients():
    torch = pytest.importorskip('torch')
    model = build_price_network(4)
    output = model(torch.randn(3, 20, 4))
    output.sum().backward()
    assert model.rcnn.recurrent_conv.weight.grad is not None
    assert float(model.rcnn.recurrent_conv.weight.grad.abs().sum()) > 0
    assert output.shape == (3, 4)


def test_price_target_allocation_and_clipping_are_auditable():
    p = synthetic_prices(observations=200, assets=10)
    config = Config(train_window=120, forecast_method='historical', forecast_target='price',
                    allocation='target_return', target_return=10.0, top_k=5)
    result = run_backtest(p, config)
    targeted = result.trades[result.trades.strategy.isin(['Forecast Markowitz', 'Black-Litterman'])]
    assert targeted.target_clipped.any()
    assert (targeted.expected_return >= targeted.effective_target - 1e-7).all()
    full = result.forecasts[result.forecasts.full_forecast_horizon]
    assert np.all(full.predicted_price > 0)
    # First 21 returns end at price index 142 for train_window=120.
    first = result.forecasts.iloc[0]
    assert first.realized_price == pytest.approx(p.iloc[142, 0])
    assert first.anchor_price == pytest.approx(p.iloc[120, 0])


def test_price_forecast_cache_does_not_leak_changed_future_data():
    p = synthetic_prices(observations=220, assets=10)
    config = Config(train_window=120, forecast_method='historical', forecast_target='price', top_k=5)
    cache = {}
    original = run_backtest(p, config, cache)
    changed = p.copy()
    changed.iloc[180:] *= 2
    result = run_backtest(changed, config, cache)
    pd.testing.assert_frame_equal(original.weights[original.weights.execution_date < p.index[180]],
                                  result.weights[result.weights.execution_date < p.index[180]])


def test_explicit_mean_return_metrics():
    result = performance([0.01, -0.02, 0.03])
    assert result['mean_daily_return'] == pytest.approx(0.02 / 3)
    assert result['mean_annual_return'] == pytest.approx(0.02 / 3 * 252)


def test_universe_has_fifty_unique_historical_members():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / 'data/set50/universe_2015.csv'
    universe = pd.read_csv(p)
    assert len(universe) == universe.symbol.nunique() == 50
    assert {'BGH', 'TUF', 'TMB'} <= set(universe.symbol)
