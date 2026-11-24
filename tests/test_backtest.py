import numpy as np
import pandas as pd
import pytest
from markowitz.data import synthetic_prices, returns_from_prices
from markowitz.backtest import Config, run_backtest, performance


def test_future_prices_cannot_change_earlier_weights():
    prices = synthetic_prices(observations=240, assets=4)
    config = Config(train_window=120, rebalance=21, max_weight=0.5)
    original = run_backtest(prices, config)
    changed = prices.copy()
    changed.iloc[180:] *= np.linspace(1, 2, len(changed) - 180)[:, None]
    modified = run_backtest(changed, config)
    cutoff = prices.index[180]
    pd.testing.assert_frame_equal(original.weights[original.weights.execution_date < cutoff],
                                  modified.weights[modified.weights.execution_date < cutoff])
    pd.testing.assert_frame_equal(original.returns.loc[:prices.index[179]],
                                  modified.returns.loc[:prices.index[179]])


def test_cost_accounting_and_drift_match_manual_equal_weight():
    prices = synthetic_prices(observations=180, assets=4)
    config = Config(train_window=120, rebalance=21, cost_bps=20, max_weight=0.5)
    result = run_backtest(prices, config)
    a = returns_from_prices(prices).to_numpy()
    start = config.train_window + 1
    w = np.full(4, 0.25)
    expected = []
    for i in range(start, start + 21):
        gross = w @ a[i]
        expected.append((1 - (0.002 if i == start else 0)) * (1 + gross) - 1)
        w = w * (1 + a[i]) / (1 + gross)
    np.testing.assert_allclose(result.returns['Equal weight'].iloc[:21], expected)
    trades = result.trades[result.trades.strategy == 'Equal weight']
    assert trades.iloc[0].gross_traded_fraction == pytest.approx(1)
    assert trades.iloc[1].gross_traded_fraction == pytest.approx(abs(w - 0.25).sum())


def test_costs_reduce_terminal_wealth():
    prices = synthetic_prices(observations=200, assets=4)
    free = run_backtest(prices, Config(train_window=120, max_weight=0.5, cost_bps=0))
    costly = run_backtest(prices, Config(train_window=120, max_weight=0.5, cost_bps=50))
    assert ((1 + costly.returns).prod() < (1 + free.returns).prod()).all()


def test_first_day_drawdown_includes_initial_capital():
    assert performance([-0.1, 0.0])['max_drawdown'] == pytest.approx(-0.1)


def test_zero_volatility_sharpe_is_undefined():
    assert np.isnan(performance([0, 0, 0])['sharpe'])


def test_top_k_and_cap_are_respected():
    prices = synthetic_prices(observations=200, assets=6)
    result = run_backtest(prices, Config(train_window=120, top_k=3, max_weight=0.5))
    w = result.weights[result.weights.strategy != 'Equal weight'].filter(like='Asset_').to_numpy()
    np.testing.assert_allclose(w.sum(axis=1), 1)
    assert np.all((w > 1e-7).sum(axis=1) <= 3)
    assert w.max() <= 0.5 + 1e-7


def test_execution_date_follows_decision_and_precedes_pnl():
    prices = synthetic_prices(observations=180, assets=4)
    result = run_backtest(prices, Config(train_window=120, max_weight=0.5))
    first = result.forecasts.iloc[0]
    assert first.decision_date < first.execution_date < result.returns.index[0]


@pytest.mark.parametrize('top_k', [5, 6, 7, 8])
def test_full_sample_selected_universe_solver_regression(top_k):
    # The direct Sharpe-ratio objective previously failed in this sensitivity sweep.
    result = run_backtest(synthetic_prices(), Config(top_k=top_k))
    w = result.weights.filter(like='Asset_').to_numpy()
    assert np.isfinite(result.returns.to_numpy()).all()
    np.testing.assert_allclose(w.sum(axis=1), 1, atol=1e-7)
    assert w.max() <= 0.35 + 1e-7
