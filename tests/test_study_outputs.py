"""Validate published experiment coverage and independently recompute net returns."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/set50'


@pytest.mark.parametrize('model', ['lstm', 'bilstm', 'rcnn_bilstm'])
@pytest.mark.parametrize('k', range(5, 11))
def test_published_portfolio_outputs_reconcile(model, k):
    folder = OUT / model / f'k{k}'
    returns = pd.read_csv(folder / 'returns.csv', index_col='date')
    metrics = pd.read_csv(folder / 'metrics.csv', index_col=0)
    for strategy in returns:
        assert np.prod(1 + returns[strategy]) - 1 == pytest.approx(metrics.loc[strategy, 'total_return'])
        assert returns[strategy].mean() == pytest.approx(metrics.loc[strategy, 'mean_daily_return'])
    weights = pd.read_csv(folder / 'weights.csv')
    optimized = weights[weights.strategy != 'Equal weight'].drop(columns=['execution_date', 'strategy'])
    assert np.all((optimized.to_numpy() > 1e-7).sum(axis=1) <= k)
    np.testing.assert_allclose(optimized.sum(axis=1), 1, atol=1e-7)
    assert optimized.to_numpy().max() <= 0.35 + 1e-7
    trades = pd.read_csv(folder / 'trades.csv')
    targeted = trades[trades.strategy.isin(['Forecast Markowitz', 'Black-Litterman'])]
    assert (targeted.expected_return >= targeted.effective_target - 1e-7).all()


def test_real_study_input_and_forecast_coverage():
    manifest = json.loads((ROOT / 'data/set50/manifest.json').read_text())
    metadata = json.loads((OUT / 'metadata.json').read_text())
    assert metadata['sha256'] == manifest['cleaned_sha256']
    assert len(metadata['assets']) == 41
    assert set(metadata['portfolio_sizes']) == set(range(5, 11))
    for model in metadata['models']:
        forecasts = pd.read_csv(OUT / model / 'price_forecasts.csv')
        assert set(forecasts.asset) == set(metadata['assets'])
        assert (forecasts.predicted_price > 0).all()
        assert (forecasts.last_train_target_index < forecasts.first_validation_anchor_index).all()
        assert forecasts.groupby('decision_date').asset.nunique().eq(41).all()
