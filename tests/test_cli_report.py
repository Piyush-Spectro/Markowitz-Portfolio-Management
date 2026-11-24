import json
import pandas as pd
import pytest
from markowitz.cli import main
from markowitz.backtest import Config
from markowitz.data import synthetic_prices


def test_cli_writes_complete_report(tmp_path):
    source = tmp_path / 'prices.csv'
    synthetic_prices(observations=180, assets=4).to_csv(source)
    destination = tmp_path / 'report'
    main(['--prices', str(source), '--output', str(destination), '--train-window', '120', '--max-weight', '0.5'])
    expected = {'report.html', 'overview.png', 'returns.csv', 'weights.csv', 'trades.csv',
                'forecasts.csv', 'metrics.csv', 'frontier.csv', 'metadata.json'}
    assert expected == {p.name for p in destination.iterdir()}
    metadata = json.loads((destination / 'metadata.json').read_text())
    assert metadata['config']['train_window'] == 120
    assert len(metadata['sha256']) == 64
    assert 'Synthetic' in (destination / 'report.html').read_text()
    assert not pd.read_csv(destination / 'returns.csv').empty


def test_cli_reports_bad_input_without_traceback(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--prices', str(tmp_path / 'missing.csv')])
    assert exc.value.code == 2
    assert 'Error:' in capsys.readouterr().err


@pytest.mark.parametrize('config', [Config(cost_bps=-1), Config(top_k=1), Config(max_weight=0.1),
                                    Config(risk_free=float('nan')), Config(rebalance=300)])
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        config.validate(4, 1000)
