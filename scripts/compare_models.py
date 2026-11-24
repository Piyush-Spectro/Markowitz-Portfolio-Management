"""Evaluate forecasting alternatives on the same dates and cost assumptions."""
import argparse
import hashlib
import json
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from markowitz.backtest import Config, run_backtest
from markowitz.data import load_prices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=Path, default=Path('data/sample_prices.csv'))
    parser.add_argument('--output', type=Path, default=Path('outputs/comparison'))
    parser.add_argument('--neural', action='store_true')
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    prices = load_prices(args.prices)
    methods = ['historical', 'ridge'] + (['lstm', 'bilstm', 'cnn_bilstm'] if args.neural else [])
    records = []
    args.output.mkdir(parents=True, exist_ok=True)
    for method in methods:
        print(f'Evaluating {method}...', flush=True)
        result = run_backtest(prices, Config(forecast_method=method, neural_epochs=args.epochs))
        folder = args.output / method
        folder.mkdir(exist_ok=True)
        for name in ['returns', 'weights', 'trades', 'forecasts']:
            getattr(result, name).to_csv(folder / f'{name}.csv', index=name == 'returns')
        (folder / 'config.json').write_text(json.dumps(result.config, indent=2) + '\n')
        error = result.forecasts.predicted_daily_return - result.forecasts.realized_daily_return
        rmse = np.sqrt(np.mean(error ** 2))
        zero_rmse = np.sqrt(np.mean(result.forecasts.realized_daily_return ** 2))
        for strategy, metrics in result.metrics.iterrows():
            records.append({'model': method, 'strategy': strategy, 'forecast_rmse': rmse,
                            'zero_forecast_rmse': zero_rmse, **metrics.to_dict()})
    table = pd.DataFrame(records)
    table.to_csv(args.output / 'model_comparison.csv', index=False)
    libraries = ['numpy', 'pandas', 'scipy'] + (['torch'] if args.neural else [])
    metadata = {'input_file': args.prices.name, 'sha256': hashlib.sha256(args.prices.read_bytes()).hexdigest(),
                'models': methods, 'epochs': args.epochs,
                'versions': {name: importlib.metadata.version(name) for name in libraries}}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
