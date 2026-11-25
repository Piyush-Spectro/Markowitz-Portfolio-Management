"""Run every proposal model and portfolio size on the downloaded SET50 cohort."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from markowitz.backtest import Config, run_backtest
from markowitz.data import load_prices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=Path, default=Path('data/set50/adjusted_prices.csv'))
    parser.add_argument('--output', type=Path, default=Path('reports/set50'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--target-return', type=float, default=0.08)
    args = parser.parse_args()
    prices = load_prices(args.prices)
    if prices.shape[1] < 10:
        raise ValueError('All six experiments require at least ten assets')
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, price_metrics = [], []
    for model in ['lstm', 'bilstm', 'rcnn_bilstm']:
        print(f'Training {model} across rolling windows...', flush=True)
        cache = {}
        for k in range(5, 11):
            config = Config(forecast_target='price', forecast_method=model, neural_epochs=args.epochs,
                            allocation='target_return', target_return=args.target_return, top_k=k)
            result = run_backtest(prices, config, forecast_cache=cache)
            folder = args.output / model / f'k{k}'
            folder.mkdir(parents=True, exist_ok=True)
            for name in ['returns', 'weights', 'trades', 'metrics']:
                getattr(result, name).to_csv(folder / f'{name}.csv', index=name in {'returns', 'metrics'})
            (folder / 'config.json').write_text(json.dumps(asdict(config), indent=2) + '\n')
            for strategy, row in result.metrics.iterrows():
                trades = result.trades[result.trades.strategy == strategy]
                metrics.append({'model': model, 'top_k': k, 'strategy': strategy, **row.to_dict(),
                                'target_clipped_count': int(trades.target_clipped.sum()),
                                'rebalance_count': len(trades),
                                'gross_traded_total': float(trades.gross_traded_fraction.sum())})
            if k == 10:
                result.forecasts.to_csv(args.output / model / 'price_forecasts.csv', index=False)
                forecasts = result.forecasts[result.forecasts.full_forecast_horizon]
                errors = forecasts.predicted_price - forecasts.realized_price
                naive_errors = forecasts.anchor_price - forecasts.realized_price
                price_metrics.append({'model': model, 'asset_predictions': len(forecasts),
                    'price_rmse_adjusted_thb': float(np.sqrt(np.mean(errors ** 2))),
                    'price_mae_adjusted_thb': float(np.mean(np.abs(errors))),
                    'price_mape': float(np.mean(np.abs(errors) / forecasts.realized_price)),
                    'no_change_price_mape': float(np.mean(np.abs(naive_errors) / forecasts.realized_price)),
                    'direction_accuracy': float(np.mean(np.sign(forecasts.predicted_price - forecasts.anchor_price)
                                                       == np.sign(forecasts.realized_price - forecasts.anchor_price)))})
                import torch
                last = list(cache.values())[-1]
                checkpoint = last.checkpoint
                checkpoint['asset_names'] = list(prices.columns)
                checkpoint['decision_date'] = str(result.forecasts.decision_date.max().date())
                checkpoint['input_sha256'] = hashlib.sha256(args.prices.read_bytes()).hexdigest()
                torch.save(checkpoint, args.output / model / 'latest_model.pt')
            print(f'  {model}, top {k}: complete', flush=True)
        pd.DataFrame(metrics).to_csv(args.output / 'portfolio_metrics.csv', index=False)
        pd.DataFrame(price_metrics).to_csv(args.output / 'price_metrics.csv', index=False)
    metadata = {'input_file': args.prices.name, 'sha256': hashlib.sha256(args.prices.read_bytes()).hexdigest(),
                'observations': len(prices), 'assets': list(prices.columns),
                'start': str(prices.index.min().date()), 'end': str(prices.index.max().date()),
                'models': ['lstm', 'bilstm', 'rcnn_bilstm'], 'portfolio_sizes': list(range(5, 11)),
                'epochs': args.epochs, 'target_return': args.target_return,
                'forecast_horizon': 21, 'execution_delay': 1, 'train_window': 252,
                'annualization': 252, 'risk_free_assumption': 0.02, 'cost_bps': 10,
                'max_weight': 0.35, 'seed': 42,
                'versions': {lib: importlib.metadata.version(lib) for lib in ['numpy', 'pandas', 'scipy', 'torch']}}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('All 18 model/portfolio-size experiments completed.')


if __name__ == '__main__':
    main()
