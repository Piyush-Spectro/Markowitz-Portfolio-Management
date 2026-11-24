"""Predefined cost and universe-size checks; no test-set parameter selection."""
from dataclasses import replace
from pathlib import Path
import argparse
import pandas as pd
from markowitz.data import load_prices
from markowitz.backtest import Config, run_backtest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prices', type=Path, default=Path('data/sample_prices.csv'))
parser.add_argument('--output', type=Path, default=Path('outputs/sensitivity'))
args = parser.parse_args()
prices = load_prices(args.prices)
base = Config()
experiments = [('cost_bps', cost, replace(base, cost_bps=cost)) for cost in [0, 10, 25, 50]]
experiments += [('top_k', k, replace(base, top_k=k)) for k in range(5, min(10, prices.shape[1]) + 1)]
rows = []
for parameter, value, config in experiments:
    result = run_backtest(prices, config)
    for strategy, metrics in result.metrics.iterrows():
        rows.append({'parameter': parameter, 'value': value, 'strategy': strategy, **metrics.to_dict()})
args.output.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(args.output / 'sensitivity.csv', index=False)
print('Sensitivity checks complete.')
