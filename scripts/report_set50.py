"""Build the checked-in research note and charts from measured study outputs."""
from pathlib import Path
import json
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/set50'
metrics = pd.read_csv(OUT / 'portfolio_metrics.csv')
price_metrics = pd.read_csv(OUT / 'price_metrics.csv')
meta = json.loads((OUT / 'metadata.json').read_text())
manifest = json.loads((ROOT / 'data/set50/manifest.json').read_text())
models = ['lstm', 'bilstm', 'rcnn_bilstm']
names = {'lstm': 'LSTM', 'bilstm': 'BiLSTM', 'rcnn_bilstm': 'R-CNN–BiLSTM'}
colors = ['#267d91', '#ce8539', '#6055a5']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10, 'axes.titleweight': 'bold'})
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
for model, color in zip(models, colors):
    ret = pd.read_csv(OUT / model / 'k10/returns.csv', index_col='date', parse_dates=True)
    axes[0, 0].plot(ret.index, (1 + ret['Forecast Markowitz']).cumprod(), label=names[model], color=color)
    sub = metrics[(metrics.model == model) & (metrics.strategy == 'Forecast Markowitz')]
    axes[0, 1].plot(sub.top_k, sub.sharpe, marker='o', color=color, label=names[model])
    predictions = pd.read_csv(OUT / model / 'price_forecasts.csv', parse_dates=['decision_date'])
    asset = predictions[(predictions.asset == 'ADVANC') & predictions.full_forecast_horizon]
    axes[1, 1].plot(asset.decision_date, asset.predicted_price, color=color, alpha=0.8, label=names[model])
axes[0, 0].plot(ret.index, (1 + ret['Equal weight']).cumprod(), label='Equal-weight cohort', color='#778899', ls='--')
axes[0, 0].set(title='Top 10 / net growth of 1 unit', ylabel='Wealth after trading costs')
axes[0, 0].legend(frameon=False, fontsize=8)
axes[0, 1].set(title='Target-return portfolio / size sensitivity', xlabel='Selected stock count', ylabel='Annualized Sharpe')
axes[0, 1].set_xticks(range(5, 11))
axes[0, 1].axhline(0, color='#94a3b8', lw=0.8)
p = price_metrics.set_index('model').loc[models]
axes[1, 0].bar([names[m] for m in models], p.price_mape, color=colors, width=0.6)
axes[1, 0].axhline(p.no_change_price_mape.iloc[0], ls='--', color='#778899', label='No-change price baseline')
axes[1, 0].set(title='Terminal-price forecast error', ylabel='Mean absolute percentage error')
axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
axes[1, 0].set_ylim(0, p.price_mape.max() * 1.25)
axes[1, 0].legend(frameon=False, fontsize=8)
axes[1, 1].plot(asset.decision_date, asset.realized_price, color='#172b3a', lw=1.4, ls='--', label='Realized terminal price')
axes[1, 1].set(title='ADVANC / terminal-price forecasts', xlabel='Decision date (22 trading days before target)', ylabel='Provider-adjusted THB')
axes[1, 1].legend(frameon=False, fontsize=8)
for ax in axes.flat:
    ax.grid(alpha=0.15)
fig.suptitle('SET50 COHORT / Real-price research study\n2015–2020 data · 41 of 50 initial constituents · 18 model/size experiments',
             x=0.02, ha='left', fontsize=16)
fig.savefig(OUT / 'overview.png', dpi=160)
plt.close(fig)

text = f'''# SET50 study: measured results

## Scope

The primary study uses real Yahoo Finance adjusted prices for **{len(meta['assets'])} stocks** from the January 2015 SET50 cohort. The panel contains **{meta['observations']:,} observations**, from **{manifest['first_date']} to {manifest['last_date']}**. The first 252 returns are training history; evaluation begins {ret.index[0].date()} and ends {ret.index[-1].date()}.

All **18 combinations** of LSTM, BiLSTM and R-CNN–BiLSTM with top 5, 6, 7, 8, 9 and 10 stocks were run. The primary strategy minimizes variance subject to an 8% annual expected-return target, a 35% position cap and full investment. Infeasible targets are lowered and logged; realized returns are not guaranteed to reach the requested target.

The historical list contained 50 stocks. Eight histories were unavailable and BANPU had insufficient early coverage. The resulting fixed coverage subset is not a survivorship-free reconstruction of the changing SET50 index. [Coverage and source audit](../data/set50/README.md).

![Real-data experiment overview](../reports/set50/overview.png)

## Top-10 target-return portfolios

These figures are net of 10 bps per unit bought or sold, including initial investment. The risk-free assumption is 2% annually. Arithmetic annual mean and CAGR are intentionally reported separately.

| Model | Total return | Annual mean | CAGR | Volatility | Sharpe | Max drawdown | Lowered targets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
'''
for model in models:
    row = metrics[(metrics.model == model) & (metrics.top_k == 10) & (metrics.strategy == 'Forecast Markowitz')].iloc[0]
    text += f'| {names[model]} | {row.total_return:.2%} | {row.mean_annual_return:.2%} | {row.cagr:.2%} | {row.volatility:.2%} | {row.sharpe:.3f} | {row.max_drawdown:.2%} | {int(row.target_clipped_count)}/{int(row.rebalance_count)} |\n'
benchmark = metrics[metrics.strategy == 'Equal weight'].iloc[0]
text += f'\nThe full-cohort equal-weight benchmark returned **{benchmark.total_return:.2%}**, with Sharpe **{benchmark.sharpe:.3f}** and maximum drawdown **{benchmark.max_drawdown:.2%}**. This is a portfolio of the included stocks, not the official SET50 total-return index.\n'
text += '''
## All portfolio sizes

Annualized Sharpe ratios for the forecast-based target-return portfolios:

| Selected stocks | LSTM | BiLSTM | R-CNN–BiLSTM |
| --- | ---: | ---: | ---: |
'''
for k in range(5, 11):
    values = [metrics[(metrics.model == m) & (metrics.top_k == k) & (metrics.strategy == 'Forecast Markowitz')].iloc[0].sharpe for m in models]
    text += f'| {k} | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |\n'
text += '''
## Price prediction

Errors use complete forecast horizons only. Price MAE/RMSE use provider-adjusted THB levels; MAPE compares relative accuracy across stocks. The no-change baseline predicts the last observed adjusted price. The ADVANC chart is a fixed alphabetical example, not the stock chosen for the best forecast fit.

| Model | Price MAE | Price RMSE | MAPE | Direction accuracy |
| --- | ---: | ---: | ---: | ---: |
'''
for row in price_metrics.itertuples():
    text += f'| {names[row.model]} | {row.price_mae_adjusted_thb:.3f} | {row.price_rmse_adjusted_thb:.3f} | {row.price_mape:.2%} | {row.direction_accuracy:.2%} |\n'
text += f'\nNo-change-price baseline MAPE: **{price_metrics.no_change_price_mape.iloc[0]:.2%}**. Each model was scored on **{int(price_metrics.asset_predictions.iloc[0]):,}** asset/horizon predictions.\n'
text += '''
## Interpretation

All three neural models had higher MAPE than the no-change-price baseline in this run. The top-10 hybrid portfolio also had a lower Sharpe ratio than either recurrent baseline. These results do not support a claim that the hybrid is superior.

A low price error does not by itself establish profitable stock selection. The ranking step, covariance estimate, target feasibility, weight cap and trading costs all affect realized portfolio outcomes. Compare model results with the simple benchmark and the no-change forecast rather than assuming that the hybrid must perform best.

These are descriptive results from one historical interval and one seed, not significance tests. No model or portfolio size is designated a winner for live investment. Future work should use licensed point-in-time constituent and delisting records, multiple seeds, transaction-cost stress tests, and nested chronological tuning.

## Reproduce

```bash
pip install -e '.[dev,neural]'
python scripts/download_set50.py
python scripts/run_set50_study.py
python scripts/report_set50.py
python scripts/execute_notebook.py
pytest -q
```

Raw provider responses and cleaned price panels stay local. Source hashes, all configurations, derived forecasts, trade logs, allocations, metrics and latest model checkpoints are included. Neither supplied PDF is included.

- [Portfolio metrics for every combination](../reports/set50/portfolio_metrics.csv)
- [Price-prediction metrics](../reports/set50/price_metrics.csv)
- [Run metadata](../reports/set50/metadata.json)
- [R-CNN–BiLSTM price forecasts](../reports/set50/rcnn_bilstm/price_forecasts.csv)
- [Executed notebook](../notebooks/portfolio_research.ipynb)
- [Methodology and assumptions](methodology.md)
'''
(ROOT / 'docs/results.md').write_text(text)
top = metrics[(metrics.top_k == 10) & (metrics.strategy == 'Forecast Markowitz')]
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SET50 | Portfolio research</title><style>
body{margin:0;background:#f4f6f8;color:#172b3a;font:16px/1.6 system-ui}main{max-width:1120px;margin:auto;padding:48px 24px}h1{font-size:40px;line-height:1.15}h2{font-size:24px}img{width:100%}.card{background:white;border:1px solid #dce3e9;border-radius:12px;padding:22px;margin:24px 0}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:10px;border-bottom:1px solid #e2e8f0;text-align:right}.muted{color:#576777}a{color:#267d91}</style><main>
<p class="muted">REAL-PRICE RESEARCH / SET50 HISTORICAL COHORT</p><h1>Price forecasts.<br>Constrained portfolios.<br>Measured outcomes.</h1>
<p>41 stocks from the January 2015 SET50 cohort, 2015–2020 prices, three neural architectures and six portfolio sizes. Nine names were excluded for unavailable or incomplete history. This is a fixed coverage subset, not an index replication.</p>
<div class="card"><img src="overview.png" alt="Real-data wealth, Sharpe sensitivity, price prediction errors and ADVANC forecasts"></div><div class="card"><h2>Top-10 target-return portfolios</h2><div class="scroll">'''
show = top[['model', 'total_return', 'mean_annual_return', 'volatility', 'sharpe', 'max_drawdown', 'target_clipped_count']].copy()
for col in ['total_return', 'mean_annual_return', 'volatility', 'max_drawdown']:
    show[col] = show[col].map(lambda v: f'{v:.2%}')
show.sharpe = show.sharpe.map(lambda v: f'{v:.3f}')
page += show.to_html(index=False, border=0, escape=True)
page += '</div></div><p>Requested expected-return target: 8% annually; weight cap: 35%; trading costs: 10 bps per unit traded. Unattainable targets are lowered and recorded, not claimed as achieved.</p><p>The study has availability bias, a fixed initial cohort, one seed and current-vintage price adjustments. Its results do not establish a live investment edge.</p><p><a href="../../docs/results.md">Full results note</a> · <a href="portfolio_metrics.csv">All portfolio metrics</a> · <a href="price_metrics.csv">Forecast metrics</a> · <a href="metadata.json">Run metadata</a></p></main></html>'
(OUT / 'report.html').write_text(page)
print('Real-data charts, HTML report and results note written.')
