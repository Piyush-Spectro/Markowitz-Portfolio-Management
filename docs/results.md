# Experiment results

## Summary

This is a completed software experiment on an artificial dataset, not evidence of a tradable strategy. The sample contains eight fictional assets generated from shared factors and independent noise. All results below are measured outputs from the repository code.

The useful finding in this demonstration is that greater model complexity did not automatically improve allocation. Minimum variance had the smallest loss and drawdown in the default comparison. Every strategy lost money over this particular sample path. The seed was not changed to produce a more attractive chart.

## Default allocation comparison

Test observations: 755; dates: 2020-12-22 to 2023-11-13. Training window: 252 observations; rebalance interval: 21; cost: 10 bps per unit traded; position cap: 35%; risk-free assumption: 2% annually.

| Strategy | Total return | CAGR | Volatility | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal weight | -25.93% | -9.53% | 10.25% | -1.120 | -31.27% |
| Minimum variance | -11.71% | -4.07% | 8.44% | -0.685 | -21.07% |
| Forecast Markowitz | -22.23% | -8.05% | 9.75% | -1.015 | -29.53% |
| Black-Litterman | -21.52% | -7.77% | 9.50% | -1.012 | -28.50% |

The expected-return estimates matter, but so do the constraints. When no feasible allocation has a positive estimated excess return, the optimizer falls back to minimum variance and records that event. The headline returns include those decisions, rather than excluding difficult periods.

## Forecasting comparison

Each model used the same dataset, training window, execution delay and fee schedule. Neural models trained for at most 12 epochs per rebalance, with purged chronological validation. These settings were not tuned against the reported test period.

| Forecast model | Daily-return RMSE | Forecast Markowitz total return | Black–Litterman total return |
| --- | ---: | ---: | ---: |
| historical | 0.001943 | -22.11% | -19.68% |
| ridge | 0.002019 | -22.23% | -21.52% |
| lstm | 0.002004 | -13.52% | -15.82% |
| bilstm | 0.002016 | -20.16% | -16.34% |
| cnn_bilstm | 0.002006 | -16.55% | -15.42% |

A zero-return forecast has RMSE 0.001959 on the same realized blocks. All neural models and ridge had higher forecast error than this baseline in the sample. LSTM produced the smallest loss among the forecast-driven Markowitz variants, but that does not establish statistical significance or outperformance in another market.

## Cost sensitivity

These runs change only transaction costs, holding the ridge forecast, position cap and full universe fixed. Costs apply to buys and sells separately through gross traded notional.

| Cost per unit traded | Forecast Markowitz total return | Black–Litterman total return |
| --- | ---: | ---: |
| 0 bps | -19.32% | -18.82% |
| 10 bps | -22.23% | -21.52% |
| 25 bps | -26.40% | -25.42% |
| 50 bps | -32.87% | -31.49% |

The same sensitivity file includes top-k selections of five through eight assets. These are diagnostic comparisons, not a recommendation to select whichever setting has the best test return. Position caps, selected-universe risk and turnover should be considered together.

## Reproduce and inspect

```bash
markowitz --output reports/demo
python scripts/compare_models.py --neural --output reports/comparison
python scripts/sensitivity.py --output reports/sensitivity
python scripts/execute_notebook.py
```

Install the `dev` and `neural` extras first. The reports record library versions and the input fingerprint. Each model comparison also includes its own daily returns, target weights, forecast records and trade log. The tested direct dependency versions are in `requirements-tested.txt`; neural runs used PyTorch 2.9.1 on CPU.

- [Default metrics](../reports/demo/metrics.csv)
- [Model comparison](../reports/comparison/model_comparison.csv)
- [Cost and top-k sensitivity](../reports/sensitivity/sensitivity.csv)
- [Run metadata](../reports/demo/metadata.json)
- [Notebook](../notebooks/portfolio_research.ipynb)

## What this does not establish

The experiment does not validate performance on SET50 or another real-market universe. The artificial returns do not reproduce market microstructure, sector changes, corporate actions or delistings. One synthetic path and one seed cannot support an investment claim.

A next empirical study should use documented point-in-time data, compare multiple non-overlapping periods, run multiple training seeds, tune only inside nested chronological validation, and report uncertainty around performance differences. Training on prices or adding more layers would not by itself resolve those limitations.
