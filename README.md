# Enhancing Portfolio Optimization through Machine Learning Stock Prediction

**Real SET50-cohort prices · R-CNN–BiLSTM, LSTM and BiLSTM · Markowitz optimization**

This project forecasts future adjusted stock prices, ranks stocks by predicted return, and constructs long-only portfolios that minimize variance for a specified expected return. It evaluates the complete process on real historical prices rather than treating forecast accuracy as proof of investment performance.

[Measured results](docs/results.md) · [Executed research notebook](notebooks/portfolio_research.ipynb) · [Methodology](docs/methodology.md) · [Data provenance](data/set50/README.md)

![SET50 research results](reports/set50/overview.png)

## Study at a glance

| Component | Implementation |
| --- | --- |
| Historical universe | January 2015 SET50 constituents, from SET's December 2014 announcement |
| Usable data | 41 of 50 requested stocks; 1,462 trading dates, January 2015–December 2020 |
| Price models | LSTM, BiLSTM, and recurrent CNN followed by BiLSTM |
| Forecast target | Adjusted terminal stock price, 21 holding days plus one execution-delay day ahead |
| Stock selection | Rank predicted returns; test top 5, 6, 7, 8, 9 and 10 stocks |
| Primary allocation | Minimum variance for an 8% annual expected-return target; 35% position cap |
| Comparators | Full-cohort equal weight, selected-universe minimum variance, and Black–Litterman |
| Evaluation | Walk-forward refitting, purged validation, price forecast errors, Sharpe, arithmetic mean return and volatility |
| Trading assumptions | One-bar execution delay, drifting holdings and 10 bps per unit bought or sold |

The published study contains **18 model/portfolio-size experiments**, their forecasts, allocations, trade logs, summary metrics and the latest trained checkpoint for each neural model.

## Important scope decisions

**Data coverage:** eight histories could not be retrieved and BANPU lacked sufficient early observations. All exclusions are documented. The 41-stock fixed cohort is not a survivorship-free reconstruction of the changing SET50 index.

**Dates:** “2015–2020” inclusive spans six calendar years. The first year is initial training history; the walk-forward evaluation covers approximately the following five years.

**Architecture:** the hybrid contains an actual recurrent convolution block with shared weights, followed by a bidirectional LSTM. Its equations and layer sizes are documented. The proposal supplies no exact paper architecture to reproduce.

**Return targets:** 8% is an expected-return constraint, not a promised realized return. When the target is infeasible, it is lowered to the feasible ceiling and recorded in the trade log.

## Reproduce the real-data study

Python 3.11+ is required. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -e '.[dev,neural]'

python scripts/download_set50.py
python scripts/run_set50_study.py
python scripts/report_set50.py
python scripts/execute_notebook.py
pytest -q
```

Open `reports/set50/report.html` locally for the visual report. The notebook can also be inspected directly on GitHub with its saved outputs.

Market downloads require internet access and depend on provider availability. Raw responses and cleaned price panels remain local and git-ignored. The repository includes the downloader, historical symbol list, source URLs, data hashes, coverage audits and derived results. Neither reference PDF is included.

### Run one configuration

```bash
markowitz --prices data/set50/adjusted_prices.csv \
  --forecast-target price --model rcnn_bilstm --epochs 20 \
  --top-k 10 --allocation target_return --target-return 0.08 \
  --max-weight 0.35 --cost-bps 10 --output outputs/custom
```

The complete study uses `run_set50_study.py` to reuse each model's forecasts across all six portfolio sizes. This avoids training a different random model for each allocation comparison.

### Replay a trained model

Each model folder under `reports/set50` includes `latest_model.pt`. Load it with `torch.load(path, weights_only=True)` and call `markowitz.price_forecast.predict_from_checkpoint(checkpoint, observed_prices)`. Supply positive adjusted prices in the saved `asset_names` order, ending on the checkpoint's `decision_date`, to reproduce its historical forecast. The notebook demonstrates the replay.

## Validation

Tests cover analytic portfolio solutions, feasible and infeasible return targets, recurrent-convolution gradients, chronological target alignment, purged validation, checkpoint replay, future-data perturbation, missing-price handling, explicit mean returns, and trading-cost accounting. GitHub Actions runs the core suite and a separate neural-model suite.

The real-data analysis reports both price errors and realized portfolio outcomes. It includes a no-change-price baseline and does not assume that the hybrid must outperform simpler models. Read the [results note](docs/results.md) before interpreting any individual Sharpe ratio.

## Repository layout

```text
src/markowitz/       Data acquisition, cleaning, models, optimization and backtesting
scripts/            Reproducible download, experiment, report and notebook commands
data/set50/         Historical universe, provenance and download/cleaning audits
reports/set50/      Real-data results, charts, logs and trained checkpoints
notebooks/          Executed walkthrough and checkpoint replay
tests/              Numerical, timing, data-quality and integration checks
docs/               Methods, proposal coverage and results
```

<details>
<summary>Offline synthetic smoke test and earlier experiments</summary>

The artificial sample is retained for quick offline checks. It is separate from the primary SET50 study.

```bash
pip install -e '.[dev]'
markowitz --output outputs/synthetic
```

`reports/demo` and `reports/comparison` contain earlier synthetic demonstrations. They are not real-market results. Historical-mean, ridge and feed-forward CNN–BiLSTM return forecasts remain available as optional baseline implementations.

</details>

## Author

[Piyush Kumar](https://github.com/Piyush-Spectro)

Educational research. One historical sample, fixed hyperparameters and a coverage-filtered cohort do not establish a deployable investment strategy.
