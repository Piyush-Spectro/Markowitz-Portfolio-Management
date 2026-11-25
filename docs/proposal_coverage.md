# Proposal coverage

| Proposal item | Delivered implementation | Evidence |
| --- | --- | --- |
| Historical SET50 prices, 2015–2020 | Real adjusted prices for 41 downloadable members of the initial-2015 cohort | `data/set50/manifest.json`, download/cleaning audits |
| Cleaning and missing-value handling | Invalid-price detection, coverage thresholds, limited internal forward-fill, no backward/terminal fill, large-return flags | `market_data.py`, cleaning tests |
| Normalize data for ML | Relative log-price windows plus training-only standardization | `price_forecast.py` |
| Predict future stock prices | Terminal adjusted-price predictions, exported with realized prices and anchor prices | Model `price_forecasts.csv` files |
| R-CNN–BiLSTM hybrid | Shared-weight recurrent convolution, then BiLSTM and linear output head | `build_price_network`, recurrent-gradient test |
| LSTM and BiLSTM comparators | Both trained under the same timing and validation rules | `reports/set50/price_metrics.csv` |
| Rank and select stocks | Rank predicted horizon returns before execution | `backtest.py`, selected flags in forecasts |
| Portfolio sizes 5, 6, 7, 8, 9, 10 | All six sizes for all three models | 18 run folders and `portfolio_metrics.csv` |
| Calculate mean and covariance | Price-derived expected returns and annualized shrunk covariance | `backtest.py`, `portfolio.py` |
| Minimize risk for a given return | Target-return constrained minimum variance; infeasible targets explicitly recorded | Each run's `trades.csv` |
| Sharpe, average return and risk | Sharpe, arithmetic daily/annual means, annualized volatility, plus CAGR/drawdown | Each run's `metrics.csv` |
| Trained models and reproducible code | Latest checkpoint per neural model, metadata, notebook and CLI | `latest_model.pt`, executed notebook |

## Boundaries that should not be hidden

- The data do not cover all 50 histories or every historical index rebalance. Nine exclusions create availability/survivorship bias. Full point-in-time coverage would require an appropriate historical security/constituent archive.
- The inclusive 2015–2020 range is six calendar years. Roughly the first year is training and the remaining five years are evaluation.
- The proposal does not specify an exact R-CNN paper or layer configuration. This implementation defines recurrent convolution explicitly and documents its choices; it does not claim an exact paper reproduction.
- High return and low risk are objectives to evaluate, not outcomes guaranteed by the code. Measured results and forecast baselines are reported without claiming statistical significance.
- Raw market-price inputs are downloaded locally. Both user-supplied PDFs are excluded from version control.
