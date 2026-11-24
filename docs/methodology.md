# Methodology

## Research design

The experiment asks whether forecast-driven allocations add value over simple portfolio rules. Every strategy uses the same out-of-sample dates, risk-free assumption, and transaction-cost convention. All model fitting and stock ranking occur within each rolling training window.

The implementation is an independent research exercise. It is not a reproduction of a specific paper: the original historical universe, corporate-action series, and experimental configuration are not supplied. The reference learning material informed the topics; its files are not distributed here.

## Returns and estimation

For adjusted price P, daily simple return is `r[t] = P[t] / P[t-1] - 1`. Arithmetic mean returns and covariance are annualized by 252. CAGR is calculated separately from compounded realized returns; it is not interchangeable with an annualized arithmetic mean.

The covariance estimator blends the sample covariance with its diagonal:

`Sigma = 0.85 × sample_cov + 0.15 × diagonal(sample_cov)`

A diagonal floor of `1e-10` ensures positive definiteness for constant or duplicated price series. The shrinkage coefficient is fixed in advance, not selected on the test set. This simple estimator is intentionally transparent; it is not Ledoit–Wolf estimated shrinkage.

## Forecasts

Each forecast estimates the mean daily return over the next holding-period horizon. A 20-observation lookback forms the input window; targets average the following 21 observations by default. Only targets entirely contained in the training window are available for fitting.

| Model | Implementation | Main tradeoff |
| --- | --- | --- |
| Historical mean | Mean of the rolling training returns | Stable baseline, slow to adapt |
| Ridge | Flattened multivariate return window; L2 penalty 100 | Cheap and inspectable, limited nonlinear structure |
| LSTM | One 16-unit recurrent layer | Learns temporal features with relatively few parameters |
| BiLSTM | One bidirectional 16-unit recurrent layer | Reads both directions inside an already observed window |
| CNN–BiLSTM | 16-channel 1D convolution, ReLU, bidirectional LSTM, linear output | Adds local temporal feature extraction at greater complexity |

The neural network is a compact CNN–BiLSTM implementation, not a claim to reproduce the ambiguously named R-CNN architecture in the proposal. Ridge is the default because it is a useful low-cost baseline.

For ridge, feature scaling and the intercept are learned within the training sample. Neural validation uses the final 20% of available windows. Training windows whose target dates overlap the validation targets are purged. Normalization uses only the neural training partition. Adam trains for a fixed maximum epoch count, and the lowest validation-loss weights are restored. There is no test-set early stopping.

Bidirectionality does not grant access to future market observations: both directions operate within the historical input window. Models retrain from scratch at each rebalance. Fixed seeds support reproducibility; they do not establish robustness across seeds.

## Allocation

Portfolios satisfy `sum(w) = 1` and `0 <= w[i] <= cap`. Optional top-k selection ranks forecasts within the training window. Equal weight retains the full input universe as an independent benchmark; minimum variance and both forecast-driven strategies use the selected universe. This means a top-k experiment changes both selection and allocation relative to equal weight.

Minimum variance minimizes `w.T @ Sigma @ w`. The efficient frontier solves that problem for a sequence of minimum expected-return constraints, starting at the global minimum-variance portfolio. The upper endpoint is the maximum feasible return under the position cap, not simply the highest asset mean.

Maximum Sharpe maximizes `(mu.T @ w - rf) / sqrt(w.T @ Sigma @ w)`. For positive excess returns, the solver uses the equivalent convex formulation: minimize `y.T @ Sigma @ y` subject to `(mu - rf).T @ y = 1`, nonnegative `y`, and linear transformed caps `y[i] <= cap × sum(y)`. It then recovers `w = y / sum(y)`. Scaling the objective and excess-return constraint improves numerical conditioning. If no feasible portfolio has a positive expected excess return, the backtest uses minimum variance and logs the fallback. Other solver failures stop the run. Solver success, budget, bounds, and target feasibility are checked; failed solutions are never silently normalized and accepted.

## Bayesian views

Black–Litterman combines an explicit prior with uncertain views. All means, views, and covariances use annual units. With equal-weight reference allocation `w_ref`, risk aversion `delta = 2.5`, and `tau = 0.05`:

```text
pi = delta × Sigma × w_ref                    # prior excess mean
C = tau × Sigma                              # prior mean uncertainty
q = annualized forecast - risk_free          # absolute excess-return views
P = identity                                 # one view for each selected asset
Omega = diagonal(P × C × P.T)                 # fixed view-uncertainty heuristic
K = C × P.T × inverse(P × C × P.T + Omega)
posterior_mean = pi + K × (q - P × pi)
posterior_mean_cov = C - K × P × C
predictive_cov = Sigma + posterior_mean_cov
```

The implementation uses linear solves instead of explicitly inverting matrices. The risk-free rate is added back before maximum-Sharpe optimization. The API also accepts relative views and a user-specified positive-definite Omega.

Equal-weight reference returns are a modeling assumption, not market-implied equilibrium evidence. The default Omega is a heuristic rather than calibrated forecast uncertainty. When Omega scales with tau this way, changing tau does not change the posterior mean; it still changes the predictive covariance. The posterior uncertainty should not be interpreted as a calibrated confidence interval for realized investment returns.

## Walk-forward timing

For a test return at index `t`:

1. Fit and select assets using returns ending at `t-2`.
2. Execute target weights at the close of `t-1`.
3. Earn the close-to-close return indexed `t`.
4. Keep positions until the next rebalance, updating weights as prices move.

That extra bar is deliberate: a strategy should not trade at the very close needed to compute its signal. Training targets refer to the next holding period after the observed window, while execution shifts the realized holding period one bar later. Forecast errors therefore measure the actionable, delayed forecast rather than a frictionless next-bar estimate.

Costs equal `cost_bps / 10000 × sum(abs(target - pretrade_weights))`. This is gross traded notional, not the half-L1 convention sometimes called one-way turnover. Initial investment has gross notional 1; a complete rotation can approach 2. Fees reduce wealth before the next return: `(1 - fee) × (1 + gross_return) - 1`. Weight drift uses asset returns, while proportional fees reduce the portfolio value uniformly. There is no terminal liquidation charge.

## Metrics and reproducibility

The report includes total return, CAGR, annualized sample volatility, daily-excess-return Sharpe, and maximum drawdown measured from initial capital. A zero-volatility Sharpe is undefined and reported as missing. Forecast RMSE averages errors across assets and rebalance blocks. The final short block is retained and identified by its `holding_days`; it has equal weight in the headline forecast error despite its shorter horizon.

Each run exports input SHA-256, asset names, library versions, configuration, daily net returns, target weights, trades, forecasts, and summary metrics. The initial frontier uses only the first training window and is shown for the full universe. It does not represent later selected subsets or realized test returns.

## Limits and next experiments

No model is assumed to outperform. Evaluate multiple market regimes and seeds, tune hyperparameters only in nested chronological validation, and compare forecasts with zero-return and historical-mean baselines. Use a point-in-time universe to avoid survivorship bias. Daily adjusted prices should be consistently defined, with splits and dividends handled by the data provider. The loader rejects missing entries; selecting only complete series can itself bias the universe and must be documented.

The execution model omits bid–ask spreads beyond the proportional fee, nonlinear market impact, taxes, borrow costs, and capacity constraints. Gaussian mean–variance assumptions do not adequately describe tail losses. This project does not include live trading, brokerage credentials, or automated recommendations.

## References

- Markowitz, H. (1952), *Portfolio Selection*, Journal of Finance, 7(1), 77–91. [DOI](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x).
- [MathWorks: Black–Litterman portfolio optimization](https://www.mathworks.com/help/finance/black-litterman-portfolio-optimization.html), for the linear-Gaussian update and the distinction between views and prior information.
- [SciPy: SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html), for the constrained numerical solver.
- [PyTorch: LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html), for recurrent-layer conventions and bidirectional hidden states.
