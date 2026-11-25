# Methodology: price forecasts to constrained portfolios

## Scope and dates

The primary experiment uses real prices for 41 downloadable members of the January 2015 SET50 cohort. It covers January 2015–December 2020 inclusive. Those endpoints span six calendar years; the proposal's wording “five years (2015–2020)” is internally ambiguous. The first 252 observations are used for initial model training, leaving roughly five years for walk-forward evaluation.

[Data provenance](../data/set50/README.md) documents all 50 requested members, symbol mappings, nine exclusions, the cleaning policy and source hashes. This is a fixed-cohort study, not an index replication. Availability bias and revised corporate-action adjustments remain limitations; the report does not present the universe as survivorship-free.

## Price target and normalization

For anchor date `a`, lookback `L = 20`, holding horizon `H = 21`, and execution delay `D = 1`:

```text
X[a] = log(P[a-L+1:a+1] / P[a])
y[a] = log(P[a+H+D] / P[a])
predicted_price[a+H+D] = P[a] * exp(predicted_y[a])
predicted_horizon_return = predicted_price / P[a] - 1
```

`P` denotes provider-adjusted closing prices. Predictions are terminal adjusted price levels, not a claim to forecast contemporaneous raw transaction prices. Relative log-price inputs avoid letting the absolute price scale of one stock dominate the network. Feature and target standardization are fitted only on the neural training partition; predicted targets are transformed back into adjusted prices.

Targets include the one-bar execution delay. A signal observed at close `a` is executed at close `a+1` and held through close `a+22`. Asset rankings use predicted returns relative to the known anchor price, never the future execution price. Expected returns are converted to annual arithmetic approximations using `predicted_horizon_return * 252 / 22` for mean-variance optimization. This approximation is an assumption, not a calibrated expected-return model.

## Models

All three proposal models are trained and evaluated on identical rolling windows:

- **LSTM:** one 16-unit recurrent layer and a linear multi-asset output head.
- **BiLSTM:** one bidirectional recurrent layer with 16 units per direction and a linear head.
- **R-CNN–BiLSTM:** a recurrent convolution block followed by the same BiLSTM and linear head.

The recurrent convolution is explicitly defined by:

```text
drive = Conv1D_input(X)
h_0 = ReLU(drive)
h_j = ReLU(drive + Conv1D_recurrent(h_(j-1))), j = 1, 2, 3
features = transpose(h_3)
output = Linear(concat(final_forward_state, final_backward_state))
```

Both convolutions use kernel width 3 and 16 output channels. The recurrent convolution **reuses its weights** over the three refinements. It is not merely the earlier feed-forward CNN renamed. The convolution operates over temporal patterns in multivariate stock sequences; no arbitrary stock-order spatial neighborhood is assumed. Bidirectionality is confined to an already observed historical input window.

The proposal does not provide a paper, kernel specification or exact layer dimensions. This is a documented implementation of its recurrent-convolution/BiLSTM design, not a claim of exact reproduction of an unidentified architecture.

Each model uses Adam (learning rate 0.003, weight decay 0.001), gradient norm clipping at 1, a maximum of 20 epochs and seed 42 on CPU. The final 20% of available training windows form chronological validation. A 22-window purge separates training targets from validation anchors. Normalization uses the retained training partition only, and the best validation-loss checkpoint is restored. Test returns do not affect normalization, early stopping, ranking or optimization.

The latest checkpoint for each model is saved with architecture, asset order, normalization parameters, decision date and input hash. These checkpoints can be reloaded with `torch.load(..., weights_only=True)` and `predict_from_checkpoint`.

## Stock selection and allocation

At each rebalance, rank all 41 stocks by predicted horizon return and select the top **5, 6, 7, 8, 9 or 10**. All six sizes are evaluated separately for all three neural models: **18 model/size experiments**. The exact same forecasts are cached and reused across sizes. Selected-universe size need not equal the number of nonzero optimized holdings.

Portfolios are long-only and fully invested, with a 35% maximum weight per asset. The primary proposal strategy solves:

```text
minimize       w.T @ Sigma @ w
subject to     sum(w) = 1
               0 <= w[i] <= 0.35
               mu.T @ w >= target_return
```

The requested annual expected-return target is **8%**, fixed before evaluation. If it exceeds the maximum feasible expected return under the cap, the effective target is lowered to that ceiling (less a numerical tolerance). Requested target, effective target, clipping flag and achieved expected return are written to every trade log. An infeasible 8% target is never reported as achieved.

Comparators are the equal-weight full-cohort portfolio, selected-universe global minimum variance, and a Black–Litterman target-return portfolio. The risk-free assumption is a constant 2% annually; it is not represented as a measured Thai risk-free-rate series.

Daily simple returns determine covariance. The estimate blends 85% sample covariance and 15% its diagonal, annualized by 252, plus a `1e-10` diagonal floor. Optimization uses SLSQP with budget/bound/return-feasibility checks. The optional maximum-Sharpe mode remains available for exploratory runs; it is not the primary proposal experiment.

## Bayesian extension

Black–Litterman remains an additional analysis beyond the proposal's minimum requirements. With equal-weight reference allocation, `delta = 2.5` and `tau = 0.05`, the prior excess mean is `pi = delta * Sigma @ w_ref`. Price-derived annual excess returns form absolute views. `Omega` is the diagonal of `tau * Sigma`; the predictive covariance includes posterior mean uncertainty. Linear solves compute the update.

This is a reference-allocation prior, not a market-cap-weighted prior. View uncertainty is a fixed heuristic, not an estimated prediction interval. The identical target-return policy is applied to the resulting posterior means and covariance.

## Execution and performance

Holdings drift with asset returns between rebalances. Trading fees are 10 basis points times gross bought-plus-sold notional, including the initial investment. Fees reduce wealth before the first return after a rebalance. There is no terminal liquidation fee, market-impact model or tax calculation.

Portfolio outputs include:

- Arithmetic mean daily return and annualized arithmetic mean return.
- Annualized sample volatility and daily-excess-return Sharpe ratio.
- Total compounded return, CAGR and maximum drawdown from initial capital.
- Turnover and infeasible-target counts.

Price-forecast outputs include RMSE and MAE in adjusted THB units, MAPE, direction accuracy and a no-change-price baseline. Forecast evaluation excludes the final incomplete holding period; portfolio performance includes its realized returns. MAPE gives a scale-comparable view alongside price errors that can otherwise be dominated by high-priced stocks.

## Limitations

The goal of high return and low risk is a research hypothesis, not a guaranteed result. A single historical period, one seed, a coverage-filtered cohort and fixed hyperparameters cannot establish a deployable investment strategy. Missing delisted histories, current-vintage adjustments, stale quotes, estimated execution prices and omitted capacity costs limit external validity. A stronger follow-up would use licensed point-in-time constituent/security histories, multiple seeds and nested chronological model tuning.

The original synthetic example remains an offline software smoke test; its outputs are clearly separated under `reports/demo` and `reports/comparison`.

## References

- Markowitz, H. (1952), *Portfolio Selection*, [DOI](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x).
- [SET's December 2014 constituent announcement](https://ptg.listedcompany.com/misc/SET100-50.pdf).
- [Yahoo Finance historical-data fields](https://finance.yahoo.com/quote/ADVANC.BK/history/).
- [SciPy SLSQP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).
- [PyTorch LSTM documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).
- [Black–Litterman update derivation](https://www.mathworks.com/help/finance/black-litterman-portfolio-optimization.html).
