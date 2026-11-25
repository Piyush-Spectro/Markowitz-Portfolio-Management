"""Rolling evaluation with a one-bar execution delay and drifting holdings."""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from .data import returns_from_prices
from .forecast import forecast
from .portfolio import covariance, black_litterman, optimize


@dataclass(frozen=True)
class Config:
    train_window: int = 252
    rebalance: int = 21
    annualization: int = 252
    risk_free: float = 0.02
    cost_bps: float = 10.0
    max_weight: float = 0.35
    top_k: int | None = None
    forecast_method: str = "ridge"
    neural_epochs: int = 12
    seed: int = 42
    forecast_target: str = "return"
    allocation: str = "max_sharpe"
    target_return: float = 0.08

    def validate(self, assets, observations):
        k = self.top_k if self.top_k is not None else assets
        if not isinstance(k, int) or not 2 <= k <= assets:
            raise ValueError("top_k must be between two and the asset count.")
        if (self.train_window < 100 or self.rebalance < 1 or self.annualization < 1
                or self.neural_epochs < 1 or self.rebalance + 60 > self.train_window):
            raise ValueError("Invalid window, horizon, annualization, or epoch count.")
        if observations <= self.train_window + 1:
            raise ValueError("Need observations beyond the training window and execution delay.")
        if not np.isfinite([self.risk_free, self.cost_bps, self.max_weight]).all():
            raise ValueError("Configuration must be finite.")
        if self.risk_free <= -1 or not 0 <= self.cost_bps < 1000:
            raise ValueError("Invalid risk-free rate or transaction cost.")
        if not 1 / k <= self.max_weight <= 1:
            raise ValueError("Weight cap is infeasible for top_k.")
        if self.forecast_target not in {"return", "price"} or self.allocation not in {"max_sharpe", "target_return"}:
            raise ValueError("Unknown forecast target or allocation policy.")
        if not np.isfinite(self.target_return):
            raise ValueError("Target return must be finite.")
        allowed = {"ridge", "historical", "rcnn_bilstm", "lstm", "bilstm"} if self.forecast_target == "price" else {"ridge", "historical", "cnn_bilstm", "lstm", "bilstm"}
        if self.forecast_method not in allowed:
            raise ValueError("Unknown forecast method.")


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    config: dict


def performance(returns, annualization=252, risk_free=0.02):
    x = np.asarray(returns, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or (x <= -1).any():
        raise ValueError("Performance needs finite returns greater than -100%.")
    if annualization <= 0 or risk_free <= -1:
        raise ValueError("Invalid annualization or risk-free rate.")
    wealth = np.cumprod(1 + x)
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    volatility = x.std(ddof=1) * np.sqrt(annualization) if len(x) > 1 else 0.0
    daily_rf = (1 + risk_free) ** (1 / annualization) - 1
    return {"total_return": wealth[-1] - 1, "cagr": wealth[-1] ** (annualization / len(x)) - 1,
            "volatility": volatility,
            "sharpe": (x.mean() - daily_rf) * annualization / volatility if volatility > 1e-12 else np.nan,
            "max_drawdown": np.min(wealth / peaks - 1),
            "mean_daily_return": x.mean(), "mean_annual_return": x.mean() * annualization}


def run_backtest(prices, config=Config(), forecast_cache=None):
    returns = returns_from_prices(prices)
    config.validate(returns.shape[1], len(returns))
    a = returns.to_numpy()
    n = a.shape[1]
    names = list(returns.columns)
    strategies = ["Equal weight", "Minimum variance", "Forecast Markowitz", "Black-Litterman"]
    holdings = {name: np.zeros(n) for name in strategies}
    net, weight_rows, trade_rows, forecast_rows = [], [], [], []
    for start in range(config.train_window + 1, len(a), config.rebalance):
        # Forecast at close start-2; execute at close start-1; first P&L at start.
        train = a[start - 1 - config.train_window:start - 1]
        price_prediction = None
        if config.forecast_target == "price":
            import hashlib
            from .price_forecast import predict_prices
            train_prices = prices.iloc[start - 1 - config.train_window:start].to_numpy()
            key = (config.forecast_method, config.rebalance, config.neural_epochs, config.seed,
                   hashlib.sha256(train_prices.tobytes()).hexdigest())
            if forecast_cache is not None and key in forecast_cache:
                price_prediction = forecast_cache[key]
            else:
                price_prediction = predict_prices(train_prices, config.forecast_method, horizon=config.rebalance,
                                                  epochs=config.neural_epochs, seed=config.seed, keep_checkpoint=True)
                if forecast_cache is not None:
                    forecast_cache[key] = price_prediction
            mu_daily = (price_prediction.prices / train_prices[-1] - 1) / (config.rebalance + 1)
        else:
            mu_daily = forecast(train, config.forecast_method, config.rebalance, config.neural_epochs, config.seed)
        selected = np.argsort(-mu_daily, kind="stable")[:config.top_k or n]
        mu = mu_daily[selected] * config.annualization
        cov = covariance(train[:, selected], config.annualization)
        reference = np.full(len(selected), 1 / len(selected))
        bl_mu, bl_cov = black_litterman(cov, reference, mu - config.risk_free)
        bl_mu += config.risk_free
        allocations = {"Equal weight": np.full(n, 1 / n)}
        fallback = {}
        target_audit = {}
        for name, means, risk, objective in [
            ("Minimum variance", mu, cov, "min_variance"),
            ("Forecast Markowitz", mu, cov, "max_sharpe"),
            ("Black-Litterman", bl_mu, bl_cov, "max_sharpe"),
        ]:
            target_audit[name] = {"requested_target": np.nan, "effective_target": np.nan, "target_clipped": False}
            if config.allocation == "target_return" and name != "Minimum variance":
                from scipy.optimize import linprog
                maximum = linprog(-means, A_eq=[np.ones(len(means))], b_eq=[1],
                                  bounds=[(0, config.max_weight)] * len(means), method="highs")
                if not maximum.success:
                    raise RuntimeError("Could not determine feasible target ceiling")
                ceiling = float(means @ maximum.x)
                # An infeasible target is reduced to the feasible ceiling and logged.
                target = min(config.target_return, ceiling - 1e-8)
                local = optimize(means, risk, "min_variance", config.risk_free, config.max_weight, target=target)
                target_audit[name] = {"requested_target": config.target_return, "effective_target": target,
                                      "target_clipped": config.target_return > ceiling}
                fallback[name] = False
            else:
                try:
                    local = optimize(means, risk, objective, config.risk_free, config.max_weight)
                    fallback[name] = False
                except ValueError as exc:
                    if "positive expected excess return" not in str(exc):
                        raise
                    local = optimize(means, risk, max_weight=config.max_weight)
                    fallback[name] = True
            target_audit[name]["expected_return"] = float(means @ local)
            allocations[name] = np.zeros(n)
            allocations[name][selected] = local
        stop = min(start + config.rebalance, len(a))
        decision_date = returns.index[start - 2]
        execution_date = returns.index[start - 1]
        for i, asset in enumerate(names):
            record = {"decision_date": decision_date, "execution_date": execution_date,
                      "asset": asset, "predicted_daily_return": mu_daily[i],
                      "realized_daily_return": a[start:stop, i].mean(),
                      "holding_days": stop - start, "selected": i in selected}
            if price_prediction is not None:
                anchor = float(prices.iloc[start - 1, i])
                record.update(anchor_price=anchor, predicted_price=price_prediction.prices[i],
                              realized_date=prices.index[stop],
                              realized_price=float(prices.iloc[stop, i]),
                              predicted_horizon_return=price_prediction.prices[i] / anchor - 1,
                              full_forecast_horizon=(stop - start == config.rebalance),
                              **price_prediction.audit)
            forecast_rows.append(record)
        costs = {}
        for name in strategies:
            target = allocations[name]
            gross_traded = float(np.abs(target - holdings[name]).sum())
            costs[name] = gross_traded * config.cost_bps / 10000
            trade_rows.append({"execution_date": execution_date, "strategy": name,
                               "gross_traded_fraction": gross_traded, "cost_fraction": costs[name],
                               "min_variance_fallback": fallback.get(name, False),
                               **target_audit.get(name, {"requested_target": np.nan, "effective_target": np.nan, "target_clipped": False})})
            weight_rows.append({"execution_date": execution_date, "strategy": name,
                                **dict(zip(names, target))})
            holdings[name] = target.copy()
        for t in range(start, stop):
            row = {"date": returns.index[t]}
            for name in strategies:
                w = holdings[name]
                gross = float(w @ a[t])
                cost = costs[name] if t == start else 0.0
                row[name] = (1 - cost) * (1 + gross) - 1
                holdings[name] = w * (1 + a[t]) / (1 + gross)
            net.append(row)
    net = pd.DataFrame(net).set_index("date")
    metrics = pd.DataFrame({name: performance(net[name], config.annualization, config.risk_free)
                            for name in strategies}).T
    return BacktestResult(net, pd.DataFrame(weight_rows), pd.DataFrame(trade_rows),
                          pd.DataFrame(forecast_rows), metrics, asdict(config))
