"""Long-only portfolio construction and a linear-Gaussian Bayesian update."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog


def covariance(returns, annualization=252, shrinkage=0.15):
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or len(x) < 3 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Covariance needs a finite return matrix with at least three rows and two assets.")
    if annualization <= 0 or not 0 <= shrinkage <= 1:
        raise ValueError("Invalid annualization or shrinkage.")
    sample = np.cov(x, rowvar=False, ddof=1) * annualization
    result = (1 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample))
    # A small diagonal floor also handles constant or duplicate series.
    return result + np.eye(x.shape[1]) * 1e-10


def _moments(mu, cov):
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    if mu.ndim != 1 or cov.shape != (len(mu), len(mu)) or len(mu) < 2:
        raise ValueError("Mean and covariance dimensions do not agree.")
    if not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise ValueError("Moments must be finite.")
    if not np.allclose(cov, cov.T, atol=1e-10) or np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError("Covariance must be symmetric positive definite.")
    return mu, cov


def optimize(mu, cov, objective="min_variance", risk_free=0.02, max_weight=1.0, target=None):
    mu, cov = _moments(mu, cov)
    n = len(mu)
    if not np.isfinite(max_weight) or not 1 / n - 1e-12 <= max_weight <= 1:
        raise ValueError("Weight cap is infeasible for the selected asset count.")
    if not np.isfinite(risk_free):
        raise ValueError("Risk-free rate must be finite.")
    bounds = [(0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target is not None:
        if not np.isfinite(target):
            raise ValueError("Target must be finite.")
        feasible = linprog(-mu, A_eq=[np.ones(n)], b_eq=[1], bounds=bounds, method="highs")
        if not feasible.success or target > mu @ feasible.x + 1e-9:
            raise ValueError("Target return is infeasible under the weight cap.")
        constraints.append({"type": "ineq", "fun": lambda w: mu @ w - target})
    scaled_cov = cov / max(float(np.diag(cov).mean()), 1e-12)
    if objective == "min_variance":
        fit = minimize(lambda w: w @ scaled_cov @ w, np.full(n, 1 / n),
                       jac=lambda w: 2 * scaled_cov @ w, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"ftol": 1e-11, "maxiter": 1000})
        w = fit.x
    elif objective == "max_sharpe":
        feasible = linprog(-mu, A_eq=[np.ones(n)], b_eq=[1], bounds=bounds, method="highs")
        if not feasible.success or mu @ feasible.x <= risk_free + 1e-12:
            raise ValueError("No feasible portfolio has positive expected excess return.")
        # For positive excess return, tangency is a convex quadratic problem:
        # minimize y' Sigma y with excess' y = 1; recover w = y / sum(y).
        # Caps become linear constraints y_i <= cap * sum(y).
        excess = mu - risk_free
        excess = excess / max(float(np.abs(excess).max()), 1e-12)
        initial = feasible.x / (excess @ feasible.x)
        cap_matrix = max_weight * np.ones((n, n)) - np.eye(n)
        transformed = [
            {"type": "eq", "fun": lambda y: excess @ y - 1, "jac": lambda y: excess},
            {"type": "ineq", "fun": lambda y: cap_matrix @ y, "jac": lambda y: cap_matrix},
        ]
        if target is not None:
            transformed.append({"type": "ineq", "fun": lambda y: (mu - target) @ y,
                                "jac": lambda y: mu - target})
        fit = minimize(lambda y: y @ scaled_cov @ y, initial,
                       jac=lambda y: 2 * scaled_cov @ y, method="SLSQP", bounds=[(0, None)] * n,
                       constraints=transformed, options={"ftol": 1e-10, "maxiter": 2000})
        w = fit.x / fit.x.sum()
    else:
        raise ValueError("Unknown optimization objective.")
    if (not fit.success or not np.isfinite(w).all() or abs(w.sum() - 1) > 1e-7 or w.min() < -1e-7
            or w.max() > max_weight + 1e-7 or (target is not None and mu @ w < target - 1e-7)):
        raise RuntimeError(f"Portfolio optimization failed: {fit.message}")
    return w


def efficient_frontier(mu, cov, points=40, max_weight=1.0):
    mu, cov = _moments(mu, cov)
    if points < 2:
        raise ValueError("Need at least two frontier points.")
    minimum = optimize(mu, cov, max_weight=max_weight)
    maximum = linprog(-mu, A_eq=[np.ones(len(mu))], b_eq=[1],
                      bounds=[(0, max_weight)] * len(mu), method="highs")
    if not maximum.success:
        raise RuntimeError("Could not find maximum feasible return.")
    rows = []
    for target in np.linspace(mu @ minimum, mu @ maximum.x, points):
        w = optimize(mu, cov, target=float(target), max_weight=max_weight)
        rows.append({"return": mu @ w, "volatility": np.sqrt(w @ cov @ w)})
    return pd.DataFrame(rows)


def black_litterman(cov, reference_weights, views, pick=None, omega=None,
                    risk_aversion=2.5, tau=0.05):
    """Return posterior excess means and predictive covariance, in annual units.

    The reference allocation is an explicit benchmark, not necessarily market caps.
    `views` are annual excess returns; rows of `pick` define absolute/relative views.
    """
    weights = np.asarray(reference_weights, float)
    _, cov = _moments(weights, cov)
    if (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Reference weights must be nonnegative and sum to one.")
    if not np.isfinite([tau, risk_aversion]).all() or tau <= 0 or risk_aversion <= 0:
        raise ValueError("tau and risk aversion must be positive.")
    p = np.eye(len(weights)) if pick is None else np.asarray(pick, float)
    q = np.asarray(views, float)
    if p.ndim != 2 or p.shape[1] != len(weights) or q.shape != (len(p),):
        raise ValueError("View dimensions do not agree.")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Views must be finite.")
    prior = risk_aversion * cov @ weights
    uncertainty = tau * cov
    if len(q) == 0:
        return prior, cov + uncertainty
    o = np.diag(np.maximum(np.diag(p @ uncertainty @ p.T), 1e-12)) if omega is None else np.asarray(omega, float)
    if (o.shape != (len(q), len(q)) or not np.isfinite(o).all()
            or not np.allclose(o, o.T) or np.linalg.eigvalsh(o).min() <= 0):
        raise ValueError("View uncertainty must be symmetric positive definite.")
    gain = np.linalg.solve(p @ uncertainty @ p.T + o, p @ uncertainty).T
    posterior = prior + gain @ (q - p @ prior)
    mean_cov = uncertainty - gain @ p @ uncertainty
    predictive = cov + (mean_cov + mean_cov.T) / 2
    return posterior, predictive
