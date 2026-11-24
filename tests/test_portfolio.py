import numpy as np
import pytest
from markowitz.portfolio import optimize, efficient_frontier, covariance, black_litterman


def test_min_variance_matches_closed_form_diagonal_solution():
    cov = np.diag([0.04, 0.09, 0.16])
    expected = 1 / np.diag(cov)
    expected /= expected.sum()
    np.testing.assert_allclose(optimize([0.1, 0.12, 0.15], cov), expected, atol=1e-5)


def test_tangency_matches_unconstrained_positive_solution():
    cov = np.diag([0.04, 0.09, 0.16])
    mu = np.array([0.10, 0.13, 0.18])
    expected = np.linalg.solve(cov, mu - 0.02)
    expected /= expected.sum()
    np.testing.assert_allclose(optimize(mu, cov, "max_sharpe"), expected, atol=1e-5)


def test_cap_and_frontier():
    mu, cov = np.array([0.04, 0.09, 0.16]), np.diag([0.03, 0.06, 0.1])
    w = optimize(mu, cov, "max_sharpe", max_weight=0.45)
    assert w.max() <= 0.45 + 1e-8
    frontier = efficient_frontier(mu, cov, points=10, max_weight=0.45)
    assert np.all(np.diff(frontier['return']) >= -1e-7)
    assert np.all(np.diff(frontier.volatility) >= -1e-7)


@pytest.mark.parametrize('kwargs', [{"max_weight": 0.2}, {"target": 0.9}, {"objective": "bad"}])
def test_invalid_optimization(kwargs):
    with pytest.raises(ValueError):
        optimize([0.1, 0.2], np.eye(2), **kwargs)


def test_negative_premium_rejected():
    with pytest.raises(ValueError, match="positive expected"):
        optimize([-0.2, -0.1], np.eye(2), "max_sharpe")


def test_bl_one_absolute_view_matches_scalar_bayes():
    cov = np.diag([0.04, 0.09])
    prior = 2.5 * cov @ [0.5, 0.5]
    posterior, predictive = black_litterman(cov, [0.5, 0.5], [0.10], pick=[[1, 0]], omega=[[0.002]])
    # Prior variance tau*0.04 = 0.002, matching observation variance.
    assert posterior[0] == pytest.approx((prior[0] + 0.10) / 2)
    assert posterior[1] == pytest.approx(prior[1])
    assert predictive[0, 0] == pytest.approx(0.041)


def test_bl_weak_views_recover_prior_and_strong_views_dominate():
    cov = np.diag([0.04, 0.09])
    prior = 2.5 * cov @ [0.5, 0.5]
    weak, _ = black_litterman(cov, [0.5, 0.5], [0.2, 0.3], omega=np.eye(2) * 1e10)
    strong, _ = black_litterman(cov, [0.5, 0.5], [0.2, 0.3], omega=np.eye(2) * 1e-12)
    np.testing.assert_allclose(weak, prior)
    np.testing.assert_allclose(strong, [0.2, 0.3])


def test_covariance_handles_duplicate_and_constant_series():
    x = np.column_stack([np.arange(20) / 100, np.arange(20) / 100, np.zeros(20)])
    assert np.linalg.eigvalsh(covariance(x)).min() > 0


def test_bl_rejects_invalid_uncertainty():
    with pytest.raises(ValueError):
        black_litterman(np.eye(2), [0.5, 0.5], [0.1, 0.2], omega=-np.eye(2))
