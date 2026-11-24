"""Strict input validation: silently filling prices can change the experiment."""
from pathlib import Path
import numpy as np
import pandas as pd


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices require a DatetimeIndex.")
    if prices.index.hasnans or prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Dates must be valid, unique, and sorted ascending.")
    if prices.shape[0] < 4 or prices.shape[1] < 2:
        raise ValueError("Supply at least four dates and two assets.")
    if prices.columns.has_duplicates:
        raise ValueError("Asset names must be unique.")
    try:
        prices = prices.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Every price must be numeric.") from exc
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite, positive, and complete; align your data first.")
    return prices


def load_prices(path: str | Path) -> pd.DataFrame:
    # Check the raw header before pandas can rename duplicate columns.
    import csv
    with open(path, newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if not header or header[0] != "date" or len(set(header)) != len(header):
        raise ValueError("CSV needs a 'date' first column and unique asset names.")
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return validate_prices(frame)


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return validate_prices(prices).pct_change(fill_method=None).iloc[1:]


def synthetic_prices(seed: int = 42, observations: int = 1009, assets: int = 8) -> pd.DataFrame:
    """Factor-driven artificial prices; no tickers or observed market history."""
    if observations < 4 or assets < 2:
        raise ValueError("Need at least four observations and two assets.")
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, [0.007, 0.004], size=(observations - 1, 2))
    exposures = rng.uniform(0.2, 1.2, (2, assets))
    noise = rng.normal(0, 0.006, (observations - 1, assets))
    log_returns = factors @ exposures + noise + np.linspace(0.0001, 0.0003, assets)
    prices = np.vstack([np.ones(assets), np.exp(np.cumsum(log_returns, axis=0))]) * 100
    return pd.DataFrame(prices, index=pd.bdate_range("2020-01-01", periods=observations, name="date"),
                        columns=[f"Asset_{i + 1:02}" for i in range(assets)])
