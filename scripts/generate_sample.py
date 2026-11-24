"""Recreate the artificial sample shipped with the repository."""
from pathlib import Path
from markowitz.data import synthetic_prices

if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "data" / "sample_prices.csv"
    synthetic_prices().to_csv(destination, float_format="%.8f")
    print(destination)
