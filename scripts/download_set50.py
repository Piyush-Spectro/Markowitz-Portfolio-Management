"""Download the fixed January 2015 SET50 cohort and create a cleaning audit."""
from pathlib import Path
from markowitz.market_data import download_universe

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    download_universe(root / 'data/set50/universe_2015.csv', root / 'data/set50')
