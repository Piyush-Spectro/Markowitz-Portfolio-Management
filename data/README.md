# Sample data

`sample_prices.csv` contains 1,009 artificial weekday price observations for eight fictional assets. It was generated with NumPy's default random generator, seed 42, using two shared Gaussian factors, asset-specific noise, and fixed drifts. Prices begin at 100.

The dates are an artificial business-day index, not an exchange calendar. The assets are not real securities; the series are not downloaded or calibrated to a historical market. There are no corporate actions, delistings, or liquidity constraints.

Regenerate the file from the repository root:

```bash
python scripts/generate_sample.py
```

For empirical analysis, supply daily adjusted prices from a source you have permission to use. Record the source, retrieval date, adjustment convention, trading calendar, and point-in-time universe selection alongside your experiment. Do not substitute fabricated ticker labels for the sample assets.
