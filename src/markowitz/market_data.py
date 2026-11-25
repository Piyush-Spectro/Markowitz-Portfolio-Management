"""Public-provider download and explicit, auditable price cleaning."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote
import numpy as np
import pandas as pd

UNIVERSE_SOURCE = 'https://ptg.listedcompany.com/misc/SET100-50.pdf'


def fetch_chart(symbol, start='2015-01-01', end='2021-01-01'):
    t1, t2 = (int(pd.Timestamp(value, tz='UTC').timestamp()) for value in [start, end])
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}?period1={t1}&period2={t2}&interval=1d&events=div%2Csplits'
    request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(request, timeout=45) as response:
        payload = response.read()
    body = json.loads(payload)
    if body['chart'].get('error') or not body['chart'].get('result'):
        raise ValueError(f'No chart data for {symbol}')
    return payload, url


def chart_to_prices(payload):
    chart = json.loads(payload)['chart']['result'][0]
    if not chart.get('timestamp'):
        raise ValueError('No historical timestamps returned')
    index = pd.to_datetime(chart['timestamp'], unit='s', utc=True).tz_convert('Asia/Bangkok').tz_localize(None).normalize()
    quote_data = chart['indicators']['quote'][0]
    adjusted = chart['indicators'].get('adjclose', [{}])[0].get('adjclose')
    if adjusted is None:
        raise ValueError('Adjusted-close field is unavailable; no unadjusted substitution')
    return pd.DataFrame({'adjusted': adjusted, 'close': quote_data['close'],
                         'volume': quote_data.get('volume')}, index=index)


def clean_panel(panel, max_gap=2, min_coverage=0.98):
    """Forward-fill only short internal gaps; never invent a start or terminal price."""
    if panel.index.has_duplicates or not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError('Input dates must be unique timestamps')
    if max_gap < 0 or not 0 < min_coverage <= 1:
        raise ValueError('Invalid cleaning policy')
    panel = panel.sort_index().replace([np.inf, -np.inf], np.nan)
    panel = panel.where(panel > 0).dropna(how='all')
    kept, audit = {}, []
    for name in panel:
        series = panel[name]
        missing = int(series.isna().sum())
        coverage = float(series.notna().mean())
        reason = ''
        if coverage < min_coverage:
            reason = 'coverage_below_threshold'
        elif pd.isna(series.iloc[0]) or pd.isna(series.iloc[-1]):
            reason = 'missing_start_or_end'
        filled = series.ffill(limit=max_gap) if max_gap else series.copy()
        if not reason and filled.isna().any():
            reason = 'gap_exceeds_fill_limit'
        filled_count = int((series.isna() & filled.notna()).sum()) if not reason else 0
        jumps = int((filled.pct_change(fill_method=None).abs() > 0.5).sum()) if not reason else 0
        audit.append({'symbol': name, 'observed': int(series.notna().sum()), 'missing': missing,
                      'coverage': coverage, 'forward_filled': filled_count,
                      'large_return_flags': jumps, 'included': not bool(reason), 'exclusion_reason': reason})
        if not reason:
            kept[name] = filled
    clean = pd.DataFrame(kept, index=panel.index)
    if clean.shape[1] < 10:
        raise ValueError(f'Only {clean.shape[1]} assets survived cleaning; need at least ten.')
    return clean, pd.DataFrame(audit)


def download_universe(universe_path, destination):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    universe = pd.read_csv(universe_path).fillna('')
    destination = Path(destination)
    raw_dir = destination / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs, series, closes = [], {}, {}

    def get(row):
        cached = raw_dir / f'{row.provider_symbol}.json'
        if cached.exists():
            payload = cached.read_bytes()
            url = f'https://query1.finance.yahoo.com/v8/finance/chart/{row.provider_symbol}'
        else:
            payload, url = fetch_chart(row.provider_symbol)
            cached.write_bytes(payload)
        frame = chart_to_prices(payload)
        frame = frame.loc['2015-01-01':'2020-12-31']
        if frame.empty:
            raise ValueError('No observations in the study period')
        return frame, url, hashlib.sha256(payload).hexdigest()

    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = {executor.submit(get, row): row for row in universe.itertuples()}
        for job in as_completed(jobs):
            row = jobs[job]
            record = {'symbol': row.symbol, 'provider_symbol': row.provider_symbol}
            try:
                frame, url, digest = job.result()
                series[row.symbol] = frame.adjusted
                closes[row.symbol] = frame.close
                record.update(status='downloaded', rows=len(frame), first=str(frame.index.min().date()),
                              last=str(frame.index.max().date()), sha256=digest, source=url)
            except Exception as exc:
                record.update(status='unavailable', error=str(exc))
            logs.append(record)
            print(f'{row.symbol}: {record["status"]}', flush=True)
    panel = pd.DataFrame(series).sort_index().reindex(columns=[s for s in universe.symbol if s in series])
    clean, audit = clean_panel(panel)
    clean.index.name = 'date'
    clean.to_csv(destination / 'adjusted_prices.csv', float_format='%.10f')
    pd.DataFrame(closes).reindex(index=clean.index, columns=clean.columns).to_csv(destination / 'raw_close.csv')
    audit.to_csv(destination / 'cleaning_audit.csv', index=False)
    pd.DataFrame(logs).sort_values('symbol').to_csv(destination / 'download_audit.csv', index=False)
    manifest = {'provider': 'Yahoo Finance chart API', 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                'start_inclusive': '2015-01-01', 'end_exclusive': '2021-01-01',
                'universe_source': UNIVERSE_SOURCE, 'universe_effective': '2015-01-01 to 2015-06-30',
                'universe_policy': 'Fixed initial-2015 constituent cohort; downloadable coverage subset, not a changing-index replication',
                'requested_assets': len(universe), 'downloaded_assets': len(series), 'included_assets': list(clean.columns),
                'observations': len(clean), 'first_date': str(clean.index.min().date()), 'last_date': str(clean.index.max().date()),
                'cleaning': {'forward_fill_limit': 2, 'minimum_coverage': 0.98, 'backfill': False,
                             'outliers': 'Flag absolute daily returns above 50%; retain genuine price moves'},
                'adjustment': 'Provider adjusted close; historical split/dividend adjustments as retrieved, not a point-in-time vintage',
                'cleaned_sha256': hashlib.sha256((destination / 'adjusted_prices.csv').read_bytes()).hexdigest()}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Clean panel: {len(clean)} dates, {clean.shape[1]} assets')
    return clean
