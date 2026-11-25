# Historical SET50 cohort: data provenance

The universe is the 50 names in SET's constituent announcement published **15 December 2014**, effective **1 January–30 June 2015**. The list was known before the study began. [Source: SET announcement hosted by PTG](https://ptg.listedcompany.com/misc/SET100-50.pdf). Only the symbols and source link are stored here; the PDF is not included.

## Coverage

The downloaded study contains **41 stocks and 1,462 daily observations**, from **5 January 2015 through 30 December 2020**. It uses Yahoo Finance adjusted closing prices in provider-adjusted THB units. Downloaded Close and Adj Close fields are retained separately in local files. Close can itself reflect splits; the study does not call it an unadjusted historical trade price.

Of the original 50 symbols, eight could not be retrieved for this interval: **BIGC, DTAC, GLOW, HEMRAJ, INTUCH, PS, ROBINS, and SCB**. BANPU was available only from October 2016 and was excluded by the full-period coverage policy. No modern replacement companies or invented price series were substituted.

The fixed cohort is **not** a reconstruction of changing SET50 membership. Restricting the analysis to histories that remain downloadable introduces availability/survivorship bias. Historical prices are the provider's current adjusted vintage, not a point-in-time archive of each day's adjustments. These are material limitations when interpreting performance.

## Reproduce

```bash
python scripts/download_set50.py
```

The script requests `2015-01-01 <= date < 2021-01-01`, writes cached responses under `raw/`, and creates `adjusted_prices.csv`. Raw provider responses and price panels are local, git-ignored inputs; the repository publishes the downloader, source manifest, hashes, audit tables and derived experiment outputs. A fresh download may differ if the provider revises its history; compare SHA-256 values before claiming an exact reproduction.

## Cleaning rules

1. Convert timestamps to Bangkok trading dates, sort them and require unique dates.
2. Treat nonfinite/nonpositive prices as missing. Use the union of observed trading dates, not an artificial weekday calendar.
3. Require at least 98% observed coverage and an observed first and final price.
4. Forward-fill at most two internal observations. Never backfill an IPO history or extend a series beyond its terminal observation. Exclude unresolved series rather than silently deleting trading dates.
5. Flag absolute daily moves above 50% for review. Do not erase large genuine market moves as statistical outliers.

The accepted panel had no missing values requiring filling and no >50% daily-return flags. Cleaning is nevertheless implemented and tested on deliberately incomplete fixtures. The frozen inclusion decision uses full-period coverage and therefore is not itself a point-in-time trading rule.

## Symbol mappings

Historical symbols remain the column labels. Three verified symbol changes are used only for retrieval:

| Historical symbol | Provider symbol | Source |
| --- | --- | --- |
| BGH | BDMS.BK | [Company announcements, 2015](https://bdms.listedcompany.com/newsroom_set.html/year/2015) |
| TMB | TTB.BK | [SET company profile and symbol history](https://www.set.or.th/en/market/product/stock/quote/ttb/company-profile/information) |
| TUF | TU.BK | [Thai Union corporate history](https://www.thaiunion.com/our-company/milestone-establishment) |

`universe_2015.csv` lists every requested name. `download_audit.csv` records the provider URL, response hash, date coverage and errors. `cleaning_audit.csv` records inclusion decisions. `manifest.json` identifies the exact cleaned panel used for the published study.
