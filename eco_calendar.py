"""
Fetch Bloomberg economic calendar (ECO) data and assemble it into a pandas DataFrame.
"""
import pandas as pd

from bloomberg_api import BloombergConnector

# Bloomberg "Index" tickers for common economic releases. Adjust to your watchlist.
DEFAULT_ECO_TICKERS = [
    "NFP TCH Index",     # US Non-Farm Payrolls
    "CPI YOY Index",      # US CPI YoY
    "GDP CQOQ Index",     # US GDP QoQ
    "FDTR Index",         # US Fed Funds Target Rate
    "USURTOT Index",      # US Unemployment Rate
    "ECCPEMUY Index",     # Eurozone CPI YoY
]

# Per-release fields: next release date/time, actual print, survey consensus, prior value.
ECO_FIELDS = [
    "ECO_RELEASE_DT",
    "ECO_RELEASE_TIME",
    "ACTUAL_RELEASE",
    "BN_SURVEY_MEDIAN",
    "BN_SURVEY_AVERAGE",
    "BN_SURVEY_HIGH",
    "BN_SURVEY_LOW",
    "PX_LAST",
    "ECO_PERIOD_FREQ",
]


def fetch_eco_calendar(connector: BloombergConnector, tickers=None, fields=None) -> pd.DataFrame:
    """BDP snapshot of ECO fields for a list of economic indicators, one row per ticker."""
    tickers = tickers or DEFAULT_ECO_TICKERS
    fields = fields or ECO_FIELDS

    raw = connector.bdp(tickers, fields)

    rows = []
    for ticker in tickers:
        row = {"ticker": ticker}
        for field in fields:
            row[field] = raw.get(field, {}).get(ticker)
        rows.append(row)

    df = pd.DataFrame(rows)
    if "ECO_RELEASE_DT" in df.columns:
        df["ECO_RELEASE_DT"] = pd.to_datetime(df["ECO_RELEASE_DT"], errors="coerce")
        df = df.sort_values("ECO_RELEASE_DT").reset_index(drop=True)
    return df


def fetch_future_releases(connector: BloombergConnector, ticker: str) -> pd.DataFrame:
    """Bulk field with the full future release-date schedule for one economic indicator."""
    df = connector.bds(ticker, "ECO_FUTURE_RELEASE_DATE_LIST")
    if not df.empty:
        df.insert(0, "ticker", ticker)
    return df


if __name__ == "__main__":
    bbg = BloombergConnector()
    bbg.connect()

    calendar_df = fetch_eco_calendar(bbg)
    print(calendar_df)

    bbg.disconnect()
