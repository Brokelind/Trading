import os
import time
import logging
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import yfinance as yf

# Local-only env support
try:
    import env
except ImportError:
    env = None

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY") or getattr(env, "ALPACA_API_KEY", None)
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY") or getattr(env, "ALPACA_SECRET_KEY", None)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Initialize Alpaca data client (only need data API here)
client = (StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
          if ALPACA_API_KEY and ALPACA_SECRET_KEY else None)


def normalize_bars(df):
    """Store Yahoo and Alpaca bars using the model's single-header schema."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
    if 'adj_close' not in df.columns and 'close' in df.columns:
        df.rename(columns={'close': 'adj_close'}, inplace=True)
    columns = ['open', 'high', 'low', 'adj_close', 'volume']
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing market data columns: {sorted(missing)}")
    df = df[columns].apply(pd.to_numeric, errors='coerce').dropna()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
    df = df[~df.index.duplicated(keep='last')].sort_index()
    if df.empty or (df['adj_close'] <= 0).any():
        raise ValueError('No valid market prices')
    df.index.name = 'Date'
    return df


def get_data(symbol, save_folder="data"):
    # Retry Yahoo Finance a few times in case of rate limiting/temporary failures.
    df = None
    last_err = None
    for attempt in range(3):
        try:
            df = yf.download(symbol, period="10y", interval="1d", auto_adjust=True, progress=False)
            if df is None or df.empty:
                raise ValueError("Empty DataFrame from Yahoo")
            df = normalize_bars(df)
            log.info("Fetched from Yahoo Finance")
            break
        except Exception as e:
            last_err = e
            log.warning(f"Yahoo Finance failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    else:
        log.warning(f"Yahoo Finance failed after retries: {last_err}, falling back to Alpaca.")
        try:
            if client is None:
                raise ValueError("Alpaca credentials unavailable")
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start="2000-01-01",
                adjustment="all"
            )
            bars = client.get_stock_bars(request).df
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.loc[symbol]
            candidate = bars[['open', 'high', 'low', 'close', 'volume']].copy()
            candidate.rename(columns={"close": "adj_close"}, inplace=True)
            if candidate is None or candidate.empty:
                raise ValueError("Empty DataFrame from Alpaca")
            df = normalize_bars(candidate)
        except Exception as e2:
            log.error(f"Alpaca also failed: {e2}")
            return

    df.index.name = "Date"
    os.makedirs(save_folder, exist_ok=True)
    path = os.path.join(save_folder, f"{symbol}_data.csv")
    temporary_path = path + '.tmp'
    df.to_csv(temporary_path, index=True, date_format="%Y-%m-%d")
    os.replace(temporary_path, path)
    return path

