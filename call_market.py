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

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("Missing Alpaca API credentials.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Initialize Alpaca data client (only need data API here)
client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def get_data(symbol, save_folder="data"):
    # Retry Yahoo Finance a few times in case of rate limiting/temporary failures.
    df = None
    last_err = None
    for attempt in range(3):
        try:
            df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
            if df is None or df.empty:
                raise ValueError("Empty DataFrame from Yahoo")
            log.info("Fetched from Yahoo Finance")
            break
        except Exception as e:
            last_err = e
            log.warning(f"Yahoo Finance failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    else:
        log.warning(f"Yahoo Finance failed after retries: {last_err}, falling back to Alpaca.")
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start="2000-01-01"
            )
            bars = client.get_stock_bars(request).df
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.loc[symbol]
            candidate = bars[['open', 'high', 'low', 'close']].copy()
            candidate.rename(columns={"close": "adj_close"}, inplace=True)
            if candidate is None or candidate.empty:
                raise ValueError("Empty DataFrame from Alpaca")
            df = candidate
        except Exception as e2:
            log.error(f"Alpaca also failed: {e2}")
            return

    df.index.name = "Date"
    os.makedirs(save_folder, exist_ok=True)
    path = os.path.join(save_folder, f"{symbol}_data.csv")
    df.to_csv(path, index=True, date_format="%Y-%m-%d")
    return path

