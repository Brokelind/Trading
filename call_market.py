import os
import time
import logging
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

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
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period="max", auto_adjust=True)
        if df.empty:
            raise ValueError("Empty DataFrame from Yahoo")
        log.info("Fetched from Yahoo Finance")
    except Exception as e:
        log.warning(f"Yahoo Finance failed: {e}, falling back to Alpaca.")
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start="2010-01-01"  # limited depth
            )
            bars = client.get_stock_bars(request).df
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.loc[symbol]
            df = bars[['open', 'high', 'low', 'close']].copy()
            df.rename(columns={"close": "adj_close"}, inplace=True)
        except Exception as e2:
            log.error(f"Alpaca also failed: {e2}")
            return

    df.index.name = "Date"
    os.makedirs(save_folder, exist_ok=True)
    path = os.path.join(save_folder, f"{symbol}_data.csv")
    df.to_csv(path, index=True, date_format="%Y-%m-%d")
    return path

