import pandas as pd
import yfinance as yf
import os
import time


def get_data(symbol, save_folder="data"):
    try:
        # More stable method
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max", auto_adjust=True)
    except Exception as e:
        print(f"[ERROR] Failed to get ticker '{symbol}': {e}")
        return

    if df.empty:
        print(f"[ERROR] No data fetched for {symbol}")
        return

    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.rename(columns={"Close": "adj_close", "Open": "open", "High": "high", "Low": "low"}, inplace=True)
    df.index.name = "Date"

    print(f"////////////new stock {symbol}/////////")
    print("Earliest date in data:", df.index.min())
    print("Latest date in data:", df.index.max())
    print(f"Total rows: {len(df)}")

    os.makedirs(save_folder, exist_ok=True)
    path = os.path.join(save_folder, f"{symbol}_data.csv")
    df.to_csv(path, index=True, date_format="%Y-%m-%d")
    print(f"Saved data for {symbol} to {path}")
    time.sleep(1)  # Avoid Yahoo rate-limiting
    return path
