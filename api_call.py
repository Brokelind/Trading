import pandas as pd
import requests
import env
import yfinance as yf
import os


API_KEY  = env.key


def get_data(symbol, start="2003-01-01", save_folder="data"):
    # Download max period data with auto_adjust True (adjusted prices)
    df = yf.download(symbol, period="max", progress=False, auto_adjust=True)

    # Flatten multiindex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Drop any rows where index is not a Timestamp (removes junk rows)
    df = df[df.index.to_series().apply(lambda x: isinstance(x, pd.Timestamp))]

    if df.empty:
        raise ValueError(f"No data fetched for {symbol}")

    # Now find the close column name (usually it will be 'Close' or 'Price_Close')
    # Try common options:
    close_col = None
    for col_candidate in ['Close', f'Price_Close', f'{symbol}_Close']:
        if col_candidate in df.columns:
            close_col = col_candidate
            break
    if close_col is None:
        # fallback: pick first column containing 'Close'
        close_cols = [col for col in df.columns if 'Close' in col]
        if close_cols:
            close_col = close_cols[0]
        else:
            raise ValueError("No Close price column found in data")

    # Select close column and rename
    df = df[[close_col]].rename(columns={close_col: "adj_close"})
    df.index.name = "Date"
    print(f"////////////new stock {symbol}/////////")
    print("Earliest date in data:", df.index.min())
    print("Latest date in data:", df.index.max())
    print(f"Total rows: {len(df)}")
    print(df.head())

    # Save to CSV
    os.makedirs(save_folder, exist_ok=True)
    path = os.path.join(save_folder, f"{symbol}_data.csv")
    df.to_csv(path, index=True, date_format="%Y-%m-%d")
    print(f"Saved data for {symbol} to {path}")
    return path
