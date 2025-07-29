import pandas as pd
import requests
import yfinance as yf
import os


def get_data(symbol, save_folder="data"):
    # Download max period data with auto_adjust True
    df = yf.download(symbol, period="max", progress=False, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    df = df[df.index.to_series().apply(lambda x: isinstance(x, pd.Timestamp))]

  
    if df.empty:
        raise ValueError(f"No data fetched for {symbol}")

    # Identify wanted columns
    wanted_cols = []
    for col_base in [f'Open_{symbol}', f'High_{symbol}', f'Low_{symbol}', f'Close_{symbol}']:
        candidates = [
            col_base,
            f'Price_{col_base}',
            f'{symbol}_{col_base}'
        ]
        for c in candidates:
            if c in df.columns:
                wanted_cols.append(c)
                break

    if not wanted_cols:
        raise ValueError(f"No usable price columns found for {symbol}")

    # Rename
    rename_dict = {old: old.split('_')[0].lower() for old in wanted_cols}
    df = df[wanted_cols].rename(columns=rename_dict)
    df = df.rename(columns={"close": "adj_close"})

    df.index.name = "Date"

    print(f"////////////new stock {symbol}/////////")
    print("Earliest date in data:", df.index.min())
    print("Latest date in data:", df.index.max())
    print(f"Total rows: {len(df)}")
    

    os.makedirs(save_folder, exist_ok=True)
    path = os.path.join(save_folder, f"{symbol}_data.csv")
    df.to_csv(path, index=True, date_format="%Y-%m-%d")
    print(f"Saved data for {symbol} to {path}")
    return path
