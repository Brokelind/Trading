# crypto_correlation.py
import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# List of cryptocurrencies to trade (sub-coins) and the lead coins (BTC and ETH)
LEAD_COINS = ["BTC-USD", "ETH-USD"]
SUB_COINS = [
    "SOL-USD", "ADA-USD", "DOT-USD", "LINK-USD", "DOGE-USD", "XRP-USD", "LTC-USD"
]

def fetch_hourly_data(tickers, days=5):
    """Fetch hourly data for specified tickers from Yahoo Finance"""
    data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"[INFO] Fetching hourly data from {start_date.date()} to {end_date.date()}...")
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False, auto_adjust=True)
            if not df.empty:
                data[ticker] = df
                print(f"[OK] Fetched {len(df)} rows for {ticker}")
            else:
                print(f"[WARN] Empty data returned for {ticker}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {ticker}: {e}")
    return data

def generate_signals():
    """Analyze lead-lag relationship and generate signals"""
    print("\n--- Running Crypto Lead-Lag Correlation Strategy ---")
    
    all_tickers = LEAD_COINS + SUB_COINS
    data = fetch_hourly_data(all_tickers)
    
    if len(data) < len(all_tickers) * 0.5:
        print("[ERROR] Insufficient crypto data fetched. Aborting strategy.")
        return []
        
    signals = []
    
    # Analyze price changes over a 3-hour window
    window_hours = 3
    
    # Calculate average leader movement
    leader_moves = []
    for leader in LEAD_COINS:
        if leader in data:
            df = data[leader]
            if len(df) >= window_hours:
                # Flatten multi-level columns if needed
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                close_now = float(df['Close'].iloc[-1])
                close_prev = float(df['Close'].iloc[-(window_hours + 1)])
                pct_change = (close_now - close_prev) / close_prev * 100
                leader_moves.append(pct_change)
                
    if not leader_moves:
        print("[ERROR] No leader coin data available.")
        return []
        
    avg_leader_move = np.mean(leader_moves)
    print(f"[INFO] Leader (BTC/ETH) average movement over last {window_hours}h: {avg_leader_move:+.2f}%")
    
    # Define thresholds
    threshold_pct = 1.25  # Leader must move at least this much
    lag_threshold_pct = 0.40  # Sub-coin must lag by this much of the leader move (or have moved less than this)
    
    for sub in SUB_COINS:
        if sub not in data:
            continue
            
        df = data[sub]
        if len(df) < window_hours:
            continue

        # Flatten multi-level columns if yfinance returned them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close_now = float(df['Close'].iloc[-1])
        close_prev = float(df['Close'].iloc[-(window_hours + 1)])
        sub_move = (close_now - close_prev) / close_prev * 100
        
        signal = "HOLD"
        pct_diff = 0.0
        confidence = 0.0
        
        # Lead-lag check:
        # If leaders went up and sub-coin is lagging, BUY
        if avg_leader_move > threshold_pct:
            if sub_move < lag_threshold_pct:
                signal = "BUY"
                pct_diff = avg_leader_move - sub_move  # Potential gap to close
                confidence = min(abs(pct_diff) / 5.0, 1.0)
                
        # If leaders went down and sub-coin is lagging, SELL
        elif avg_leader_move < -threshold_pct:
            if sub_move > -lag_threshold_pct:
                signal = "SELL"
                pct_diff = avg_leader_move - sub_move  # Potential gap to close
                confidence = min(abs(pct_diff) / 5.0, 1.0)
                
        print(f"[SCAN] {sub:8s} | Return: {sub_move:+.2f}% | Gap to Leader: {pct_diff:+.2f}% | Signal: {signal}")
        
        if signal != "HOLD":
            signals.append({
                "ticker": sub,
                "timestamp": datetime.utcnow().isoformat(),
                "last_price": float(close_now),
                "leader_move": float(avg_leader_move),
                "sub_move": float(sub_move),
                "signal": signal,
                "pct_diff": float(pct_diff),
                "confidence": float(round(confidence, 2)),
                "chosen_model": "Lead-Lag Correlation"
            })
            
    # Save results to JSON
    fname = os.path.join(RESULTS_DIR, "crypto_signals.json")
    with open(fname, "w") as f:
        json.dump(signals, f, indent=2)
    print(f"[INFO] Wrote {len(signals)} crypto signals to {fname}")
    return signals

if __name__ == "__main__":
    generate_signals()
