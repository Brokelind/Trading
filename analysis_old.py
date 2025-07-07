import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras import layers, models
import pandas_ta as ta
import os


def decide_on_signal(
    ticker,
    best_model_name,
    csv_results_file="strategy_performance.csv",
    save_signals=False,
    threshold=0.5,
):
    # Load data
    df = pd.read_csv(f"data\\{ticker}_data.csv", index_col=0, parse_dates=True)
    data = df[['adj_close']].values

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences for full training
    window_size = 20

    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_full, y_full = create_sequences(scaled_data, window_size)
    X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)

    # Create last input window for prediction
    last_window = scaled_data[-window_size:].reshape(1, window_size, 1)

    # Train the chosen model
    if best_model_name == "LSTM":
        model = models.Sequential([
            layers.LSTM(50, input_shape=(window_size, 1)),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_full, y_full, epochs=10, batch_size=16, verbose=0)
        pred_scaled = model.predict(last_window, verbose=0)
    elif best_model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=5)
        model.fit(X_full.reshape(X_full.shape[0], -1), y_full)
        pred_scaled = model.predict(last_window.reshape(1, -1)).reshape(-1, 1)
    elif best_model_name == "XGBoost":
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=0)
        model.fit(X_full.reshape(X_full.shape[0], -1), y_full)
        pred_scaled = model.predict(last_window.reshape(1, -1)).reshape(-1, 1)
    elif best_model_name == "Dense NN":
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_full.shape[1] * X_full.shape[2],)),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_full.reshape(X_full.shape[0], -1), y_full, epochs=10, verbose=0)
        pred_scaled = model.predict(last_window.reshape(1, -1), verbose=0)
    elif best_model_name == "RSI":
        # For RSI, no ML prediction, just generate signal directly
        df["RSI"] = ta.rsi(df["adj_close"], length=14)
        latest_rsi = df["RSI"].iloc[-1]
        if latest_rsi < 30:
            signal = "BUY"
        elif latest_rsi > 70:
            signal = "SELL"
        else:
            signal = "HOLD"

        predicted_price = np.nan
        today_price = df["adj_close"].iloc[-1]
        pct_diff = np.nan

        # Save RSI signal if required
        if save_signals:
            signal_row = {
                "Date": pd.Timestamp.today().date(),
                "PredictedPrice": predicted_price,
                "TodayPrice": today_price,
                "PctDiff": pct_diff,
                "Signal": signal,
                "Ticker": ticker,
                "Strategy": best_model_name,
            }
            signals_df = pd.DataFrame([signal_row])
            signals_filename = f"data\\signals_{ticker}_{best_model_name.replace(' ', '_')}.csv"
            signals_df.to_csv(signals_filename, index=False)
            print(f"Signal for {ticker} saved to {signals_filename}")

        return signal, pct_diff

    else:
        raise ValueError(f"Unknown model: {best_model_name}")

    # Inverse transform to price
    predicted_price = scaler.inverse_transform(pred_scaled)[0, 0]
    today_price = df["adj_close"].iloc[-1]
    pct_diff = (predicted_price - today_price) / today_price * 100

    if pct_diff > threshold:
        signal = "BUY"
    elif pct_diff < -threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Save signal if required
    if save_signals:
        signal_row = {
            "Date": pd.Timestamp.today().date(),
            "PredictedPrice": predicted_price,
            "TodayPrice": today_price,
            "PctDiff": pct_diff,
            "Signal": signal,
            "Ticker": ticker,
            "Strategy": best_model_name,
        }
        signals_df = pd.DataFrame([signal_row])
        signals_filename = f"data\\signals_{ticker}_{best_model_name.replace(' ', '_')}.csv"
        signals_df.to_csv(signals_filename, index=False)
        print(f"Signal for {ticker} saved to {signals_filename}")

    return signal, pct_diff
    
def perform_analysis(ticker, csv_results_file="strategy_performance.csv", save_signals=False):
    # Load data once
    df = pd.read_csv(f"data\\{ticker}_data.csv", index_col=0, parse_dates=True)
    data = df[['adj_close']].values

    # Scale data for ML models
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences for supervised learning
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    window_size = 20
    X, y = create_sequences(scaled_data, window_size)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    split = int(0.8 * len(X))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    test_dates = df.iloc[window_size + split:].index

    last_prices_scaled = np.array([x_seq[-1, 0] for x_seq in X_test])
    last_prices = scaler.inverse_transform(last_prices_scaled.reshape(-1, 1)).flatten()

    def backtest_strategy(preds, last_prices, dates, threshold=0.5):
        pct_diff = (preds.flatten() - last_prices) / last_prices * 100
        signals = []
        cash, position = 10_000, 0
        portfolio_values = []

        for diff, price in zip(pct_diff, last_prices):
            if diff > threshold and position == 0:
                position = cash // price
                cash -= position * price
                signals.append("BUY")
            elif diff < -threshold and position > 0:
                cash += position * price
                position = 0
                signals.append("SELL")
            else:
                signals.append("HOLD")
            portfolio_values.append(cash + position * price)

        return pd.DataFrame({"Date": dates, "Signal": signals, "PortfolioValue": portfolio_values})

    results = {}

    # LSTM
    lstm_model = models.Sequential([
        layers.LSTM(50, input_shape=(window_size, 1)),
        layers.Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    lstm_preds = lstm_model.predict(X_test, verbose=0)
    lstm_preds = scaler.inverse_transform(lstm_preds)
    results["LSTM"] = backtest_strategy(lstm_preds, last_prices, test_dates)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5)
    rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    rf_preds = rf_model.predict(X_test.reshape(X_test.shape[0], -1))
    rf_preds = scaler.inverse_transform(rf_preds.reshape(-1, 1))
    results["Random Forest"] = backtest_strategy(rf_preds, last_prices, test_dates)

    # XGBoost
    xgb_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=0)
    xgb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    xgb_preds = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))
    xgb_preds = scaler.inverse_transform(xgb_preds.reshape(-1, 1))
    results["XGBoost"] = backtest_strategy(xgb_preds, last_prices, test_dates)

    # Dense NN
    dense_model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1] * X_train.shape[2],)),
        layers.Dense(1)
    ])
    dense_model.compile(optimizer='adam', loss='mse')
    dense_model.fit(X_train.reshape(X_train.shape[0], -1), y_train, epochs=10, verbose=0)
    dense_preds = dense_model.predict(X_test.reshape(X_test.shape[0], -1), verbose=0)
    dense_preds = scaler.inverse_transform(dense_preds)
    results["Dense NN"] = backtest_strategy(dense_preds, last_prices, test_dates)

    # RSI Rule-Based
    df["RSI"] = ta.rsi(df["adj_close"], length=14)
    df["Signal"] = 0
    df.loc[df["RSI"] < 30, "Signal"] = 1
    df.loc[df["RSI"] > 70, "Signal"] = -1
    signals_rsi = df.iloc[window_size + split:]["Signal"].values

    cash, position = 10_000, 0
    portfolio_values = []
    signals = []

    for signal, price in zip(signals_rsi, last_prices):
        if signal == 1 and position == 0:
            position = cash // price
            cash -= position * price
            signals.append("BUY")
        elif signal == -1 and position > 0:
            cash += position * price
            position = 0
            signals.append("SELL")
        else:
            signals.append("HOLD")
        portfolio_values.append(cash + position * price)

    results["RSI"] = pd.DataFrame({"Date": test_dates, "Signal": signals, "PortfolioValue": portfolio_values})

    # Buy & Hold Benchmark
    start_price = last_prices[0]
    bh_curve = last_prices * (10_000 / start_price)
    results["Buy & Hold"] = pd.DataFrame({"Date": test_dates, "PortfolioValue": bh_curve})

    # Evaluate performances and pick best model by final portfolio value
    bh_final = results["Buy & Hold"]["PortfolioValue"].iloc[-1]

    summary_rows = []
    best_model_name = None
    best_final_val = -np.inf

    for name, res in results.items():
        final_val = res["PortfolioValue"].iloc[-1]
        ret_pct = (final_val / 10_000 - 1) * 100
        outperformed = "Yes" if final_val > bh_final else "No"

        summary_rows.append({
            "Ticker": ticker,
            "Strategy": name,
            "Final_Portfolio_Value": round(final_val, 2),
            "Total_Return_%": round(ret_pct, 2),
            "Outperformed_BuyHold": outperformed if name != "Buy & Hold" else "-"
        })

        if name != "Buy & Hold" and final_val > best_final_val:
            best_final_val = final_val
            best_model_name = name

    summary_df = pd.DataFrame(summary_rows)

    # Save summary CSV once
    if os.path.exists(csv_results_file):
        summary_df.to_csv(csv_results_file, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(csv_results_file, index=False)

    print(summary_df)
    print(results[best_model_name])

    # Optionally save signals for best model only
    if save_signals and best_model_name != "Buy & Hold":
        signals_df = results[best_model_name]
        signals_df["Ticker"] = ticker
        signals_df["Strategy"] = best_model_name
        signals_filename = f"data\\signals_{ticker}_{best_model_name.replace(' ', '_')}.csv"
        signals_df.to_csv(signals_filename, index=False)
        print(f"Signals for best model '{best_model_name}' saved to {signals_filename}")

    # Return best model name, its signals dataframe, and summary dataframe
    return best_model_name, results[best_model_name], summary_df

