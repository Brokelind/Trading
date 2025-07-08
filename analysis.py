import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class TradingModelSystem:
    def __init__(self):
        self.model_dir = "saved_models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def check_data_quality(self, df, ticker):
        """Validate data before processing"""
        if df.isnull().values.any():
            print(f"Warning: {ticker} contains missing values")
        if len(df) < 100:
            print(f"Warning: {ticker} has only {len(df)} data points")
        if 'adj_close' not in df.columns:
            if 'close' in df.columns:
                df['adj_close'] = df['close']
            else:
                raise ValueError(f"{ticker} missing price column")

    def prepare_data(self, ticker):
        """Load and preprocess data with technical indicators"""
        try:
            df = pd.read_csv(f"{self.data_dir}/{ticker}_data.csv", index_col=0, parse_dates=True)
            self.check_data_quality(df, ticker)
            
            # Basic indicators
            df['RSI'] = ta.rsi(df['adj_close'], length=14)
            df['SMA_20'] = ta.sma(df['adj_close'], length=20)
            df['EMA_50'] = ta.ema(df['adj_close'], length=50)
            df['MACD'] = ta.macd(df['adj_close'])['MACD_12_26_9']
            
            # Volatility indicators
            if all(col in df.columns for col in ['high', 'low', 'adj_close']):
                df['ATR'] = ta.atr(df['high'], df['low'], df['adj_close'], length=14)
            else:
                df['Volatility'] = df['adj_close'].pct_change().rolling(14).std() * 100
            
            # Target: Next day's return
            df['target'] = df['adj_close'].pct_change().shift(-1)
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Data preparation failed for {ticker}: {str(e)}")
            return None

    def create_sequences(self, data, window_size):
        """Create input sequences for time series models"""
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def train_model(self, model_type, X_train, y_train, ticker):
        """Train and save a single model"""
        model_path = f"{self.model_dir}/{ticker}_{model_type.replace(' ', '_')}"
        
        try:
            if model_type == "LSTM":
                if len(X_train.shape) != 3:
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    
                model = Sequential([
                    Input(shape=(X_train.shape[1], X_train.shape[2])),
                    LSTM(64, return_sequences=False),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
                save_model(model, f"{model_path}.keras")
                
            elif model_type == "Dense NN":
                X_train = X_train.reshape(X_train.shape[0], -1)
                model = Sequential([
                    Input(shape=(X_train.shape[1],)),
                    Dense(128, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=15, verbose=0)
                save_model(model, f"{model_path}.keras")
                
            elif model_type == "Random Forest":
                X_train = X_train.reshape(X_train.shape[0], -1)
                model = RandomForestRegressor(n_estimators=150, max_depth=7, n_jobs=-1)
                model.fit(X_train, y_train)
                joblib.dump(model, f"{model_path}.joblib")
                
            elif model_type == "XGBoost":
                X_train = X_train.reshape(X_train.shape[0], -1)
                model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, n_jobs=-1)
                model.fit(X_train, y_train)
                joblib.dump(model, f"{model_path}.joblib")
                
            return model
            
        except Exception as e:
            print(f"Error training {model_type} for {ticker}: {str(e)}")
            return None

    def load_model(self, model_type, ticker):
        """Load a pre-trained model"""
        model_path = f"{self.model_dir}/{ticker}_{model_type.replace(' ', '_')}"
        
        try:
            if model_type in ["LSTM", "Dense NN"]:
                return load_model(f"{model_path}.keras", compile=False)
            else:
                return joblib.load(f"{model_path}.joblib")
        except Exception as e:
            print(f"Error loading {model_type} for {ticker}: {str(e)}")
            raise

    def walk_forward_backtest(self, df, model_type, ticker, window_size=20, threshold=1.0):
        """Robust walk-forward backtesting with full error handling"""
        try:
            # Validate data length
            if len(df) < window_size + 2:
                raise ValueError(f"Insufficient data ({len(df)} points) for window size {window_size}")
            
            # Prepare data
            data = df[['adj_close']].values
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Create windows with valid indices
            windows = []
            valid_indices = []
            for i in range(window_size, len(scaled_data)-1):
                if i >= window_size:
                    windows.append(scaled_data[i-window_size:i])
                    valid_indices.append(i)
            
            if not windows:
                return pd.DataFrame(columns=["Date", "TruePrice", "PredictedPrice", "Signal", "PortfolioValue"])
            
            windows = np.array(windows)
            
            # Load or train model
            try:
                model = self.load_model(model_type, ticker)
            except:
                X_train, y_train = self.create_sequences(scaled_data[:-window_size], window_size)
                if model_type in ["LSTM", "Dense NN"]:
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                model = self.train_model(model_type, X_train, y_train, ticker)
                if model is None:
                    raise ValueError(f"Failed to train {model_type} model")
            
            # Make predictions
            if model_type == "LSTM":
                preds_scaled = model.predict(windows, verbose=0)
            elif model_type == "Dense NN":
                preds_scaled = model.predict(windows.reshape(len(windows), -1), verbose=0)
            else:
                preds_scaled = model.predict(windows.reshape(len(windows), -1))
            
            # Validate predictions
            if len(preds_scaled) == 0:
                raise ValueError("Empty predictions array")
                
            preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            true_prices = df['adj_close'].iloc[valid_indices].values
            
            if len(preds) != len(true_prices):
                raise ValueError(f"Prediction length mismatch: {len(preds)} vs {len(true_prices)}")
            
            # Generate signals
            pct_diffs = (preds - true_prices) / true_prices * 100
            signals = np.where(pct_diffs > threshold, "BUY",
                             np.where(pct_diffs < -threshold, "SELL", "HOLD"))
            
            # Simulate portfolio
            positions = np.zeros(len(true_prices))
            cash = 10_000
            portfolio_values = []
            
            for i in range(len(true_prices)):
                if signals[i] == "BUY" and (i == 0 or positions[i-1] == 0):
                    positions[i:] = cash // true_prices[i]
                    cash -= positions[i] * true_prices[i]
                elif signals[i] == "SELL" and i > 0 and positions[i-1] > 0:
                    cash += positions[i-1] * true_prices[i]
                    positions[i:] = 0
                
                portfolio_values.append(cash + positions[i] * true_prices[i])
            
            return pd.DataFrame({
                "Date": df.index[valid_indices],
                "TruePrice": true_prices,
                "PredictedPrice": preds,
                "Signal": signals,
                "PortfolioValue": portfolio_values
            })
            
        except Exception as e:
            print(f"Backtest failed for {ticker} ({model_type}): {str(e)}")
            return pd.DataFrame(columns=["Date", "TruePrice", "PredictedPrice", "Signal", "PortfolioValue"])

    def rsi_backtest(self, df, ticker):
        """RSI strategy backtest with error handling"""
        try:
            df["RSI"] = ta.rsi(df['adj_close'], length=14)
            signals = np.where(df["RSI"] < 30, 1, np.where(df["RSI"] > 70, -1, 0))
            
            positions = np.zeros(len(df))
            cash = 10_000
            portfolio_values = []
            
            for i in range(len(df)):
                if signals[i] == 1 and (i == 0 or positions[i-1] == 0):
                    positions[i:] = cash // df['adj_close'].iloc[i]
                    cash -= positions[i] * df['adj_close'].iloc[i]
                elif signals[i] == -1 and i > 0 and positions[i-1] > 0:
                    cash += positions[i-1] * df['adj_close'].iloc[i]
                    positions[i:] = 0
                
                portfolio_values.append(cash + positions[i] * df['adj_close'].iloc[i])
            
            return pd.DataFrame({
                "Date": df.index,
                "Signal": np.where(signals == 1, "BUY", np.where(signals == -1, "SELL", "HOLD")),
                "PortfolioValue": portfolio_values
            })
            
        except Exception as e:
            print(f"RSI backtest failed for {ticker}: {str(e)}")
            return pd.DataFrame(columns=["Date", "Signal", "PortfolioValue"])

    def calculate_metrics(self, results):
        """Calculate comprehensive performance metrics including prediction accuracy"""
        metrics = {}
        
        for strategy, data in results.items():
            try:
                if len(data) == 0:
                    continue
                    
                # Basic performance metrics
                returns = data['PortfolioValue'].pct_change()
                cum_return = (data['PortfolioValue'].iloc[-1] / 10000 - 1) * 100
                
                rolling_max = data['PortfolioValue'].cummax()
                drawdown = (data['PortfolioValue'] - rolling_max) / rolling_max
                max_drawdown = drawdown.min() * 100
                
                volatility = returns.std() * np.sqrt(252) * 100
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                
                # Trade metrics
                if 'Signal' in data.columns:
                    trades = data[data['Signal'].isin(['BUY', 'SELL'])]
                    trade_returns = trades['PortfolioValue'].pct_change()
                    win_rate = (trade_returns > 0).mean() * 100 if len(trade_returns) > 0 else np.nan
                    profit_factor = -trade_returns[trade_returns > 0].sum() / trade_returns[trade_returns < 0].sum() if len(trade_returns[trade_returns < 0]) > 0 else np.inf
                else:
                    win_rate = np.nan
                    profit_factor = np.nan
                
                # Prediction accuracy metrics (if available)
                pred_accuracy = {}
                if all(col in data.columns for col in ['TruePrice', 'PredictedPrice']):
                    errors = data['PredictedPrice'] - data['TruePrice']
                    pred_accuracy = {
                        'MAE (%)': (errors.abs() / data['TruePrice']).mean() * 100,
                        'RMSE (%)': np.sqrt((errors**2).mean()) / data['TruePrice'].mean() * 100,
                        'Direction Accuracy (%)': (np.sign(data['PredictedPrice'].diff()) == np.sign(data['TruePrice'].diff())).mean() * 100,
                        'R-squared': max(0, 1 - (errors**2).sum() / ((data['TruePrice'] - data['TruePrice'].mean())**2).sum())
                    }
                
                metrics[strategy] = {
                    # Performance Metrics
                    'Return (%)': cum_return,
                    'Max Drawdown (%)': max_drawdown,
                    'Volatility (%)': volatility,
                    'Sharpe Ratio': sharpe_ratio,
                    'Win Rate (%)': win_rate,
                    'Profit Factor': profit_factor,
                    'Final Value ($)': data['PortfolioValue'].iloc[-1],
                    
                    # Prediction Accuracy Metrics
                    **pred_accuracy,
                    
                    # Activity Metrics
                    'Trade Count': len(trades) if 'Signal' in data.columns else 0,
                    'Hold Period (days)': len(data) / (len(trades)/2) if len(trades) > 0 else len(data)
                }
                
            except Exception as e:
                print(f"Error calculating metrics for {strategy}: {str(e)}")
                continue
        
        return pd.DataFrame(metrics).T

    def visualize_results(self, results, ticker):
        """Create interactive visualization with trade signals."""
        try:
            import os
            import pandas as pd
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Ensure output dir exists
            os.makedirs(self.data_dir, exist_ok=True)

            # Create figure with 3 subplots: Price+Signals, Portfolio, Drawdown
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.3, 0.2],
                subplot_titles=(
                    f"{ticker} Price with Trade Signals",
                    "Portfolio Value",
                    "Drawdown"
                )
            )
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            # 1. Price and Signals Chart
            price_data = results.get("Buy & Hold", pd.DataFrame())
            if not price_data.empty and 'Date' in price_data.columns:
                y_col = "TruePrice" if "TruePrice" in price_data.columns else "PortfolioValue"
                fig.add_trace(
                    go.Scatter(
                        x=price_data['Date'],
                        y=price_data[y_col],
                        name="Price",
                        line=dict(color='#333333', width=2),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
            
            # Add buy/sell signals for each strategy
            for i, (strategy, data) in enumerate(results.items()):
                if strategy == "Buy & Hold" or data.empty:
                    continue

                if 'Signal' in data.columns and 'TruePrice' in data.columns:
                    # Buy signals
                    buy_signals = data[data['Signal'] == 'BUY']
                    if not buy_signals.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=buy_signals['Date'],
                                y=buy_signals['TruePrice'],
                                name=f"{strategy} Buy",
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-up',
                                    size=10,
                                    color='green',
                                    line=dict(width=1, color='DarkSlateGrey')
                                )
                            ),
                            row=1, col=1
                        )
                    
                    # Sell signals
                    sell_signals = data[data['Signal'] == 'SELL']
                    if not sell_signals.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=sell_signals['Date'],
                                y=sell_signals['TruePrice'],
                                name=f"{strategy} Sell",
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-down',
                                    size=10,
                                    color='red',
                                    line=dict(width=1, color='DarkSlateGrey')
                                )
                            ),
                            row=1, col=1
                        )
            
            # 2. Portfolio Value Chart
            for i, (strategy, data) in enumerate(results.items()):
                if data.empty:
                    continue
                if 'Date' in data.columns and 'PortfolioValue' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data['Date'],
                            y=data['PortfolioValue'],
                            name=strategy,
                            line=dict(color=colors[i % len(colors)], width=2),
                            opacity=0.8
                        ),
                        row=2, col=1
                    )
            
            # 3. Drawdown Chart
            for strategy, data in results.items():
                if strategy == "Buy & Hold" or data.empty:
                    continue
                if 'Date' in data.columns and 'PortfolioValue' in data.columns:
                    rolling_max = data['PortfolioValue'].cummax()
                    drawdown = (data['PortfolioValue'] - rolling_max) / rolling_max * 100
                    fig.add_trace(
                        go.Scatter(
                            x=data['Date'],
                            y=drawdown,
                            name=f"{strategy} Drawdown",
                            line=dict(width=1),
                            opacity=0.6,
                            showlegend=False
                        ),
                        row=3, col=1
                    )
            
            # Layout tweaks
            fig.update_layout(
                title=f"{ticker} Trading Strategy Analysis",
                height=1000,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
            
            # Spikes
            fig.update_xaxes(showspikes=True, spikecolor="grey", spikesnap="cursor", spikemode="across")
            fig.update_yaxes(showspikes=True, spikecolor="grey", spikethickness=1)
            fig.update_layout(spikedistance=1000, hoverdistance=100)
            
            html_file = f"{self.data_dir}/{ticker}_performance.html"
            fig.write_html(html_file)
            print(f"Saved visualization to {html_file}")
            
        except Exception as e:
            print(f"Visualization failed for {ticker}: {str(e)}")


    def predict_tomorrow(self, ticker, window_size=20):
        """Predict next day's price with error handling"""
        try:
            df = self.prepare_data(ticker)
            if df is None:
                return {}
                
            data = df[['adj_close']].values
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            last_window = scaled_data[-window_size:].reshape(1, window_size, 1)
            
            predictions = {}
            model_types = ["LSTM", "Random Forest", "XGBoost", "Dense NN", "RSI"]
            
            for model_type in model_types:
                try:
                    if model_type == "RSI":
                        rsi = df['RSI'].iloc[-1]
                        signal = "BUY" if rsi < 30 else "SELL" if rsi > 70 else "HOLD"
                        predictions["RSI"] = {"signal": signal, "prediction": None}
                        continue
                    
                    model = self.load_model(model_type, ticker)
                    
                    if model_type == "LSTM":
                        pred_scaled = model.predict(last_window, verbose=0)[0][0]
                    elif model_type == "Dense NN":
                        pred_scaled = model.predict(last_window.reshape(1, -1), verbose=0)[0][0]
                    else:
                        pred_scaled = model.predict(last_window.reshape(1, -1))[0]
                    
                    prediction = scaler.inverse_transform([[pred_scaled]])[0][0]
                    today_price = df['adj_close'].iloc[-1]
                    pct_diff = (prediction - today_price) / today_price * 100
                    
                    signal = "BUY" if pct_diff > 1.0 else "SELL" if pct_diff < -1.0 else "HOLD"
                    predictions[model_type] = {
                        "prediction": prediction,
                        "signal": signal,
                        "pct_diff": pct_diff
                    }
                    
                except Exception as e:
                    print(f"Prediction failed for {model_type}: {str(e)}")
                    predictions[model_type] = None
            
            return predictions
            
        except Exception as e:
            print(f"Tomorrow prediction failed for {ticker}: {str(e)}")
            return {}

    def run_analysis(self, ticker):
        """Complete analysis pipeline with full error handling"""
        try:
            print(f"\nAnalyzing {ticker}...")
            df = self.prepare_data(ticker)
            if df is None:
                return None, None, None
                
            # Train models in parallel
            model_types = ["LSTM", "Random Forest", "XGBoost", "Dense NN"]
            scaled_data = MinMaxScaler().fit_transform(df[['adj_close']].values)
            X_train, y_train = self.create_sequences(scaled_data[:-20], 20)
            
            with ThreadPoolExecutor() as executor:
                futures = []
                for model_type in model_types:
                    if model_type in ["LSTM", "Dense NN"]:
                        X = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) if model_type == "LSTM" else X_train.reshape(X_train.shape[0], -1)
                    else:
                        X = X_train.reshape(X_train.shape[0], -1)
                    futures.append(executor.submit(self.train_model, model_type, X, y_train, ticker))
                
                for future in tqdm(futures, desc="Training models"):
                    future.result()
            
            # Run backtests
            results = {}
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self.walk_forward_backtest, df.copy(), model, ticker): model 
                    for model in model_types
                }
                futures[executor.submit(self.rsi_backtest, df.copy(), ticker)] = "RSI"
                
                for future in tqdm(futures, desc="Running backtests"):
                    results[futures[future]] = future.result()
            
            # Add Buy & Hold baseline
            results["Buy & Hold"] = pd.DataFrame({
                "Date": df.index,
                "PortfolioValue": (df['adj_close'] / df['adj_close'].iloc[0]) * 10000
            })
            
            # Calculate metrics
            metrics = self.calculate_metrics(results)
            if not metrics.empty:
                metrics.to_csv(f"{self.data_dir}/{ticker}_metrics.csv")
                print(f"\nPerformance Metrics:\n{metrics.round(2)}")
            
            # Visualize results
            self.visualize_results(results, ticker)
            
            # Generate tomorrow's predictions
            tomorrow_pred = self.predict_tomorrow(ticker)
            print("\nTomorrow's Predictions:")
            for model, pred in tomorrow_pred.items():
                print(f"{model}: {pred}")
            
            return results, metrics, tomorrow_pred
            
        except Exception as e:
            print(f"Analysis failed for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None