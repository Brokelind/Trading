import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import logging
from typing import Dict, Tuple, Optional, Union

class TradingModelSystem:
    def __init__(self, config: Optional[dict] = None):
        # Configuration with defaults matching main.py expectations
        self.config = {
            'data_dir': 'data',
            'model_dir': 'saved_models',
            'window_size': 20,
            'prediction_threshold': 0.5,
            'initial_capital': 10000,
            'rsi_buy': 30,
            'rsi_sell': 70
        }
        if config:
            self.config.update(config)
            
        os.makedirs(self.config['data_dir'], exist_ok=True)
        os.makedirs(self.config['model_dir'], exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_data_quality(self, df: pd.DataFrame, ticker: str) -> bool:
        """Validate data before processing (matches main.py requirements)"""
        if df.isnull().values.any():
            self.logger.warning(f"{ticker} contains missing values")
        if len(df) < 100:
            self.logger.warning(f"{ticker} has only {len(df)} data points")
            return False
        if 'adj_close' not in df.columns:
            if 'close' in df.columns:
                df['adj_close'] = df['close']
            else:
                self.logger.error(f"{ticker} missing price column")
                return False
        return True

    def prepare_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Enhanced feature engineering compatible with main.py"""
        try:
            df = pd.read_csv(f"{self.config['data_dir']}/{ticker}_data.csv", 
                           index_col=0, parse_dates=True)
            if not self.check_data_quality(df, ticker):
                return None
                
            # Price Transformations
            df['log_ret'] = np.log(df['adj_close']/df['adj_close'].shift(1))
            df['volatility'] = df['log_ret'].rolling(21).std() * np.sqrt(252)
            
            # Technical Indicators (matching main.py expected features)
            df['RSI_14'] = ta.rsi(df['adj_close'], length=14)
            df['RSI_21'] = ta.rsi(df['adj_close'], length=21)
            macd = ta.macd(df['adj_close'])
            df['MACD'] = macd['MACD_12_26_9']

            #print(df.head())    
            adx = ta.adx(df['high'], df['low'], df['adj_close'])
            df['ADX_14'] = adx['ADX_14']
            
            # Volume Features
            if 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(10).mean()
                df['volume_z'] = (df['volume'] - df['volume'].rolling(21).mean()) / df['volume'].rolling(21).std()
            
            # Market Regime Features
            df['trend_strength'] = df['ADX_14']
            df['volatility_regime'] = pd.qcut(df['volatility'], q=4, labels=False)
            
            # Target Engineering (compatible with main.py prediction format)
            df['target_direction'] = np.where(df['adj_close'].shift(-1) > df['adj_close'], 1, 0)
            df['target'] = df['adj_close'].pct_change().shift(-1)
            
            return df.dropna()
            
        except Exception as e:
            self.logger.error(f"Data prep failed for {ticker}: {str(e)}")

            return None

    def train_model(self, model_type: str, X_train: np.ndarray, 
                   y_train: np.ndarray, ticker: str):
        """Model training compatible with main.py's expected model types"""
        model_path = f"{self.config['model_dir']}/{ticker}_{model_type.replace(' ', '_')}"
        
        try:
            if model_type == "LSTM":
                model = Sequential([
                    Input(shape=(X_train.shape[1], X_train.shape[2])),
                    LSTM(128, return_sequences=True, dropout=0.2),
                    LSTM(64, dropout=0.2),
                    Dense(32, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                early_stop = EarlyStopping(monitor='val_loss', patience=5)
                model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0
                )
                save_model(model, f"{model_path}.keras")
                
            elif model_type == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    n_jobs=-1,
                    random_state=42
                )
                model.fit(X_train, y_train)
                joblib.dump(model, f"{model_path}.joblib")
                
            elif model_type == "XGBoost":
                model = XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                    random_state=42
                )
                model.fit(X_train, y_train)
                joblib.dump(model, f"{model_path}.joblib")
                
            elif model_type == "Dense NN":
                model = Sequential([
                    Input(shape=(X_train.shape[1],)),
                    Dense(256, activation='relu'),
                    Dropout(0.3),
                    Dense(128, activation='relu'),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                save_model(model, f"{model_path}.keras")
                
            return model
            
        except Exception as e:
            self.logger.error(f"Training failed for {model_type}: {str(e)}")
            return None

    def walk_forward_backtest(self, df: pd.DataFrame, model_type: str, 
                            ticker: str) -> pd.DataFrame:
        """Backtesting that returns DataFrame matching main.py expectations"""
        try:
            window_size = self.config['window_size']
            threshold = self.config['prediction_threshold']
            
            if len(df) < window_size + 2:
                self.logger.warning(f"Insufficient data for {ticker}")
                return pd.DataFrame()
                
            data = df[['adj_close']].values
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            X, y = [], []
            for i in range(window_size, len(data)):
                X.append(scaled_data[i-window_size:i, 0])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)
            
            # Reshape for different model types
            if model_type == "LSTM":
                X = X.reshape(X.shape[0], X.shape[1], 1)
            elif model_type == "Dense NN":
                X = X.reshape(X.shape[0], -1)
            else:
                X = X.reshape(X.shape[0], -1)
            
            # Load or train model
            try:
                model = self.load_model(model_type, ticker)
            except:
                model = self.train_model(model_type, X[:-window_size], y[:-window_size], ticker)
                if model is None:
                    return pd.DataFrame()
            
            # Predictions - remove verbose for non-Keras models
            if model_type in ["LSTM", "Dense NN"]:
                preds_scaled = model.predict(X, verbose=0)
            else:
                preds_scaled = model.predict(X)  # Remove verbose parameter
            
            preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            true_prices = df['adj_close'].iloc[window_size:].values
            
            # Generate signals
            pct_diffs = (preds - true_prices) / true_prices * 100
            signals = np.where(pct_diffs > threshold, "BUY",
                              np.where(pct_diffs < -threshold, "SELL", "HOLD"))
            
            # Portfolio simulation
            positions = np.zeros(len(true_prices))
            cash = self.config['initial_capital']
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
                "Date": df.index[window_size:],
                "TruePrice": true_prices,
                "PredictedPrice": preds,
                "Signal": signals,
                "PortfolioValue": portfolio_values
            })
            
        except Exception as e:
            self.logger.error(f"Backtest failed for {ticker}: {str(e)}")
            return pd.DataFrame()

    def rsi_backtest(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """RSI backtest matching main.py expected output format"""
        try:
            df["RSI"] = ta.rsi(df['adj_close'], length=14)
            signals = np.where(df["RSI"] < self.config['rsi_buy'], 1, 
                             np.where(df["RSI"] > self.config['rsi_sell'], -1, 0))
            
            positions = np.zeros(len(df))
            cash = self.config['initial_capital']
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
            self.logger.error(f"RSI backtest failed: {str(e)}")
            return pd.DataFrame()

    def calculate_metrics(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Metrics calculation matching main.py expected format"""
        metrics = {}
        
        for strategy, data in results.items():
            if len(data) == 0:
                continue
                
            returns = data['PortfolioValue'].pct_change()
            cum_return = (data['PortfolioValue'].iloc[-1] / self.config['initial_capital'] - 1) * 100
            
            rolling_max = data['PortfolioValue'].cummax()
            drawdown = (data['PortfolioValue'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            volatility = returns.std() * np.sqrt(252) * 100
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Prediction accuracy if available
            pred_metrics = {}
            if all(col in data.columns for col in ['TruePrice', 'PredictedPrice']):
                errors = data['PredictedPrice'] - data['TruePrice']
                pred_metrics = {
                    'Direction Accuracy (%)': (np.sign(data['PredictedPrice'].diff()) == 
                                             np.sign(data['TruePrice'].diff())).mean() * 100,
                    'MAE (%)': (errors.abs() / data['TruePrice']).mean() * 100
                }
            
            metrics[strategy] = {
                'Return (%)': cum_return,
                'Max Drawdown (%)': max_drawdown,
                'Volatility (%)': volatility,
                'Sharpe Ratio': sharpe_ratio,
                **pred_metrics
            }
        
        return pd.DataFrame(metrics).T

    def predict_tomorrow(self, ticker: str) -> Dict[str, Dict[str, Union[float, str]]]:
        """Fixed prediction without verbose for non-Keras models"""
        try:
            df = self.prepare_data(ticker)
            if df is None:
                return {}
                
            window_size = self.config['window_size']
            data = df[['adj_close']].values[-window_size:]
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            predictions = {}
            for model_type in ["LSTM", "Random Forest", "XGBoost", "Dense NN"]:
                try:
                    model = self.load_model(model_type, ticker)
                    if model is None:
                        continue
                        
                    if model_type == "LSTM":
                        x = scaled_data.reshape(1, window_size, 1)
                        pred_scaled = model.predict(x, verbose=0)
                    elif model_type == "Dense NN":
                        x = scaled_data.reshape(1, -1)
                        pred_scaled = model.predict(x, verbose=0)
                    else:
                        x = scaled_data.reshape(1, -1)
                        pred_scaled = model.predict(x)  # No verbose parameter
                    
                    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                    last_price = df['adj_close'].iloc[-1]
                    pct_diff = (pred - last_price) / last_price * 100
                    
                    predictions[model_type] = {
                        'predicted_price': pred,
                        'pct_diff': pct_diff,
                        'signal': "BUY" if pct_diff > self.config['prediction_threshold'] else 
                                "SELL" if pct_diff < -self.config['prediction_threshold'] else "HOLD"
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Prediction failed for {model_type}: {str(e)}")
                    continue
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return {}
    def create_sequences(self, data, window_size):
        """Create input sequences for time series models"""
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    def load_model(self, model_type: str, ticker: str):
        """Model loading matching main.py expected behavior"""
        model_path = f"{self.config['model_dir']}/{ticker}_{model_type.replace(' ', '_')}"
        try:
            if model_type in ["LSTM", "Dense NN"]:
                return load_model(f"{model_path}.keras", compile=False)
            else:
                return joblib.load(f"{model_path}.joblib")
        except Exception as e:
            self.logger.warning(f"Failed to load {model_type}: {str(e)}")
            return None

    def run_analysis(self, ticker: str) -> Tuple[Dict[str, pd.DataFrame], 
                                               Optional[pd.DataFrame], 
                                               Dict[str, Dict[str, Union[float, str]]]]:
        """Main analysis method matching main.py expected return signature"""
        try:
            df = self.prepare_data(ticker)
            if df is None:
                return {}, None, {}
                
            # Train models
            model_types = ["LSTM", "Random Forest", "XGBoost", "Dense NN"]
            scaled_data = MinMaxScaler().fit_transform(df[['adj_close']].values)
            X_train, y_train = self.create_sequences(scaled_data[:-20], 20)
            
            with ThreadPoolExecutor() as executor:
                futures = []
                for model_type in model_types:
                    if model_type == "LSTM":
                        X = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    elif model_type == "Dense NN":
                        X = X_train.reshape(X_train.shape[0], -1)
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
                "PortfolioValue": (df['adj_close'] / df['adj_close'].iloc[0]) * self.config['initial_capital']
            })
            
            # Calculate metrics
            metrics = self.calculate_metrics(results)
            
            # Visualize results with metrics
            self.visualize_results(results, ticker, metrics)
            
            # Generate predictions
            tomorrow_pred = self.predict_tomorrow(ticker)
            
            return results, metrics, tomorrow_pred
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {}, None, {}

    def visualize_results(self, results: Dict[str, pd.DataFrame], ticker: str, metrics: pd.DataFrame = None):
        """Enhanced visualization with metrics and improved layout"""
        try:
            # Create figure with 3 subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.3, 0.2],
                subplot_titles=(
                    f"{ticker} Price and Signals",
                    "Portfolio Value Comparison",
                    "Drawdown Analysis"
                ),
                specs=[[{"secondary_y": True}], [{}], [{}]]
            )

            # Colors for different strategies
            colors = {
                'Buy & Hold': '#1f77b4',
                'LSTM': '#ff7f0e',
                'Random Forest': '#2ca02c',
                'XGBoost': '#d62728',
                'Dense NN': '#9467bd',
                'RSI': '#8c564b'
            }

            # 1. Price and Signals (Top Panel)
            if "TruePrice" in results.get(list(results.keys())[0], pd.DataFrame()).columns:
                base_df = next(df for df in results.values() if 'TruePrice' in df.columns)
                
                # Price line
                fig.add_trace(
                    go.Scatter(
                        x=base_df['Date'],
                        y=base_df['TruePrice'],
                        name="Price",
                        line=dict(color='#333333', width=2),
                        opacity=0.8
                    ),
                    row=1, col=1,
                    secondary_y=False
                )
                
                # Add signals for all strategies
                for strategy, data in results.items():
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
                                        color=colors.get(strategy, '#17BECF'),
                                        line=dict(width=1, color='DarkSlateGrey')
                                    ),
                                    showlegend=False
                                ),
                                row=1, col=1,
                                secondary_y=False
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
                                        color=colors.get(strategy, '#17BECF'),
                                        line=dict(width=1, color='DarkSlateGrey')
                                    ),
                                    showlegend=False
                                ),
                                row=1, col=1,
                                secondary_y=False
                            )

            # 2. Portfolio Values (Middle Panel)
            for strategy, data in results.items():
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data['PortfolioValue'],
                        name=strategy,
                        line=dict(color=colors.get(strategy, '#7F7F7F')),
                        opacity=0.8
                    ),
                    row=2, col=1
                )

            # 3. Drawdown Analysis (Bottom Panel)
            for strategy, data in results.items():
                rolling_max = data['PortfolioValue'].cummax()
                drawdown = (data['PortfolioValue'] - rolling_max) / rolling_max
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=drawdown,
                        name=f"{strategy} Drawdown",
                        line=dict(color=colors.get(strategy, '#7F7F7F'), width=1),
                        fill='tozeroy',
                        opacity=0.6,
                        showlegend=False
                    ),
                    row=3, col=1
                )

            # Add metrics annotation if available
            if metrics is not None and not metrics.empty:
                metrics_text = []
                for strategy in results.keys():
                    if strategy in metrics.index:
                        strat_metrics = metrics.loc[strategy]
                        text = (f"<b>{strategy}</b><br>"
                                f"Return: {strat_metrics.get('Return (%)', 'N/A'):.1f}%<br>"
                                f"Sharpe: {strat_metrics.get('Sharpe Ratio', 'N/A'):.2f}<br>"
                                f"Drawdown: {strat_metrics.get('Max Drawdown (%)', 'N/A'):.1f}%<br>"
                                f"accuracy: {strat_metrics.get('Direction Accuracy (%)', 'N/A'):.1f}%<br>"
                                f"MAE: {strat_metrics.get('MAE (%)', 'N/A'):.1f}%"
                                )

                        metrics_text.append(text)
                
                if metrics_text:
                    fig.add_annotation(
                        x=0.02,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text="<br>".join(metrics_text),
                        showarrow=False,
                        align="left",
                        bordercolor="#c7c7c7",
                        borderwidth=1,
                        borderpad=4,
                        bgcolor="#ffffff",
                        opacity=0.8
                    )

            # Update layout
            fig.update_layout(
                title=f"{ticker} Trading Strategy Analysis - {pd.Timestamp.now().strftime('%Y-%m-%d')}",
                height=1000,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                plot_bgcolor='rgba(240,240,240,0.8)'
            )

            # Update y-axes titles
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)

            # Add range slider
            fig.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                row=3, col=1
            )

            # Save to HTML
            html_file = f"{self.config['data_dir']}/{ticker}_performance.html"
            fig.write_html(html_file)
            self.logger.info(f"Saved enhanced visualization to {html_file}")
            
        except Exception as e:
            self.logger.error(f"Enhanced visualization failed: {str(e)}")
