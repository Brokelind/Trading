import os
import json
import time
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Any, Union
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib
import talib
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import (LSTM, Dense, Input, Dropout, 
                                    BatchNormalization, Attention)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError

# ---- CONFIG ----
MODEL_DIR_DEFAULT = "saved_models"
DATA_DIR_DEFAULT = "data"
MODEL_META_FILENAME = "model_meta.json"
RETRAIN_DAYS = 7  # autoretrain interval
ENSEMBLE_METHODS = ['mean', 'median', 'weighted']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TradingModelSystem")


class TradingModelSystem:
    """
    Enhanced trading model system with:
    - Multiple improved model architectures
    - Probabilistic forecasting
    - Advanced feature engineering
    - Ensemble methods
    - Comprehensive risk metrics
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = {
            "data_dir": DATA_DIR_DEFAULT,
            "model_dir": MODEL_DIR_DEFAULT,
            "window_size": 30,  
            "prediction_threshold_pct": 0.25,
            "initial_capital": 10000,
            "min_data_points": 200,  # Increased minimum data requirement
            "retrain_days": RETRAIN_DAYS,
            "verbose": False,
            "enable_uncertainty": True,  # Enable prediction intervals
            "n_cv_folds": 5,  # For time series cross-validation
            "feature_lookback": 10,  # How many past steps to use for feature creation
            "enable_lightgbm": False
        }
        if config:
            cfg.update(config)
        self.config = cfg
        self.scaler = MinMaxScaler()
        self.feature_list = None

        os.makedirs(self.config["data_dir"], exist_ok=True)
        os.makedirs(self.config["model_dir"], exist_ok=True)


    # ---------- Data / features ----------
    def load_raw(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load raw data with additional checks and preprocessing"""
        path = os.path.join(self.config["data_dir"], f"{ticker}_data.csv")
        if not os.path.exists(path):
            logger.warning(f"No data file for {ticker} at {path}")
            return None
        
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if df.empty:
                logger.warning(f"Empty dataframe for {ticker}")
                return None
            
            # Handle missing or invalid columns
            if "adj_close" not in df.columns:
                if "close" in df.columns:
                    df["adj_close"] = df["close"]
                else:
                    logger.error(f"No price column found for {ticker}")
                    return None
            
            # Ensure datetime index is properly set
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    logger.error(f"Could not parse index as datetime: {e}")
                    return None
            
            return df
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None


    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Using TA-Lib instead of pandas-ta"""
        if df.empty:
            return df
        try:
            # Price-based indicators
            df['log_ret'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
            df['volatility'] = df['log_ret'].rolling(21).std() * np.sqrt(252)
            
            # RSI
            df['RSI_14'] = talib.RSI(df['adj_close'], timeperiod=14)
            df['RSI_7'] = talib.RSI(df['adj_close'], timeperiod=7)
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['adj_close'])
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['adj_close'])
            df['MACD'] = macd
            df['MACD_signal'] = macd_signal
            df['MACD_hist'] = macd_hist
            
            # ADX
            df['ADX'] = talib.ADX(df['high'], df['low'], df['adj_close'])
            
            # ATR
            df['ATR'] = talib.ATR(df['high'], df['low'], df['adj_close'])
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['adj_close'])
            df['BB_upper'] = upper
            df['BB_lower'] = lower
            df['BB_width'] = (upper - lower) / df['adj_close']
            
            # OBV
            if 'volume' in df.columns:
                df['OBV'] = talib.OBV(df['adj_close'], df['volume'])
                df['volume_ma10'] = df['volume'].rolling(10).mean()
                df['volume_ma20'] = df['volume'].rolling(20).mean()
            
            # Create lagged features
            lookback = self.config['feature_lookback']
            for lag in range(1, lookback + 1):
                df[f'ret_lag_{lag}'] = df['log_ret'].shift(lag)
                df[f'vol_lag_{lag}'] = df['volatility'].shift(lag)
            
            # Target variables
            df['target_price'] = df['adj_close'].shift(-1)
            df['target_return'] = df['target_price'] / df['adj_close'] - 1.0
            df['target_direction'] = np.where(df['target_return'] > 0, 1, 0)
            
            # Drop any remaining NA values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error in _add_technical_indicators: {e}")
            raise

    def prepare_features(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            print(f"DEBUG: Starting feature preparation for {ticker}")
            
            # Load raw data
            df = self.load_raw(ticker)
            print(f"DEBUG: Raw data loaded, shape: {df.shape if df is not None else 'None'}")
            
            if df is None or len(df) < self.config["min_data_points"]:
                logger.warning(f"{ticker}: Not enough raw data or failed to load")
                return None

            # Drop initial NA values
            df = df.dropna()
            print(f"DEBUG: After initial dropna, shape: {df.shape}")

            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Drop NA values created by technical indicators
            df = df.dropna()
            print(f"DEBUG: After final dropna, shape: {df.shape}")

            # Check if we still have enough data
            if len(df) < self.config["min_data_points"]:
                logger.warning(f"{ticker}: Only {len(df)} rows after feature engineering (min: {self.config['min_data_points']})")
                return None

            # Create target variables
            df['target_return'] = df['adj_close'].pct_change().shift(-1)
            print(f"DEBUG: After target creation, shape: {df.shape}")
            
            # Remove any rows with NaN targets
            df = df.dropna(subset=['target_return'])
            print(f"DEBUG: After target dropna, shape: {df.shape}")
            
            # Remove extreme outliers
            returns = df['target_return']
            median = returns.median()
            mad = (returns - median).abs().median()
            df = df[(returns >= median - 5*mad) & (returns <= median + 5*mad)]
            print(f"DEBUG: After outlier removal, shape: {df.shape}")
            
            # Final data check
            if len(df) < self.config["min_data_points"]:
                logger.warning(f"{ticker}: Only {len(df)} rows after outlier removal")
                return None

            print(f"DEBUG: Successfully prepared features for {ticker}, final shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error preparing features for {ticker}: {e}")
            import traceback
            traceback.print_exc()  # This will show the exact line where it fails
            return None

    # ---------- model path & metadata ----------
    def _model_path(self, ticker: str, model_type: str) -> str:
        safe = f"{ticker}_{model_type.replace(' ', '_')}"
        ext = ".keras" if model_type in ("LSTM", "Dense NN", "Transformer") else ".joblib"
        return os.path.join(self.config["model_dir"], safe + ext)

    def _meta_path(self, ticker: str) -> str:
        return os.path.join(self.config["model_dir"], f"{ticker}_{MODEL_META_FILENAME}")

    def load_meta(self, ticker: str) -> Dict[str, Any]:
        p = self._meta_path(ticker)
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load meta for {ticker}: {e}")
                return {}
        return {}

    def save_meta(self, ticker: str, meta: Dict[str, Any]):
        p = self._meta_path(ticker)
        try:
            with open(p, "w") as f:
                json.dump(meta, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save meta for {ticker}: {e}")

    # ---------- training helpers ----------
    def create_sequences(self, features: np.ndarray, targets: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models with multiple features"""
        X, y = [], []
        for i in range(window, len(features)):
            X.append(features[i-window:i])
            y.append(targets[i])
        return np.array(X), np.array(y)

    def _create_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Fixed LSTM model without attention layer issues"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(64, return_sequences=False),  # Changed to return_sequences=False
            Dropout(0.2),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(
            optimizer=Nadam(learning_rate=0.001),
            loss='huber_loss',
            metrics=[RootMeanSquaredError()]
        )
        return model

    def _create_dense_model(self, input_shape: Tuple[int]) -> Sequential:
        """Enhanced dense neural network"""
        model = Sequential([
            Input(shape=(input_shape[0],)),
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(1)
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=[RootMeanSquaredError()]
        )
        return model

    def _create_ensemble_model(self, base_models: List[Any]) -> VotingRegressor:
        """Create weighted ensemble of models"""
        estimators = [(f'model_{i}', model) for i, model in enumerate(base_models)]
        return VotingRegressor(estimators=estimators, weights=np.linspace(0.5, 1.5, len(base_models)))

    # ---------- model training ----------
    def train_all_models(self, ticker: str, force: bool = False) -> Dict[str, Any]:
        """
        Complete fixed implementation handling:
        - XGBoost NaN values
        - Ensemble training requirements
        - Missing calculate_metrics method
        - Backtesting integration
        """
        # 1. Data Preparation with NaN handling
        try:
            df = self.prepare_features(ticker)
            if df is None:
                raise ValueError("No data available")
                
            # Ensure we have required columns
            if 'adj_close' not in df.columns:
                df['adj_close'] = df.get('close', np.nan)
                
            # Create targets if missing
            df['target_price'] = df['adj_close'].shift(-1)
            df['target_return'] = (df['target_price'] / df['adj_close']) - 1.0
            df['target_direction'] = np.where(df['target_return'] > 0, 1, 0)
            
            # Critical: Remove any NaN/inf values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            if len(df) < self.config["min_data_points"]:
                raise ValueError(f"Only {len(df)} clean samples available")

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return {"error": f"Data preparation failed: {str(e)}"}

        # 2. Feature Engineering
        feature_cols = [c for c in df.columns if c not in 
                    ['target_price', 'target_return', 'target_direction']]

        # Save feature information
        meta = self.load_meta(ticker)
        meta['feature_columns'] = feature_cols  # Save the exact columns used
        meta['n_features'] = len(feature_cols)
        self.save_meta(ticker, meta)            
        
        # Scale features and targets separately
        feature_scaler = RobustScaler()
        target_scaler = StandardScaler()

        self.feature_list = feature_cols
        self._last_feature_scaler = feature_scaler
        self._last_target_scaler = target_scaler
        X = feature_scaler.fit_transform(df[feature_cols].values)
        y = target_scaler.fit_transform(df[['target_return']].values)

                
        joblib.dump(feature_scaler, os.path.join(self.config["model_dir"], f"{ticker}_feature_scaler.joblib"))
        joblib.dump(target_scaler, os.path.join(self.config["model_dir"], f"{ticker}_target_scaler.joblib"))

        # Create sequences
        window = self.config["window_size"]
        X_seq, y_seq = self.create_sequences(X, y, window)
        
        # Train-val split
        val_size = min(max(int(len(X_seq) * 0.2), 5), 30)  # 5-30 validation samples
        X_train, X_val = X_seq[:-val_size], X_seq[-val_size:]
        y_train, y_val = y_seq[:-val_size], y_seq[-val_size:]

        # 3. Model Training with Enhanced Error Handling
        # Define model training methods
        models = {
            "LSTM": self._train_lstm,
            "Dense NN": self._train_dnn,  
            "Random Forest": self._train_rf,
            "XGBoost": self._train_xgb,
        }
        
        
        results = {}
        trained_models = {}
        #print(X_train, y_train,X_val, y_val)
        
        for name, train_func in models.items():
            try:
                model_path = self._model_path(ticker, name)
                
                if not force and os.path.exists(model_path):
                    model = self.load_model(name, ticker)
                    results[name] = {"status": "loaded"}
                else:
                    logger.info(f"Training {name} for {ticker}")
                    model = train_func(X_train, y_train, X_val, y_val, len(feature_cols))
                    self._save_model(model, name, ticker)
                    results[name] = {"status": "trained"}
                    
                trained_models[name] = model
            except Exception as e:
                logger.error(f"Training failed for {name}: {e}")
                results[name] = {"status": f"error: {str(e)}"}

       # 4. Ensemble Training (only if we have at least 2 regressors)
        try:
            sk_models = {
                name: m for name, m in trained_models.items()
                if name in ("Random Forest", "XGBoost") and "Regressor" in str(type(m))
            }

            if len(sk_models) >= 2:
                ensemble = VotingRegressor(list(sk_models.items()))
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                ensemble.fit(X_train_flat, y_train.ravel())
                joblib.dump(ensemble, self._model_path(ticker, "Ensemble"))
                results["Ensemble"] = {"status": "trained"}
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            results["Ensemble"] = {"status": f"error: {str(e)}"}




        # 5. Backtesting and Metrics
        backtest_results = {}
        for name, model in trained_models.items():
            if model is None:
                continue
            try:
                bt_pack = self.walk_forward_backtest(df.copy(), model, name,
                                            feature_scaler, target_scaler,
                                            feature_cols, ticker)
                backtest_results[name] = bt_pack
            except Exception as e:
                logger.error(f"Backtest failed for {name}: {e}")
                backtest_results[name] = {
                    "walk_forward": pd.DataFrame(),
                    "cv_metrics": {},
                    "prediction_intervals": {}
                }

        # 6. Save final metrics
        #metrics_df = self._calculate_advanced_metrics(backtest_results)
        #metrics_df.to_csv(os.path.join(self.config["model_dir"], f"{ticker}_metrics.csv"))
        # 7. Save model meta
        # DEBUG: inspect backtest_results structure
        for name, val in backtest_results.items():
            print(f"--- BACKTEST RESULT FOR: {name} ---")
            print("type:", type(val))
            if isinstance(val, dict):
                print("keys:", list(val.keys()))
                wf = val.get("walk_forward")
                print("walk_forward type:", type(wf))
                if isinstance(wf, pd.DataFrame):
                    print("walk_forward cols:", wf.columns.tolist())
                    print("walk_forward head:\n", wf.head(3))
                    print("y_true non-null:", wf['y_true'].notna().sum() if 'y_true' in wf.columns else 'n/a')
                    print("y_pred non-null:", wf['y_pred'].notna().sum() if 'y_pred' in wf.columns else 'n/a')
                    print("PortfolioValue present:", 'PortfolioValue' in wf.columns)
            elif isinstance(val, pd.DataFrame):
                print("DataFrame cols:", val.columns.tolist())
                print("head:\n", val.head(3))
                print("y_true non-null:", val['y_true'].notna().sum() if 'y_true' in val.columns else 'n/a')
                print("y_pred non-null:", val['y_pred'].notna().sum() if 'y_pred' in val.columns else 'n/a')
                print("PortfolioValue present:", 'PortfolioValue' in val.columns)
            else:
                print("Value preview:", str(val)[:200])


        self._save_training_results(ticker, results, backtest_results)

        return {
            "training_results": results,
            "backtest": backtest_results,
            "best_model": self._select_best_model(backtest_results)
        }


    # Helper Methods --------------------------------------------------
    def _save_model(self, model: Any, model_type: str, ticker: str):
        """Save a trained model with proper error handling"""
        path = self._model_path(ticker, model_type)
        try:
            if model_type in ("LSTM", "Dense NN"):
                save_model(model, path)
            else:
                joblib.dump(model, path)
            logger.info(f"Saved {model_type} model for {ticker}")
        except Exception as e:
            logger.error(f"Failed to save {model_type} model: {e}")
            raise
    
    def _train_lstm(self, X_train, y_train, X_val, y_val, n_features):
        """Train LSTM model"""
        try:
            model = self._create_lstm_model((self.config["window_size"], n_features))
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                verbose=0,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=5)
                ]
            )
            return model
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            raise

    def _train_dnn(self, X_train, y_train, X_val, y_val, n_features):
        from tensorflow.keras import Sequential, layers, callbacks

        # Flatten sequences to 2D for Dense network
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        model = Sequential([
            layers.Input(shape=(X_train_flat.shape[1],)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(
            X_train_flat, y_train,
            validation_data=(X_val_flat, y_val),
            epochs=100,
            batch_size=32,
            verbose=0,
            callbacks=[es]
        )
        return model



    def _train_rf(self, X_train, y_train, X_val, y_val, n_features):
        """Train Random Forest classifier on direction (up/down)"""
        # Convert targets to direction
        y_train_cls = (y_train > 0).astype(int).ravel()
        y_val_cls = (y_val > 0).astype(int).ravel()

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_flat, y_train_cls)
        return model


    def _train_xgb(self, X_train, y_train, X_val, y_val, n_features):
        """Train XGBoost classifier on direction (up/down)"""
        y_train_cls = (y_train > 0).astype(int).ravel()
        y_val_cls = (y_val > 0).astype(int).ravel()

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
        )
        model.fit(
            X_train_flat, y_train_cls,
            eval_set=[(X_val_flat, y_val_cls)],
            early_stopping_rounds=20,
            verbose=False
        )
        return model

  

    def _calculate_backtest_metrics(self, bt_df):
        """Calculate comprehensive backtest metrics"""
        if bt_df.empty:
            return {}
        
        returns = bt_df['PortfolioValue'].pct_change().dropna()
        metrics = {
            "Return": (bt_df['PortfolioValue'].iloc[-1] / bt_df['PortfolioValue'].iloc[0] - 1) * 100,
            "MaxDrawdown": (bt_df['PortfolioValue'] / bt_df['PortfolioValue'].cummax() - 1).min() * 100,
            "Volatility": returns.std() * np.sqrt(252) * 100,
            "Sharpe": returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            "WinRate": (bt_df['Signal'] == np.where(bt_df['TruePrice'].diff() > 0, 'BUY', 'SELL')).mean() * 100
        }
        return metrics



    def _cross_validate_model(self, model_type: str, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        features = df[feature_cols].values
        targets = df['target_return'].values.reshape(-1, 1)

        feature_scaler = RobustScaler()
        scaled_features = feature_scaler.fit_transform(features)

        target_scaler = StandardScaler()
        scaled_targets = target_scaler.fit_transform(targets)

        window = self.config["window_size"]
        X, y = self.create_sequences(scaled_features, scaled_targets, window)

        tscv = TimeSeriesSplit(n_splits=self.config["n_cv_folds"])
        metrics = defaultdict(list)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                if model_type == "LSTM":
                    model = self._create_lstm_model(X_train.shape[1:])
                    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
                    preds_scaled = model.predict(X_test, verbose=0).flatten()
                    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
                    y_true = target_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
                    direction_acc = np.mean(np.sign(y_true) == np.sign(preds)) * 100

                elif model_type == "Dense NN":
                    model = self._create_dense_model((X_train.shape[1]*X_train.shape[2],))
                    model.fit(X_train.reshape(X_train.shape[0], -1), y_train, epochs=30, batch_size=32, verbose=0)
                    preds_scaled = model.predict(X_test.reshape(X_test.shape[0], -1), verbose=0).flatten()
                    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
                    y_true = target_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
                    direction_acc = np.mean(np.sign(y_true) == np.sign(preds)) * 100

                else:
                    # Tree-based models as classifiers
                    # 1 if next day return > 0, else 0
                    y_train_cls = (y_train > 0).astype(int).ravel()
                    y_test_cls = (y_test > 0).astype(int).ravel()
                    
                    if model_type == "Random Forest":
                        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
                    else:  # XGBoost
                        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.01,
                                            subsample=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')
                    
                    model.fit(X_train.reshape(X_train.shape[0], -1), y_train_cls)
                    preds_cls = model.predict(X_test.reshape(X_test.shape[0], -1))
                    direction_acc = np.mean(preds_cls == y_test_cls) * 100

                # Store metrics
                metrics['direction_accuracy'].append(direction_acc)

            except Exception as e:
                logger.warning(f"CV fold {fold} failed for {model_type}: {e}")
                continue

        # Return mean + std for all folds
        return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in metrics.items()}



    def _calculate_prediction_intervals(self, model: Any, df: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """Calculate prediction intervals for uncertainty estimation"""
        try:
            feature_cols = [c for c in df.columns if c not in ['target_price', 'target_return', 'target_direction']]
            features = df[feature_cols].values
            targets = df['target_return'].values.reshape(-1, 1)

            feature_scaler = RobustScaler()
            scaled_features = feature_scaler.fit_transform(features)
            
            target_scaler = StandardScaler()
            scaled_targets = target_scaler.fit_transform(targets)

            window = self.config["window_size"]
            X, y = self.create_sequences(scaled_features, scaled_targets, window)
            
            # FIX: Proper reshaping based on model type
            if model_type in ["LSTM"]:
                # LSTM expects 3D: (samples, timesteps, features)
                input_data = X
            else:
                # Tree-based models expect 2D: (samples, features)
                input_data = X.reshape(X.shape[0], -1)
            
            # Get predictions
            preds = model.predict(input_data)
            preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            y_true = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
            
            # Calculate residuals and standard deviation
            residuals = y_true - preds
            std_dev = np.std(residuals)
            
            # Calculate prediction intervals
            intervals = {}
            for alpha in [0.68, 0.95]:
                z_score = norm.ppf(1 - (1 - alpha)/2)
                margin = z_score * std_dev
                intervals[f"{int(alpha*100)}%"] = {
                    "lower": preds[-1] - margin,
                    "upper": preds[-1] + margin,
                    "width": 2 * margin
                }
                
            return {
                "last_prediction": preds[-1],
                "std_dev": std_dev,
                "intervals": intervals
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate prediction intervals: {e}")
            return {}

    def _save_training_results(self, ticker: str, training_results: Dict[str, Any], 
                           backtest_results: Dict[str, Any]) -> None:
        """
        Normalize backtest_results, compute metrics, save CSV, save meta.
        Ensures metrics JSON is records-orientated and JSON-safe (NaN -> None).
        """

        import pandas as pd
        import numpy as np
        from datetime import datetime

        normalized = {}


        for m, val in backtest_results.items():
            if isinstance(val, pd.DataFrame):
                normalized[m] = val
            elif isinstance(val, dict) and isinstance(val.get("walk_forward"), pd.DataFrame):
                normalized[m] = val.get("walk_forward")
            else:
                # fallback: empty df with expected columns
                normalized[m] = pd.DataFrame(columns=[
                    "Date","TruePrice","PredictedPrice","Signal","PortfolioValue","PredictedReturn","y_true","y_pred"
                ])


        # Suppose normalized is your dict of DataFrames
        wrapped_results = {model: {"walk_forward": df} for model, df in normalized.items()}

        metrics_df = self._calculate_advanced_metrics(wrapped_results)
        print(metrics_df)


        # Save metrics CSV (index=False)
        metrics_csv = os.path.join(self.config["model_dir"], f"{ticker}_metrics.csv")
        try:
            metrics_df.to_csv(metrics_csv, index=False)
        except Exception as e:
            logger.warning(f"Failed to write metrics CSV: {e}")

        # JSON-safe metrics: replace NaN with None
        metrics_df_clean = metrics_df.where(pd.notnull(metrics_df), None)

        # Persist metadata: make metrics a records list
        meta = {
            "last_trained": datetime.utcnow().isoformat(),
            "model_paths": {m: self._model_path(ticker, m) for m in training_results.keys()},
            "metrics_file": metrics_csv,
            "metrics": metrics_df_clean.to_dict(orient="records"),
            # store the 'raw' best model decision from selector (it should accept normalized dict)
            "best_model": self._select_best_model(normalized),
        }

        # Save meta to disk using your save_meta function
        try:
            self.save_meta(ticker, meta)
        except Exception as e:
            logger.error(f"Could not save meta for {ticker}: {e}")

        logger.info(f"Saved training results for {ticker} -> {metrics_csv}")



    def _calculate_advanced_metrics(self, backtest_results: dict, target_scaler=None) -> pd.DataFrame:
        """
        Compute performance metrics for all model backtest results.
        Handles:
            - Regression (MAE, RMSE, R2, Direction Accuracy)
            - Classification (Direction Accuracy)
            - Portfolio metrics (Final value, Sharpe, Volatility)
        Applies inverse transform to NN outputs if target_scaler is provided.
        """

        import numpy as np
        import pandas as pd
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = []
        print("calc;:" , backtest_results)
        for model_name, result in backtest_results.items():
            print(f"\nDEBUG: --- Calculating metrics for model: {model_name} ---")

            # Extract walk-forward DataFrame
            df = result.get("walk_forward")
            cv_metrics = result.get("cv_metrics", {})
            print(f"DEBUG: Raw walk_forward DataFrame: {type(df)}, shape: {getattr(df, 'shape', None)}")

            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                print(f"DEBUG: Empty or invalid DataFrame for {model_name}, using cv_metrics or NaN")
                metrics.append({
                    "Model": model_name,
                    "MAE": cv_metrics.get("MAE", np.nan),
                    "RMSE": cv_metrics.get("RMSE", np.nan),
                    "R2": cv_metrics.get("R2", np.nan),
                    "DirectionAcc": np.nan,
                    "FinalPortfolio": np.nan,
                    "Sharpe": np.nan,
                    "Volatility": np.nan
                })
                continue

            df = df.copy()
            print(f"DEBUG: Columns in DataFrame: {df.columns.tolist()}")
            print(f"DEBUG: First 3 rows:\n{df.head(3)}")

            # --- Regression / NN metrics ---
            mae = rmse = r2 = dir_acc = np.nan
            if "y_true" in df.columns and "y_pred" in df.columns:
                df = df.dropna(subset=["y_true", "y_pred"])
                print(f"DEBUG: Dropped NA from y_true/y_pred, remaining rows: {len(df)}")

                if not df.empty:
                    y_true = df["y_true"].values
                    y_pred = df["y_pred"].values
                    print(f"DEBUG: Sample y_true: {y_true[:5]}")
                    print(f"DEBUG: Sample y_pred before scaling: {y_pred[:5]}")

                    # Inverse scale if scaler provided
                    if target_scaler is not None:
                        y_true = target_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
                        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                        print(f"DEBUG: Sample y_true after inverse scaling: {y_true[:5]}")
                        print(f"DEBUG: Sample y_pred after inverse scaling: {y_pred[:5]}")

                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    r2 = r2_score(y_true, y_pred)
                    dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true))

                    print(f"DEBUG: MAE={mae}, RMSE={rmse}, R2={r2}, DirectionAcc={dir_acc}")
                else:
                    print(f"DEBUG: Empty after dropping NA, metrics will be NaN")
            else:
                print(f"DEBUG: y_true/y_pred not found, checking signals for direction accuracy")
                if "Signal" in df.columns and "y_true" in df.columns:
                    y_true_dir = np.sign(df["y_true"].values)
                    signal_dir = np.array([1 if s == "BUY" else -1 if s == "SELL" else 0 for s in df["Signal"].values])
                    dir_acc = np.mean(signal_dir == y_true_dir)
                    print(f"DEBUG: DirectionAcc from signals: {dir_acc}")
                else:
                    print(f"DEBUG: No Signal column or y_true, DirectionAcc=NaN")

            # --- Portfolio metrics ---
            final_port = sharpe = vol = np.nan
            if "PortfolioValue" in df.columns:
                final_port = df["PortfolioValue"].iloc[-1]
                returns = df["PortfolioValue"].pct_change().dropna()
                if not returns.empty:
                    mean_ret = returns.mean()
                    vol = returns.std()
                    sharpe = mean_ret / vol * np.sqrt(252) if vol > 0 else np.nan
                print(f"DEBUG: FinalPortfolio={final_port}, Sharpe={sharpe}, Volatility={vol}")
            else:
                print(f"DEBUG: PortfolioValue not found, portfolio metrics NaN")

            metrics.append({
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "DirectionAcc": dir_acc,
                "FinalPortfolio": final_port,
                "Sharpe": sharpe,
                "Volatility": vol
            })

        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.sort_values(by="DirectionAcc", ascending=False)
        print("\nDEBUG: Final metrics DataFrame:\n", metrics_df)
        return metrics_df







    def _select_best_model(self, backtest_results: Dict[str, Any]) -> str:
        """
        Choose the best model based on regression R2 or DirectionAcc.
        Handles cases where backtest_results values may be DataFrames.
        """
        best_model = None
        best_score = -np.inf

        for model_name, result in backtest_results.items():
            # If result is a DataFrame, compute score from metrics
            if isinstance(result, pd.DataFrame):
                df = result.copy()
                if "y_true" in df.columns and "y_pred" in df.columns:
                    # Regression: use R²
                    try:
                        score = r2_score(df["y_true"], df["y_pred"])
                    except Exception:
                        score = -np.inf
                elif "Signal" in df.columns and "y_true" in df.columns:
                    # Direction accuracy
                    signals = df["Signal"].values
                    y_true_dir = np.sign(df["y_true"].values)
                    signal_dir = np.array([1 if s=="BUY" else -1 if s=="SELL" else 0 for s in signals])
                    score = np.mean(signal_dir == y_true_dir)
                else:
                    score = -np.inf
            elif isinstance(result, dict):
                # Already a metrics dict
                score = result.get("R2", result.get("DirectionAcc", -np.inf))
            else:
                score = -np.inf

            if score > best_score:
                best_score = score
                best_model = model_name

        return best_model



    # ---------- load/save ----------
    def load_model(self, model_type: str, ticker: str) -> Any:
        """Load a trained model with additional checks"""
        path = self._model_path(ticker, model_type)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
            
        try:
            if model_type in ("LSTM", "Dense NN"):
                model = load_model(path, compile=False)
                # Recompile if needed
                if not hasattr(model, 'optimizer'):
                    model.compile(optimizer=Adam(), loss='mse')
                return model
            else:
                return joblib.load(path)
        except Exception as e:
            raise ValueError(f"Could not load model: {e}")

    # ---------- prediction ----------
    def predict_tomorrow(self, ticker: str) -> Dict[str, Any]:
        """
        Predict next-day price/return for all models and generate an ensemble.
        Returns structured dictionary with individual predictions + ensemble.
        """
        df = self.prepare_features(ticker)
        if df is None:
            return {"error": "Could not prepare data"}

        feature_cols = [c for c in df.columns if c not in ['target_price', 'target_return', 'target_direction']]
        features = df[feature_cols].values

        # Load scalers
        fs_path = os.path.join(self.config["model_dir"], f"{ticker}_feature_scaler.joblib")
        ts_path = os.path.join(self.config["model_dir"], f"{ticker}_target_scaler.joblib")
        feature_scaler = joblib.load(fs_path) if os.path.exists(fs_path) else RobustScaler().fit(features)
        target_scaler = joblib.load(ts_path) if os.path.exists(ts_path) else StandardScaler().fit(df[['target_return']])

        scaled_features = feature_scaler.transform(features)
        window = self.config["window_size"]

        # Build last window for time series models
        X_seq = []
        for i in range(window, len(scaled_features)+1):
            X_seq.append(scaled_features[i-window:i])
        X_seq = np.array(X_seq)
        if X_seq.size == 0:
            return {"error": "Not enough data for prediction"}
        last_window = X_seq[-1:]

        predictions = {}
        last_price = float(df["adj_close"].iloc[-1])
        threshold = self.config["prediction_threshold_pct"]
        model_types = ["LSTM", "Dense NN", "Random Forest", "XGBoost"]

        for model_type in model_types:
            try:
                model = self.load_model(model_type, ticker)
                input_data = last_window if model_type in ["LSTM", "Dense NN"] else last_window.reshape(1, -1)

                # Predict scaled returns
                if model_type == "LSTM":
                    pred_scaled = model.predict(input_data, verbose=0).flatten()[0]
                elif model_type == "Dense NN":
                    # Flatten sequence window for Dense model
                    flat_input = input_data.reshape(1, -1)
                    pred_scaled = model.predict(flat_input, verbose=0).flatten()[0]
                else:
                    pred_scaled = model.predict(input_data).flatten()[0]

                # Inverse scale to returns
                pred_return = target_scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0]
                pred_price = last_price * (1 + pred_return)
                pct_diff = (pred_price - last_price) / last_price * 100
                signal = "BUY" if pct_diff > threshold else ("SELL" if pct_diff < -threshold else "HOLD")

                predictions[model_type] = {
                    "predicted_price": float(pred_price),
                    "predicted_return": float(pred_return),
                    "pct_diff": float(pct_diff),
                    "signal": signal
                }
            except Exception as e:
                predictions[model_type] = {"error": str(e)}

        # Generate ensemble prediction automatically
        ensemble_pred = self._generate_ensemble_predictions(predictions)
        if ensemble_pred:
            predictions["Ensemble"] = ensemble_pred

        return predictions



    def _generate_ensemble_predictions(self, model_preds: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine model predictions into a single ensemble signal and predicted price.
        Regressors contribute via predicted return; classifiers contribute via BUY/SELL.
        """
        signals = []
        prices = []

        for model, pred in model_preds.items():
            if "error" in pred:
                continue
            # Collect signals
            signal = pred.get("signal")
            if signal:
                signals.append(signal)

            # Collect predicted prices (only regressors really matter)
            pred_price = pred.get("predicted_price")
            if pred_price:
                prices.append(pred_price)

        if not signals:
            return {"error": "No valid model predictions for ensemble"}

        # --- Directional vote ---
        buy_votes = signals.count("BUY")
        sell_votes = signals.count("SELL")
        hold_votes = signals.count("HOLD")

        # Choose signal with most votes
        if buy_votes > sell_votes and buy_votes > hold_votes:
            ensemble_signal = "BUY"
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            ensemble_signal = "SELL"
        else:
            ensemble_signal = "HOLD"

        # Ensemble predicted price = mean of regressor prices (ignore classifiers)
        ensemble_price = np.mean(prices) if prices else None

        return {
            "signal": ensemble_signal,
            "predicted_price": float(ensemble_price) if ensemble_price is not None else None
        }


    # ---------- backtesting ----------
    def walk_forward_backtest(self, df: pd.DataFrame, model: Any, model_type: str,
                          feature_scaler: Any, target_scaler: Any,
                          feature_cols: List[str], ticker: str) -> pd.DataFrame:
        """
        Walk-forward backtest producing fully structured DataFrame compatible with metrics calculation.
        Handles different model types (LSTM, Dense NN, RF, XGB, classifiers) correctly.
        """
        try:
            # --- Determine features ---
            training_features = feature_cols
            if hasattr(self, 'feature_list') and self.feature_list is not None:
                training_features = [c for c in self.feature_list if c in df.columns]
            else:
                meta = self.load_meta(ticker)
                training_features = meta.get('feature_columns', feature_cols)
                training_features = [f for f in training_features if f in df.columns]

            window = self.config.get("window_size", 30)
            threshold_pct = self.config.get("prediction_threshold_pct", 0.01)

            if len(df) < window + 10:
                logger.warning("Not enough data for backtest")
                return pd.DataFrame(columns=[
                    "Date","TruePrice","PredictedPrice","Signal","PortfolioValue",
                    "PredictedReturn","y_true","y_pred"
                ])

            # --- Prepare features and targets ---
            features = df[training_features].values
            true_returns = df['target_return'].values.reshape(-1, 1)
            prices = df['adj_close'].values
            dates = df.index

            scaled_features = feature_scaler.transform(features)
            scaled_returns = target_scaler.transform(true_returns)

            # --- Build sequences for LSTM / sequential models ---
            X_seq = np.array([scaled_features[i-window:i] for i in range(window, len(scaled_returns))])
            y_seq = np.array([scaled_returns[i] for i in range(window, len(scaled_returns))]).reshape(-1,1)
            n = X_seq.shape[0]

            if n == 0:
                logger.warning("No sequences created for backtest")
                return pd.DataFrame(columns=[
                    "Date","TruePrice","PredictedPrice","Signal","PortfolioValue",
                    "PredictedReturn","y_true","y_pred"
                ])

            # --- Safe prediction function ---
            def _safe_predict(m, X):

                try:
                    # Classifier
                    if hasattr(m, "predict_proba"):
                        X_for = X.reshape(X.shape[0], -1) if X.ndim==3 else X
                        preds = m.predict(X_for)
                        proba = None
                        try:
                            proba = m.predict_proba(X_for)
                        except Exception:
                            proba = None
                        return {"kind": "classifier", "preds": np.array(preds).ravel(), "proba": proba}

                    # Keras / TensorFlow models
                    elif "keras" in str(type(m)).lower() or "tensorflow" in str(type(m)).lower():
                        input_shape = m.input_shape
                        if len(input_shape) == 2:  # Dense NN
                            X_for = X.reshape(X.shape[0], -1)
                        else:  # LSTM
                            X_for = X
                        preds = m.predict(X_for, verbose=0)
                        return {"kind": "regressor", "preds": np.array(preds).ravel(), "proba": None}


                    # Standard scikit-learn regressors (RF, XGB)
                    else:
                        X_for = X.reshape(X.shape[0], -1) if X.ndim==3 else X
                        preds = m.predict(X_for)
                        return {"kind": "regressor", "preds": np.array(preds).ravel(), "proba": None}

                except Exception as e:
                    logger.error(f"Prediction failed: {e}")
                    return {"kind": "error", "error": str(e)}


            pred_info = _safe_predict(model, X_seq)

            # --- Initialize containers ---
            y_true = target_scaler.inverse_transform(y_seq).ravel()
            prev_prices = prices[window-1:-1]
            true_prices = prices[window:]
            signals = np.array(["HOLD"] * n)
            pred_returns = np.full(n, np.nan, dtype=float)
            pred_prices = np.full(n, np.nan, dtype=float)

            # --- Process predictions ---
            if pred_info["kind"] == "regressor":
                preds_scaled = pred_info["preds"]
                if len(preds_scaled) != n:
                    preds_scaled = np.resize(preds_scaled, n)
                pred_returns = target_scaler.inverse_transform(preds_scaled.reshape(-1,1)).ravel()
                pred_prices = prev_prices * (1.0 + pred_returns)
                pct_diffs = (pred_prices - true_prices) / true_prices
                signals = np.where(pct_diffs > threshold_pct, "BUY",
                                np.where(pct_diffs < -threshold_pct, "SELL", "HOLD"))

            elif pred_info["kind"] == "classifier":
                cls_preds = pred_info["preds"]
                if len(cls_preds) != n:
                    cls_preds = np.resize(cls_preds, n)
                proba = pred_info.get("proba")
                hold_mask = (proba is not None) and (np.max(proba, axis=1) < 0.6)
                signals = np.where(cls_preds == 1, "BUY", "SELL")
                if proba is not None:
                    signals[hold_mask] = "HOLD"
                pred_prices = prev_prices
                # Use numeric encoding for metrics
                pred_returns = np.where(signals=="BUY", 1, np.where(signals=="SELL",-1,0))

            else:
                logger.warning("Prediction kind unknown")
                pred_returns = np.full(n, np.nan)

            # --- Portfolio simulation ---
            cash = self.config.get("initial_capital", 10000)
            positions = 0
            portfolio = []
            for i, sig in enumerate(signals):
                price = true_prices[i]
                if sig == "BUY" and positions == 0:
                    qty = int((cash*0.95)//price)
                    if qty>0:
                        positions = qty
                        cash -= qty*price
                elif sig == "SELL" and positions>0:
                    cash += positions*price
                    positions = 0
                portfolio.append(cash + positions*price)
            if positions>0:
                cash += positions*prices[-1]
                portfolio[-1] = cash

            # --- Build result DataFrame ---
            df_result = pd.DataFrame({
                "Date": dates[window:],
                "TruePrice": true_prices,
                "PredictedPrice": pred_prices,
                "Signal": signals,
                "PortfolioValue": portfolio,
                "PredictedReturn": pred_returns,
                "y_true": y_true,
                "y_pred": pred_returns  # numeric predicted returns for metrics
            })

            return df_result

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return pd.DataFrame(columns=[
                "Date","TruePrice","PredictedPrice","Signal","PortfolioValue",
                "PredictedReturn","y_true","y_pred"
            ])





    # ---------- convenience ----------
    def ensure_trained(self, ticker: str, force: bool = False) -> Dict[str, Any]:
        """Public wrapper with enhanced error handling"""
        try:
            start_time = time.time()
            res = self.train_all_models(ticker, force=force)
            res['training_time'] = time.time() - start_time
            return res
        except Exception as e:
            logger.exception("Training failed: %s", e)
            return {"error": str(e), "status": "failed"}


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Initialize with custom config
    config = {
        "window_size": 30,
        "prediction_threshold_pct": 0.2,
        "enable_uncertainty": True,
        "verbose": True
    }
    
    system = TradingModelSystem(config)
    ticker = "AAPL"
    
    # Train or load models
    training_result = system.ensure_trained(ticker, force=False)
    print("\nTraining Summary:")
    print(f"Best model: {training_result.get('best_model', 'unknown')}")
    print(f"Metrics: {training_result.get('metrics', {}).to_dict()}")
    
    # Get predictions
    predictions = system.predict_tomorrow(ticker)
    print("\nTomorrow's Predictions:")
    for model, pred in predictions.items():
        print(f"{model}: {pred.get('signal', 'N/A')} ({pred.get('pct_diff', 0):.2f}%)")
    
    # Show ensemble prediction if available
    if 'Ensemble' in predictions:
        ensemble = predictions['Ensemble']
        print(f"\nEnsemble Prediction:")
        print(f"Mean: {ensemble['mean_prediction']:.2f} ({ensemble['pct_diff_mean']:.2f}%)")
        print(f"Weighted: {ensemble['weighted_prediction']:.2f} ({ensemble['pct_diff_weighted']:.2f}%)")
        print(f"Signal: {ensemble['signal_weighted']}")
