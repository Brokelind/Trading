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
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from xgboost import XGBRegressor
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
            print("DEBUG: Adding technical indicators...")
            df = self._add_technical_indicators(df)
            print(f"DEBUG: After technical indicators, shape: {df.shape}")
            print(f"DEBUG: Columns: {list(df.columns)}")
            
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
            "Dense NN": self._train_dnn,  # Changed from _train_dense to _train_dnn
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

        # 4. Ensemble Training (only if we have at least 2 models)
        try:
            sk_models = {name: m for name, m in trained_models.items()
                        if name in ("Random Forest", "XGBoost")}
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
                bt_pack = self._run_backtest(df.copy(), model, name,
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
        metrics_df = self._calculate_advanced_metrics(backtest_results)
        metrics_df.to_csv(os.path.join(self.config["model_dir"], f"{ticker}_metrics.csv"))
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
        """Train Dense Neural Network"""
        try:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            
            model = self._create_dense_model((X_train_flat.shape[1],))
            model.fit(
                X_train_flat, y_train,
                validation_data=(X_val_flat, y_val),
                epochs=50,
                batch_size=32,
                verbose=0,
                callbacks=[
                    EarlyStopping(patience=8, restore_best_weights=True)
                ]
            )
            return model
        except Exception as e:
            logger.error(f"Dense NN training failed: {e}")
            raise

    
    def _train_rf(self, X_train, y_train, X_val, y_val, n_features):
        """Train Random Forest model"""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_flat, y_train.ravel())
        return model

    def _train_xgb(self, X_train, y_train, X_val, y_val, n_features):
        """Train XGBoost model with proper feature handling"""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        # Ensure no NaN/inf in targets
        y_train = np.nan_to_num(y_train.ravel())
        y_val = np.nan_to_num(y_val.ravel())
        
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(
            X_train_flat, y_train,
            eval_set=[(X_val_flat, y_val)],
            early_stopping_rounds=20,
            verbose=0
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

    def _run_backtest(
        self,
        df: pd.DataFrame,
        model: Any,
        model_type: str,
        feature_scaler: Any,
        target_scaler: Any,
        feature_cols: List[str],
        ticker: str
    ) -> Dict[str, Any]:
        """Run walk-forward backtest, CV, and prediction intervals for a single model."""
        try:
            wf_result = self.walk_forward_backtest(
                df.copy(),
                model,
                model_type,
                feature_scaler,
                target_scaler,
                feature_cols,
                ticker
            )

            cv_metrics = self._cross_validate_model(
                model_type,
                df.copy(),
                feature_cols
            )

            pred_intervals = {}
            if self.config.get("enable_uncertainty", False) and model_type not in ["LSTM", "Dense NN"]:
                pred_intervals = self._calculate_prediction_intervals(
                    model,
                    df.copy(),
                    model_type
                )

            result = {
                "walk_forward": wf_result if isinstance(wf_result, pd.DataFrame) else pd.DataFrame(),
                "cv_metrics": cv_metrics,
                "prediction_intervals": pred_intervals
            }

            if not result["walk_forward"].empty:
                bt_csv = os.path.join(self.config["model_dir"], f"{ticker}_{model_type}_backtest.csv")
                result["walk_forward"].to_csv(bt_csv, index=False)

            return result

        except Exception as e:
            logger.error(f"Backtest failed for {model_type}: {e}")
            return {
                "walk_forward": pd.DataFrame(),
                "cv_metrics": {},
                "prediction_intervals": {}
            }



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
                    preds = model.predict(X_test, verbose=0).flatten()
                elif model_type == "Dense NN":
                    model = self._create_dense_model((X_train.shape[1] * X_train.shape[2],))
                    model.fit(X_train.reshape(X_train.shape[0], -1), y_train, epochs=30, batch_size=32, verbose=0)
                    preds = model.predict(X_test.reshape(X_test.shape[0], -1), verbose=0).flatten()
                else:
                    model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5,
                                                random_state=42, n_jobs=-1) if model_type == "Random Forest" \
                            else XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.01,
                                            subsample=0.8, random_state=42, n_jobs=-1)
                    model.fit(X_train.reshape(X_train.shape[0], -1), y_train.ravel())
                    preds = model.predict(X_test.reshape(X_test.shape[0], -1))

                preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
                y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

                metrics['mae'].append(mean_absolute_error(y_true, preds))
                metrics['mse'].append(mean_squared_error(y_true, preds))
                metrics['r2'].append(r2_score(y_true, preds))
                metrics['direction_accuracy'].append(
                    np.mean(np.sign(y_true) == np.sign(preds)) * 100
                )
            except Exception as e:
                logger.warning(f"CV fold {fold} failed for {model_type}: {e}")
                continue

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
        """Save all training results and metadata"""
        # Calculate comprehensive metrics
        metrics_df = self._calculate_advanced_metrics(backtest_results)
        metrics_csv = os.path.join(self.config["model_dir"], f"{ticker}_metrics.csv")
        metrics_df.to_csv(metrics_csv)
        
        # Save metadata
        meta = {
            "last_trained": datetime.utcnow().isoformat(),
            "model_paths": {m: self._model_path(ticker, m) for m in training_results.keys()},
            "metrics_file": metrics_csv,
            "metrics": metrics_df.to_dict(),
            "best_model": self._select_best_model(backtest_results),
        }
        self.save_meta(ticker, meta)
        
        logger.info(f"Saved training results for {ticker}")

    def _calculate_advanced_metrics(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Enhanced metric calculation with additional performance measures"""
        metrics = {}
        
        for model_name, result in results.items():
            if result['walk_forward'].empty:
                continue
                
            df = result['walk_forward']
            cv_metrics = result.get('cv_metrics', {})
            
            # Basic metrics
            returns = df['PortfolioValue'].pct_change().dropna()
            cum_return = (df['PortfolioValue'].iloc[-1] / self.config["initial_capital"] - 1) * 100
            rolling_max = df['PortfolioValue'].cummax()
            drawdown = (df['PortfolioValue'] - rolling_max) / rolling_max
            max_dd = drawdown.min() * 100
            vol = returns.std() * math.sqrt(252) * 100 if len(returns) > 1 else 0.0
            sharpe = returns.mean() / returns.std() * math.sqrt(252) if returns.std() > 0 else 0.0
            sortino = returns.mean() / returns[returns < 0].std() * math.sqrt(252) if len(returns[returns < 0]) > 0 else 0.0
            
            # Prediction metrics
            pred_metrics = {}
            if set(['TruePrice', 'PredictedPrice']).issubset(df.columns):
                errors = np.abs(df['PredictedPrice'] - df['TruePrice'])
                mae_pct = (errors / df['TruePrice']).mean() * 100
                dir_acc = (np.sign(df['PredictedPrice'].diff()) == np.sign(df['TruePrice'].diff())).mean() * 100
                pred_metrics = {
                    "MAE (%)": mae_pct,
                    "Direction Accuracy (%)": dir_acc,
                    "Hit Rate (%)": (df['Signal'] == np.where(df['TruePrice'].diff() > 0, 'BUY', 'SELL')).mean() * 100
                }
            
            # Combine all metrics
            metrics[model_name] = {
                "Return (%)": cum_return,
                "Max Drawdown (%)": float(max_dd),
                "Volatility (%)": float(vol),
                "Sharpe Ratio": float(sharpe),
                "Sortino Ratio": float(sortino),
                "CV MAE (mean)": cv_metrics.get('mae', {}).get('mean', 0),
                "CV R2 (mean)": cv_metrics.get('r2', {}).get('mean', 0),
                **pred_metrics
            }
            
        return pd.DataFrame(metrics).T

    def _select_best_model(self, backtest_results: Dict[str, Any]) -> str:
        """Select best model based on multiple weighted metrics"""
        if not backtest_results:
            return "none"
            
        model_scores = []
        for model_name, result in backtest_results.items():
            if result['walk_forward'].empty:
                continue
                
            # Get metrics
            df = result['walk_forward']
            returns = df['PortfolioValue'].pct_change().dropna()
            sharpe = returns.mean() / returns.std() * math.sqrt(252) if returns.std() > 0 else 0.0
            max_dd = (df['PortfolioValue'] / df['PortfolioValue'].cummax() - 1).min()
            
            # Get prediction accuracy if available
            if set(['TruePrice', 'PredictedPrice']).issubset(df.columns):
                dir_acc = (np.sign(df['PredictedPrice'].diff()) == np.sign(df['TruePrice'].diff())).mean()
            else:
                dir_acc = 0.5
                
            # Combine metrics into a score (weights can be adjusted)
            score = (sharpe * 0.4 + 
                    (1 - abs(max_dd)) * 0.3 + 
                    dir_acc * 0.3)
                    
            model_scores.append((model_name, score))
            
        if not model_scores:
            return "none"
            
        return max(model_scores, key=lambda x: x[1])[0]

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
    def predict_tomorrow(self, ticker: str) -> Dict[str, Dict[str, Any]]:
        df = self.prepare_features(ticker)
        if df is None:
            return {"error": "Could not prepare data"}

        # Use same feature set logic as training/backtest
        feature_cols = [c for c in df.columns if c not in ['target_price', 'target_return', 'target_direction']]
        features = df[feature_cols].values

        # Load scalers if available (best), else fit anew
        fs_path = os.path.join(self.config["model_dir"], f"{ticker}_feature_scaler.joblib")
        ts_path = os.path.join(self.config["model_dir"], f"{ticker}_target_scaler.joblib")

        if os.path.exists(fs_path):
            feature_scaler = joblib.load(fs_path)
        else:
            feature_scaler = RobustScaler().fit(features)

        # IMPORTANT: target scaler must be on returns
        if os.path.exists(ts_path):
            target_scaler = joblib.load(ts_path)
        else:
            target_scaler = StandardScaler().fit(df[['target_return']])

        scaled_features = feature_scaler.transform(features)

        window = self.config["window_size"]
        # Build last window
        X_seq = []
        for i in range(window, len(scaled_features)+1):
            X_seq.append(scaled_features[i-window:i])
        X_seq = np.array(X_seq)
        if X_seq.size == 0:
            return {"error": "Not enough data for prediction"}

        last_window = X_seq[-1:]

        predictions = {}
        # Only models you actually train and/or aggregate
        model_types = ["LSTM", "Dense NN", "Random Forest", "XGBoost"]  # drop LightGBM, exclude Ensemble here

        last_price = float(df["adj_close"].iloc[-1])
        threshold = self.config["prediction_threshold_pct"]

        for model_type in model_types:
            try:
                model = self.load_model(model_type, ticker)
                input_data = last_window if model_type == "LSTM" else last_window.reshape(1, -1)

                # Predict scaled returns; inverse-scale to returns
                pred_scaled = (model.predict(input_data, verbose=0).flatten()[0]
                            if model_type in ["LSTM", "Dense NN"]
                            else model.predict(input_data).flatten()[0])

                pred_return = target_scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0]
                pred_price = last_price * (1.0 + pred_return)
                pct_diff = (pred_price - last_price) / last_price * 100.0
                signal = "BUY" if pct_diff > threshold else ("SELL" if pct_diff < -threshold else "HOLD")

                # Uncertainty only for tree models, if you want to keep your current routine
                uncertainty = {}
                if self.config["enable_uncertainty"] and model_type not in ["LSTM", "Dense NN"]:
                    intervals = self._calculate_prediction_intervals(model, df.copy(), model_type)
                    if intervals:
                        uncertainty = {
                            "confidence_intervals": intervals["intervals"],
                            "std_dev": intervals["std_dev"]
                        }

                predictions[model_type] = {
                    "predicted_price": float(pred_price),
                    "predicted_return": float(pred_return),
                    "pct_diff": float(pct_diff),
                    "signal": signal,
                    "last_price": float(last_price),
                    **uncertainty
                }
            except Exception as e:
                logger.warning(f"Prediction failed for {model_type}: {e}")
                predictions[model_type] = {"error": str(e)}

        # Aggregate ensemble from the individual predictions only (don’t load a separate file here)
        ensemble = self._generate_ensemble_predictions(predictions)
        if ensemble:
            predictions["Ensemble"] = ensemble

        return predictions


    def _generate_ensemble_predictions(self, individual_preds: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate ensemble predictions from individual model predictions"""
        valid_preds = []
        last_prices = []
        
        # Collect valid predictions
        for model_type, pred in individual_preds.items():
            if model_type != "Ensemble" and "predicted_price" in pred:
                valid_preds.append(pred["predicted_price"])
                last_prices.append(pred["last_price"])
        
        if len(valid_preds) < 2:
            return None
            
        # Calculate ensemble statistics
        mean_pred = np.mean(valid_preds)
        median_pred = np.median(valid_preds)
        last_price = last_prices[0]  # All should be the same
        
        # Weighted average (more weight to recent models)
        weights = np.linspace(0.5, 1.5, len(valid_preds))
        weighted_pred = np.average(valid_preds, weights=weights)
        
        # Calculate metrics
        pct_diff_mean = (mean_pred - last_price) / last_price * 100.0
        pct_diff_median = (median_pred - last_price) / last_price * 100.0
        pct_diff_weighted = (weighted_pred - last_price) / last_price * 100.0
        
        threshold = self.config["prediction_threshold_pct"]
        
        return {
            "mean_prediction": float(mean_pred),
            "median_prediction": float(median_pred),
            "weighted_prediction": float(weighted_pred),
            "pct_diff_mean": float(pct_diff_mean),
            "pct_diff_median": float(pct_diff_median),
            "pct_diff_weighted": float(pct_diff_weighted),
            "signal_mean": "BUY" if pct_diff_mean > threshold else ("SELL" if pct_diff_mean < -threshold else "HOLD"),
            "signal_median": "BUY" if pct_diff_median > threshold else ("SELL" if pct_diff_median < -threshold else "HOLD"),
            "signal_weighted": "BUY" if pct_diff_weighted > threshold else ("SELL" if pct_diff_weighted < -threshold else "HOLD"),
            "model_count": len(valid_preds),
            "std_dev": float(np.std(valid_preds)) if len(valid_preds) > 1 else 0.0
        }

    # ---------- backtesting ----------
    def walk_forward_backtest(self, df: pd.DataFrame, model: Any, model_type: str,
                            feature_scaler: Any, target_scaler: Any,
                            feature_cols: List[str], ticker: str) -> pd.DataFrame:
        
        # Get the EXACT features used during training
        if hasattr(self, 'feature_list') and self.feature_list is not None:
            # Use the feature list stored during training
            training_features = [col for col in self.feature_list if col in df.columns]
        else:
            # Fallback to metadata or original feature_cols
            meta = self.load_meta(ticker)
            training_features = meta.get('feature_columns', feature_cols)
            training_features = [f for f in training_features if f in df.columns]
        
        logger.info(f"Model type: {model_type}")
        logger.info(f"Using {len(training_features)} features from training")
        logger.info(f"Features: {training_features}")
        
        window = self.config["window_size"]
        threshold_pct = self.config["prediction_threshold_pct"]

        if len(df) < window + 10:
            logger.warning("Not enough data for backtest")
            return pd.DataFrame()

        # Use ONLY training features
        features = df[training_features].values

        # --- target should be returns, not prices ---
        ret = df['target_return'].values.reshape(-1, 1)

        # Transform with PRE-FITTED scalers (trained on features / returns)
        scaled_features = feature_scaler.transform(features)
        scaled_ret = target_scaler.transform(ret)  # returns, not prices

        # Build sequences aligned with the return at index i
        X, y = [], []
        for i in range(window, len(scaled_ret)):
            X.append(scaled_features[i-window:i])
            y.append(scaled_ret[i])
        X = np.array(X)
        y = np.array(y)

        # Shape per model
        Xm = X if model_type == "LSTM" else X.reshape(X.shape[0], -1)

        # Predict scaled returns
        preds_scaled = (model.predict(Xm, verbose=0).reshape(-1)
                        if model_type in ["LSTM", "Dense NN"]
                        else model.predict(Xm).reshape(-1))

        # Inverse-scale to returns
        pred_ret = target_scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()

        # Map predicted return to next-day price using today's price
        true_prices = df['adj_close'].values[window:]
        prev_prices = df['adj_close'].values[window-1:-1]
        pred_prices = prev_prices * (1.0 + pred_ret)

        pct_diffs = (pred_prices - true_prices) / true_prices * 100.0
        signals = np.where(pct_diffs > threshold_pct, "BUY",
                np.where(pct_diffs < -threshold_pct, "SELL", "HOLD"))


        # Portfolio sim
        cash = self.config["initial_capital"]
        positions = 0
        portfolio = []
        trade_history = []

        for i in range(len(true_prices)):
            price = true_prices[i]
            sig = signals[i]

            if sig == "BUY" and positions == 0:
                positions = (cash * 0.95) // price
                if positions > 0:
                    cash -= positions * price
                    trade_history.append({
                        'date': df.index[window + i],
                        'type': 'BUY',
                        'price': price,
                        'shares': positions
                    })
            elif sig == "SELL" and positions > 0:
                cash += positions * price
                trade_history.append({
                    'date': df.index[window + i],
                    'type': 'SELL',
                    'price': price,
                    'shares': positions
                })
                positions = 0

            portfolio.append(cash + positions * price)
              
        if positions > 0:
            cash += positions * true_prices[-1]
            positions = 0
        portfolio[-1] = cash  # overwrite last value with liquidated value


        result = pd.DataFrame({
            "Date": df.index[window:],
            "TruePrice": true_prices,
            "PredictedPrice": pred_prices,
            "Signal": signals,
            "PortfolioValue": portfolio,
            "DailyReturn": pd.Series(portfolio).pct_change()
        })


        # Save trades
        trades_df = pd.DataFrame(trade_history)
        if not trades_df.empty:
            trades_df.to_csv(
                os.path.join(self.config["model_dir"], f"{ticker}_{model_type}_trades.csv"),
                index=False
            )

        return result



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
