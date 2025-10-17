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
import tensorflow as tf
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, RandomForestClassifier, VotingClassifier
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
from feature_selector import FeatureSelector
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

try:
    import env
except ImportError:
    env = None

# ---- CONFIG ----
MODEL_DIR_DEFAULT = "saved_models"
DATA_DIR_DEFAULT = "data"
MODEL_META_FILENAME = "model_meta.json"
RETRAIN_DAYS = 7  # autoretrain interval
ENSEMBLE_METHODS = ['mean', 'median', 'weighted']
MODELLING_TYPE = os.environ.get("MODELLING_TYPE") or getattr(env, "MODELLING_TYPE", "regression")  # "regression" or "classification"
print(f"MODELLING_TYPE set to: {MODELLING_TYPE}")

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
            "prediction_threshold_pct": 0.5,
            "min_trade_confidence": 0.40,
            "initial_capital": 10000,
            "min_data_points": 200,  # Increased minimum data requirement
            "retrain_days": RETRAIN_DAYS,
            "verbose": False,
            "enable_uncertainty": True,  # Enable prediction intervals
            "n_cv_folds": 5,  # For time series cross-validation
            "feature_lookback": 10,  # How many past steps to use for lag feature creation
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
    def _check_data_quality(self, X: np.ndarray, y: np.ndarray, name: str = "Data"):
        """Check for data quality issues that cause model collapse"""
        print(f"\n{'='*60}")
        print(f"DATA QUALITY CHECK: {name}")
        print(f"{'='*60}")
        
        # Check for NaN/Inf
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        
        if nan_count > 0 or inf_count > 0:
            logger.error(f"❌ Found {nan_count} NaN and {inf_count} Inf values in features!")
            return False
        
        # Check feature variance
        feature_std = X.reshape(X.shape[0], -1).std(axis=0)
        zero_var_features = (feature_std < 1e-10).sum()
        
        if zero_var_features > 0:
            logger.warning(f"⚠️  Found {zero_var_features} zero-variance features")
        
        # Check target variance
        target_std = y.std()
        target_mean = y.mean()
        
        print(f"📊 Target statistics:")
        print(f"   Mean: {target_mean:.6f}")
        print(f"   Std:  {target_std:.6f}")
        print(f"   Min:  {y.min():.6f}")
        print(f"   Max:  {y.max():.6f}")
        
        if target_std < 0.001:
            logger.error(f"❌ Target variance too low: {target_std:.6f}")
            logger.error("   This will cause model collapse to mean prediction!")
            return False
        
        # Check for outliers
        z_scores = np.abs((y - target_mean) / (target_std + 1e-10))
        outliers = (z_scores > 5).sum()
        
        if outliers > len(y) * 0.1:  # More than 10% outliers
            logger.warning(f"⚠️  Found {outliers} extreme outliers ({outliers/len(y)*100:.1f}%)")
        
        print(f"✅ Data quality check passed")
        print(f"{'='*60}\n")
        return True
    
    def load_raw(self, ticker: str) -> Optional[pd.DataFrame]:
        """Enhanced data loader with robust split handling"""
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
            
            # Sort by date
            df = df.sort_index()
            
            # 🔥 CORRECTED split detection and adjustment
            df_adjusted = self._auto_detect_and_adjust_splits(df, ticker)
            
            # 🔥 Ensure sensible price range
            df_final = self._get_sensible_data_range(df_adjusted, ticker)
            
            return df_final
            
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None

    def _auto_detect_and_adjust_splits(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Automatically detect and adjust for stock splits using price pattern analysis.
        """
        print(f"🔄 Auto-detecting splits for {ticker}...")
        
        # Detect potential splits
        splits = self._detect_splits_from_price_patterns(df)
        
        if not splits:
            print(f"✅ No splits detected for {ticker}")
            return df
        
        print(f"🔧 Adjusting for {len(splits)} detected splits...")
        
        # Adjust prices for splits
        df_adjusted = self._apply_split_adjustments(df, splits)
        
        # Verify adjustment worked
        self._verify_split_adjustment(df_adjusted, splits, ticker)
        
        return df_adjusted

    def _detect_splits_from_price_patterns(self, df: pd.DataFrame) -> List[dict]:
        """
        Detect splits by analyzing price patterns and volume spikes.
        """
        splits = []
        
        if len(df) < 10:
            return splits
        
        # Calculate daily returns and volume changes
        returns = df['adj_close'].pct_change()
        
        # Look for patterns that indicate splits
        for i in range(1, len(returns) - 1):
            current_return = returns.iloc[i]
            prev_return = returns.iloc[i-1]
            next_return = returns.iloc[i+1]
            
            # Split pattern: Large negative return followed by normal trading
            is_large_drop = current_return < -0.3  # More than 30% drop
            is_normal_after = abs(next_return) < 0.1  # Normal trading after
            is_not_crash = prev_return > -0.1  # Not part of a crash
            
            if is_large_drop and is_normal_after and is_not_crash:
                date = returns.index[i]
                prev_price = df['adj_close'].iloc[i-1]
                current_price = df['adj_close'].iloc[i]
                
                # Calculate actual ratio
                actual_ratio = prev_price / current_price
                
                # Check if ratio is close to common split ratios
                common_ratios = [2.0, 3.0, 4.0, 5.0, 10.0, 20.0]
                closest_ratio = min(common_ratios, key=lambda x: abs(x - actual_ratio))
                
                # Only accept if reasonably close to common ratio
                if abs(actual_ratio - closest_ratio) / closest_ratio < 0.2:  # Within 20%
                    splits.append({
                        'date': date,
                        'detected_ratio': actual_ratio,
                        'applied_ratio': closest_ratio,
                        'prev_price': prev_price,
                        'split_price': current_price,
                        'return_pct': current_return * 100
                    })
                    
                    print(f"   📈 Detected {closest_ratio:.1f}:1 split on {date.date()}")
                    print(f"      Price: ${prev_price:.2f} → ${current_price:.2f}")
                    print(f"      Return: {current_return:.1%}")
        
        return splits

    def _apply_split_adjustments(self, df: pd.DataFrame, splits: List[dict]) -> pd.DataFrame:
        """
        CORRECTED: Apply split adjustments to historical prices.
        """
        if not splits:
            return df
        
        df_adjusted = df.copy()
        
        # Sort splits chronologically and apply in reverse order
        splits_sorted = sorted(splits, key=lambda x: x['date'])
        
        cumulative_ratio = 1.0
        adjustment_log = []
        
        for split in reversed(splits_sorted):
            split_date = split['date']
            ratio = split['applied_ratio']
            
            # 🔥 CRITICAL FIX: We need to DIVIDE pre-split prices by the ratio
            # For a 20:1 split, pre-split prices should be divided by 20
            # This brings them down to post-split levels
            
            # Adjust all prices BEFORE the split date
            mask = df_adjusted.index < split_date
            
            # Adjust price columns - DIVIDE by ratio to scale down pre-split prices
            price_cols = ['adj_close', 'open', 'high', 'low']
            for col in price_cols:
                if col in df_adjusted.columns:
                    df_adjusted.loc[mask, col] = df_adjusted.loc[mask, col] / ratio
            
            cumulative_ratio /= ratio  # Track the cumulative division
            adjustment_log.append({
                'date': split_date,
                'ratio': ratio,
                'cumulative_ratio': cumulative_ratio
            })
            
            print(f"   🔧 Adjusted prices before {split_date.date()} by 1:{ratio:.1f}")
        
        print(f"   📊 Total cumulative adjustment: 1:{1/cumulative_ratio:.1f}")
        
        return df_adjusted

    def _verify_split_adjustment(self, df_adjusted: pd.DataFrame, splits: List[dict], ticker: str):
        """
        Enhanced verification with better diagnostics.
        """
        print(f"🔍 Verifying split adjustment for {ticker}...")
        
        # Check price range makes sense
        min_price = df_adjusted['adj_close'].min()
        max_price = df_adjusted['adj_close'].max()
        
        print(f"   📊 Price range: ${min_price:.2f} - ${max_price:.2f}")
        
        # Check each split location
        for split in splits:
            split_date = split['date']
            
            if split_date in df_adjusted.index:
                split_idx = df_adjusted.index.get_loc(split_date)
                if split_idx > 0 and split_idx < len(df_adjusted) - 1:
                    day_before = df_adjusted['adj_close'].iloc[split_idx - 1]
                    day_of = df_adjusted['adj_close'].iloc[split_idx]
                    day_after = df_adjusted['adj_close'].iloc[split_idx + 1]
                    
                    return_day_of = (day_of / day_before - 1) * 100
                    return_day_after = (day_after / day_of - 1) * 100
                    
                    print(f"   📅 {split_date.date()}:")
                    print(f"      Day before: ${day_before:.2f}")
                    print(f"      Day of: ${day_of:.2f} ({return_day_of:+.1f}%)")
                    print(f"      Day after: ${day_after:.2f} ({return_day_after:+.1f}%)")
                    
                    # After adjustment, split day should show normal trading
                    if abs(return_day_of) < 10 and abs(return_day_after) < 10:
                        print(f"      ✅ Normal trading pattern - adjustment successful")
                    else:
                        print(f"      ⚠️  Unusual pattern - may need manual review")
        
        # Check overall data quality
        returns = df_adjusted['adj_close'].pct_change().dropna()
        extreme_moves = returns[abs(returns) > 0.2]  # Moves > 20%
        
        print(f"   📈 Data quality:")
        print(f"      Records: {len(df_adjusted)}")
        print(f"      Average daily return: {returns.mean() * 100:.3f}%")
        print(f"      Daily return std: {returns.std() * 100:.3f}%")
        print(f"      Extreme moves (>20%): {len(extreme_moves)}")
        
        if len(extreme_moves) > 0:
            print(f"      ⚠️  Found {len(extreme_moves)} extreme price moves")
            for date, move in extreme_moves.head(3).items():
                print(f"         {date.date()}: {move * 100:+.1f}%")

    def _get_sensible_data_range(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Ensure the price range is sensible after adjustment.
        If prices are still unreasonable, use only post-split data.
        """
        max_reasonable_price = 5000  # No stock should be above $5000 after adjustment
        min_reasonable_price = 1     # No stock should be below $1
        
        if (df['adj_close'] > max_reasonable_price).any() or (df['adj_close'] < min_reasonable_price).any():
            print(f"⚠️  Price range still unreasonable after adjustment")
            print(f"   Using only data after last detected split...")
            
            # Find the most recent split
            splits = self._detect_splits_from_price_patterns(df)
            if splits:
                last_split = max(splits, key=lambda x: x['date'])
                last_split_date = last_split['date']
                
                # Use data starting 30 days after last split to avoid adjustment artifacts
                start_date = last_split_date + pd.Timedelta(days=30)
                sensible_data = df[df.index >= start_date].copy()
                
                print(f"   Using data from {start_date.date()} onwards")
                print(f"   Records: {len(sensible_data)}")
                print(f"   New price range: ${sensible_data['adj_close'].min():.2f} - ${sensible_data['adj_close'].max():.2f}")
                
                return sensible_data
        
        return df

    def analyze_features(self, ticker: str, save_plot: bool = True):
        """
        Analyze and visualize feature importance
        """
        df = self.prepare_features(ticker)
        if df is None:
            print("Could not prepare features")
            return
        
        feature_cols = [c for c in df.columns 
                    if c not in ['target_price', 'target_return', 'target_direction']]
        
        X = df[feature_cols].values
        y = df['target_return'].values
        
        print(f"\n{'='*60}")
        print(f"FEATURE ANALYSIS: {ticker}")
        print(f"{'='*60}\n")
        
        # Test multiple selection methods
        methods = ['mutual_info', 'f_test', 'importance', 'correlation']
        
        for method in methods:
            print(f"\n--- Method: {method.upper()} ---")
            selector = FeatureSelector(method=method, n_features=20)
            selector.fit(X, y, feature_names=feature_cols)
            
            print(f"Top 10 features:")
            print(selector.feature_scores_.head(10).to_string(index=False))
            
            if save_plot:
                selector.plot_feature_scores(top_n=20)
                plt.savefig(f"{ticker}_{method}_features.png", dpi=150, bbox_inches='tight')
                plt.close()
        
        # Combined method
        print(f"\n--- Method: COMBINED (Ensemble) ---")
        selector = FeatureSelector(method='combined', n_features=30)
        selector.fit(X, y, feature_names=feature_cols)
        print(f"Top 15 features:")
        print(selector.feature_scores_.head(15).to_string(index=False))
        
        if save_plot:
            selector.plot_feature_scores(top_n=30)
            plt.savefig(f"{ticker}_combined_features.png", dpi=150, bbox_inches='tight')
            plt.close()

            
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced technical indicators with better predictive power"""
        if df.empty:
            return df
        
        try:
            # Price-based core features
            df['log_ret'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
            df['volatility_5'] = df['log_ret'].rolling(5).std()
            df['volatility_20'] = df['log_ret'].rolling(20).std()
            
            # 🔥 CRITICAL: Add momentum features that actually predict returns
            # Price momentum (strong predictor)
            for period in [1, 2, 3, 5, 10]:
                df[f'momentum_{period}'] = df['adj_close'].pct_change(period)
            
            # Mean reversion features
            df['price_vs_sma_10'] = (df['adj_close'] / df['adj_close'].rolling(10).mean()) - 1
            df['price_vs_sma_20'] = (df['adj_close'] / df['adj_close'].rolling(20).mean()) - 1
            df['price_vs_sma_50'] = (df['adj_close'] / df['adj_close'].rolling(50).mean()) - 1
            
            # Volatility features
            df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
            df['high_low_range'] = (df['high'] - df['low']) / df['adj_close']
            
            # 🔥 NEW: Add predictive technical indicators
            # RSI (momentum)
            df['RSI_14'] = talib.RSI(df['adj_close'], timeperiod=14)
            df['RSI_7'] = talib.RSI(df['adj_close'], timeperiod=7)
            
            # MACD (trend)
            macd, macd_signal, macd_hist = talib.MACD(df['adj_close'])
            df['MACD'] = macd
            df['MACD_signal'] = macd_signal
            df['MACD_hist'] = macd_hist
            
            # Bollinger Bands (volatility)
            upper, middle, lower = talib.BBANDS(df['adj_close'])
            df['BB_upper'] = upper
            df['BB_lower'] = lower
            df['BB_position'] = (df['adj_close'] - lower) / (upper - lower)
            
            # Stochastic (momentum)
            stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['adj_close'])
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # ATR (volatility)
            df['ATR'] = talib.ATR(df['high'], df['low'], df['adj_close'])
            
            # 🔥 NEW: Market regime features
            df['trend_strength'] = talib.ADX(df['high'], df['low'], df['adj_close'])
            df['market_regime'] = np.where(df['trend_strength'] > 25, 1, 0)
            
            # Volume features (if available)
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                df['price_volume_trend'] = df['log_ret'] * df['volume_ratio']
            
            # 🔥 NEW: Lagged returns (autocorrelation)
            for lag in [1, 2, 3, 5]:
                df[f'return_lag_{lag}'] = df['log_ret'].shift(lag)
            
            # 🔥 NEW: Rolling statistics
            for window in [5, 10]:
                df[f'return_skew_{window}'] = df['log_ret'].rolling(window).skew()
                df[f'return_kurt_{window}'] = df['log_ret'].rolling(window).kurt()
            
            # 🔥 NEW: Support/resistance levels
            df['resistance_20'] = df['high'].rolling(20).max()
            df['support_20'] = df['low'].rolling(20).min()
            df['price_vs_resistance'] = (df['adj_close'] - df['resistance_20']) / df['adj_close']
            df['price_vs_support'] = (df['adj_close'] - df['support_20']) / df['adj_close']
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error in enhanced technical indicators: {e}")
            raise
    

    def create_better_targets(self, df: pd.DataFrame, lookforward_days: int = 5, threshold: float = 0.015) -> pd.DataFrame:
        """
        Create better trading signals with clearer separation
        """
        # Calculate forward returns
        df['target_return'] = df['adj_close'].shift(-lookforward_days) / df['adj_close'] - 1
        
        # Remove extreme outliers using IQR
        Q1 = df['target_return'].quantile(0.05)
        Q3 = df['target_return'].quantile(0.95)
        IQR = Q3 - Q1
        df = df[(df['target_return'] >= Q1 - 1.5 * IQR) & (df['target_return'] <= Q3 + 1.5 * IQR)]
        
        # Create binary target with threshold (not just >0)
        df['target_direction'] = np.where(df['target_return'] > threshold, 1, 
                                        np.where(df['target_return'] < -threshold, 0, -1))  # -1 for hold
        
        # Also keep the return for regression models if needed
        df['target_return'] = df['target_return']
        
        # Remove hold samples for cleaner classification
        df = df[df['target_direction'] != -1]
        
        print(f"🔍 Improved target distribution:")
        print(f"   Class 0 (DOWN): {(df['target_direction'] == 0).sum()}")
        print(f"   Class 1 (UP): {(df['target_direction'] == 1).sum()}")
        print(f"   Balance ratio: {(df['target_direction'] == 1).sum() / len(df):.3f}")
        
        return df

        
    def prepare_features(self, ticker: str, lookahead_days: int = 1) -> Optional[pd.DataFrame]:
        """
        FIXED: Apply split adjustment BEFORE calculating technical indicators
        """
        try:
            print(f"DEBUG: Starting feature preparation for {ticker}")
            
            # Load and SPLIT-ADJUST raw data FIRST
            df = self.load_raw(ticker)
            
            if df is None or len(df) < self.config["min_data_points"]:
                logger.warning(f"{ticker}: Not enough raw data or failed to load")
                return None

            print(f"DEBUG: Split-adjusted data loaded - shape: {df.shape}")
            print(f"DEBUG: Adjusted price range: ${df['adj_close'].min():.2f} - ${df['adj_close'].max():.2f}")

            # Drop initial NA values
            df = df.dropna()

            # 🔥 CRITICAL: NOW add technical indicators to SPLIT-ADJUSTED data
            df = self._add_technical_indicators(df)
            
            # Drop NA values created by technical indicators
            df = df.dropna()

            # Check if we still have enough data
            if len(df) < self.config["min_data_points"]:
                logger.warning(f"{ticker}: Only {len(df)} rows after feature engineering")
                return None

            # Create targets
            if MODELLING_TYPE == "classification":
                df = self.create_better_targets(df, lookforward_days=5, threshold=0.015)
            else:
                df['target_price'] = df['adj_close'].shift(-lookahead_days)
                df['target_return'] = (df['target_price'] / df['adj_close']) - 1.0
                df['target_direction'] = np.where(df['target_return'] > 0, 1, 0)

            # Remove any rows with NaN targets
            df = df.dropna(subset=['target_return'])
            
            # Additional outlier removal
            returns = df['target_return']
            median = returns.median()
            mad = (returns - median).abs().median()
            df = df[(returns >= median - 5*mad) & (returns <= median + 5*mad)]

            # Final data check
            if len(df) < self.config["min_data_points"]:
                logger.warning(f"{ticker}: Only {len(df)} rows after outlier removal")
                return None

            print(f"DEBUG: Successfully prepared features for {ticker}, final shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error preparing features for {ticker}: {e}")
            import traceback
            traceback.print_exc()
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

    def _create_lstm_classifier(self, input_shape: Tuple[int, int]) -> Sequential:
        """Fixed LSTM classifier without focal loss"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
            BatchNormalization(),
            LSTM(32, dropout=0.2, recurrent_dropout=0.1),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']  # Fixed: use string names, not function references
        )
        return model
    
    def _create_dense_classifier(self, input_shape: Tuple[int]) -> Sequential:
            """Dense NN for binary classification"""
            model = Sequential([
                Input(shape=(input_shape[0],)),
                Dense(256, activation='relu'),
                Dropout(0.4),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')  # 🔥 SIGMOID for binary classification
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model  
    
    def _create_dense_model(self, input_shape: Tuple[int]) -> Sequential:
        """FIXED Dense NN with proper architecture"""
        model = Sequential([
            Input(shape=(input_shape[0],)),
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')  # Linear for regression
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
        return model

    def _create_ensemble_model(self, base_models: List[Any]) -> VotingRegressor:
        """Create weighted ensemble of models"""
        estimators = [(f'model_{i}', model) for i, model in enumerate(base_models)]
        return VotingRegressor(estimators=estimators, weights=np.linspace(0.5, 1.5, len(base_models)))

    # ---------- model training ----------
    def train_all_models(self, ticker: str, force: bool = False) -> Dict[str, Any]:
        """
        Enhanced implementation with proper train/validation/backtest split
        and separate metrics for training and backtesting.
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
            #df['target_return'] = (df['target_price'] / df['adj_close']) - 1.0
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

        # 3. Data Validation Debugging - MOVED AFTER feature_cols IS DEFINED
        print(f"DEBUG: Target return stats - Min: {df['target_return'].min():.6f}, "
            f"Max: {df['target_return'].max():.6f}, Mean: {df['target_return'].mean():.6f}, "
            f"Std: {df['target_return'].std():.6f}")

        # Check for any constant columns that might cause issues
        constant_cols = []
        for col in feature_cols:  # Now feature_cols is defined
            if df[col].std() < 1e-10:  # Near constant
                constant_cols.append(col)
        if constant_cols:
            print(f"DEBUG: Warning - Constant/near-constant columns: {constant_cols}")
            # Remove constant columns
            feature_cols = [c for c in feature_cols if c not in constant_cols]

        # Save feature information
        meta = self.load_meta(ticker)
        meta['feature_columns'] = feature_cols
        meta['n_features'] = len(feature_cols)
        
        # 4. STRATIFIED TIME-BASED DATA SPLIT
        total_samples = len(df)
        
        # Split: 70% training, 15% validation, 15% backtesting (unseen)
        train_size = int(total_samples * 0.7)
        val_size = int(total_samples * 0.15)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        backtest_df = df.iloc[train_size + val_size:]
        
        print(f"DEBUG: Data split - Train: {len(train_df)}, Val: {len(val_df)}, Backtest: {len(backtest_df)}")
        
        meta['data_split'] = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'backtest_size': len(backtest_df),
            'split_date_backtest': backtest_df.index[0] if len(backtest_df) > 0 else None
        }
        self.save_meta(ticker, meta)
        
        # ============================================================================
        # 🔥 ADD FEATURE SELECTION HERE - BEFORE SCALING 🔥
        # ============================================================================
        
        print(f"\n{'='*60}")
        print(f"FEATURE SELECTION")
        print(f"{'='*60}")
        print(f"Original features: {len(feature_cols)}")
        
        # Extract training data for feature selection
        X_train_raw = train_df[feature_cols].values
        y_train_raw = train_df['target_return'].values
        
        # Initialize feature selector with combined method
        n_features_to_select = min(50, len(feature_cols) // 2)  # Select top 50 or half
        
        try:
            selector = FeatureSelector(
                method='combined',  # Use ensemble of methods
                n_features=n_features_to_select
            )
            
            # Fit selector on training data only (no data leakage!)
            selector.fit(X_train_raw, y_train_raw, feature_names=feature_cols)
            
            # Update feature_cols to selected features
            feature_cols = selector.selected_features_
            
            print(f"Selected features: {len(feature_cols)}")
            print(f"\nTop 15 selected features:")
            for i, feat in enumerate(feature_cols[:15], 1):
                print(f"  {i:2d}. {feat}")
            
            # Save selector for prediction
            selector_path = os.path.join(self.config["model_dir"], f"{ticker}_feature_selector.joblib")
            joblib.dump(selector, selector_path)
            
            # Update metadata
            meta['selected_features'] = feature_cols
            meta['n_selected_features'] = len(feature_cols)
            meta['feature_selection_method'] = 'combined'
            meta['original_n_features'] = len(train_df.columns) - 3  # Exclude targets
            self.save_meta(ticker, meta)
            
            print(f"Feature selector saved to: {selector_path}")
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")
            # Keep original feature_cols if selection fails
            pass
        
        print(f"{'='*60}\n")
        
        # ============================================================================
        # END FEATURE SELECTION
        # ============================================================================
        
        # 5. Scale features and targets using TRAINING data only (with selected features)
        feature_scaler = StandardScaler()

        print(f"DEBUG: Raw target stats:")
        print(f"  Min: {train_df['target_return'].min():.6f}, Max: {train_df['target_return'].max():.6f}")
        print(f"  Mean: {train_df['target_return'].mean():.6f}, Std: {train_df['target_return'].std():.6f}")

        # Scale features (important for neural networks)
        X_train_scaled = feature_scaler.fit_transform(train_df[feature_cols].values)
        X_val_scaled = feature_scaler.transform(val_df[feature_cols].values)
        X_backtest_scaled = feature_scaler.transform(backtest_df[feature_cols].values)

        y_train = train_df['target_direction'].values.reshape(-1, 1)
        y_val = val_df['target_direction'].values.reshape(-1, 1)
        y_backtest = backtest_df['target_direction'].values.reshape(-1, 1)

        
        # Save scalers (trained on training data only)
        joblib.dump(feature_scaler, os.path.join(self.config["model_dir"], f"{ticker}_feature_scaler.joblib"))
    
        # 6. Create sequences for time-series models
        window = self.config["window_size"]
        
        # Training sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train, window)
        
        # Validation sequences (for early stopping)
        X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val, window)
        
        # Backtest sequences (for final evaluation on unseen data)
        X_backtest_seq, y_backtest_seq = self.create_sequences(X_backtest_scaled, y_backtest, window)

        print(f"DEBUG: Sequence shapes - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Backtest: {X_backtest_seq.shape}")
          # Critical data quality check
        if not self._check_data_quality(X_train_seq, y_train_seq, "Training Data"):
            return {"error": "Data quality check failed - aborting training"}
        
        if not self._check_data_quality(X_val_seq, y_val_seq, "Validation Data"):
            return {"error": "Validation data quality check failed"}
        # 7. Model Training with Enhanced Error Handling
        #regression models
        if MODELLING_TYPE == "regression":
            models = {
            "LSTM": self._train_lstm,
            "Dense NN": self._train_dnn,  
            "Random Forest": self._train_rf,
            "XGBoost": self._train_xgb,
            }
        
        else:
            models = {
                "LSTM":  self._train_lstm_classifier,  
                "Dense NN":  self._train_dense_classifier,  
                "Random Forest":  self._train_rf_classifier,
                "XGBoost":  self._train_xgb_classifier,
                }
    


        results = {}
        trained_models = {}
        
        for name, train_func in models.items():
            try:
                model_path = self._model_path(ticker, name)
                
                if not force and os.path.exists(model_path):
                    model = self.load_model(name, ticker)
                    results[name] = {"status": "loaded"}
                else:
                    logger.info(f"Training {name} for {ticker}")
                    # Train on training data, validate on validation data
                    model = train_func(X_train_seq, y_train_seq, X_val_seq, y_val_seq, len(feature_cols))
                    self._save_model(model, name, ticker)
                    results[name] = {"status": "trained"}
                    
                trained_models[name] = model
            except Exception as e:
                logger.error(f"Training failed for {name}: {e}")
                results[name] = {"status": f"error: {str(e)}"}
                trained_models[name] = None  # Ensure it's set to None on failure

        # 8. Ensemble Training (only if we have at least 2 models)
        try:
            sk_models = {
                name: m for name, m in trained_models.items()
                if name in ("Random Forest", "XGBoost") and "Regressor" in str(type(m))
            }

            if len(sk_models) >= 2:
                if MODELLING_TYPE == "regression":
                    ensemble = VotingRegressor(list(sk_models.items()))
                else:
                    ensemble = VotingClassifier(
                    estimators=list(sk_models.items()),
                        voting='soft',  # Use probability voting
                        n_jobs=-1
                    )

                X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
                ensemble.fit(X_train_flat, y_train_seq.ravel())
                joblib.dump(ensemble, self._model_path(ticker, "Ensemble"))
                trained_models["Ensemble"] = ensemble
                results["Ensemble"] = {"status": "trained"}
            else:
                print(f"DEBUG: Not enough successful models for ensemble. Available: {list(sk_models.keys())}")
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            results["Ensemble"] = {"status": f"error: {str(e)}"}

        # 9. TRAINING METRICS (on validation data - seen during training)
        training_metrics = {}
        for name, model in trained_models.items():
            if model is None:
                continue
            try:
                # Evaluate on validation data (seen during training)
                if MODELLING_TYPE == "regression":
                    val_metrics = self._evaluate_on_validation(
                        model, name, X_val_seq, y_val_seq, 
                    )
                else:
                    val_metrics = self._evaluate_on_validation_classification(
                        model, name, X_val_seq, y_val_seq, 
                    )
                training_metrics[name] = val_metrics
            except Exception as e:
                logger.error(f"Training metrics failed for {name}: {e}")
                training_metrics[name] = {"error": str(e)}

        # 10. BACKTESTING METRICS (on completely unseen data)
        backtest_results = {}
        for name, model in trained_models.items():
            if model is None:
                continue
            try:
                # Backtest on completely unseen backtest data
                if MODELLING_TYPE == "regression":
                    bt_df = self.walk_forward_backtest(
                        backtest_df.copy(), model, name,
                        feature_scaler, 
                        feature_cols, ticker
                    )
                else:
                    bt_df = self.walk_forward_backtest_classification(
                        backtest_df.copy(), model, name,
                        feature_scaler, 
                        feature_cols, ticker
                    )

                # Ensure consistent structure
                backtest_results[name] = {
                    "walk_forward": bt_df,
                    "cv_metrics": {},
                    "prediction_intervals": {}
                }
            except Exception as e:
                logger.error(f"Backtest failed for {name}: {e}")
                backtest_results[name] = {
                    "walk_forward": pd.DataFrame(),
                    "cv_metrics": {},
                    "prediction_intervals": {}
                }

        # 11. Calculate and save both training and backtest metrics
        if MODELLING_TYPE == "regression":
            training_metrics_df = self._calculate_training_metrics(training_metrics)
            backtest_metrics_df = self._calculate_advanced_metrics(backtest_results)
        else:
            backtest_metrics_df = self._calculate_classification_metrics(backtest_results)
            training_metrics_df = self._calculate_training_metrics_classification(training_metrics)
            #print(f"DEBUG: Training metrics:\n{training_metrics_df}")

        # Save both metrics files
        training_metrics_df.to_csv(os.path.join(self.config["model_dir"], f"{ticker}_training_metrics.csv"), index=False)
        backtest_metrics_df.to_csv(os.path.join(self.config["model_dir"], f"{ticker}_backtest_metrics.csv"), index=False)

        # 12. Save comprehensive training results
        self._save_training_results(
            ticker, 
            results, 
            backtest_results,
            training_metrics_df,
            backtest_metrics_df
        )

        return {
            "training_results": results,
            "training_metrics": training_metrics_df.to_dict(orient='records'),
            "backtest_metrics": backtest_metrics_df.to_dict(orient='records'),
            "backtest": backtest_results,
            "best_model": self._select_best_model(backtest_results),  # Select based on backtest performance
            "data_split_info": meta['data_split']
        }




    def _train_xgb_classifier(self, X_train, y_train, X_val, y_val, n_features):
        """FIXED XGBoost with proper calibration"""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Calculate scale_pos_weight for imbalance
        n_positive = np.sum(y_train == 1)
        n_negative = np.sum(y_train == 0)
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1
        
        model = XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,  # Lower learning rate
            max_depth=4,         # Reduced depth to prevent overfitting
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,       # Increased regularization
            reg_lambda=1.0,      # Increased regularization
            scale_pos_weight=scale_pos_weight,  # Handle class imbalance
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        
        # Add early stopping
        model.fit(
            X_train_flat, y_train.ravel(),
            eval_set=[(X_val.reshape(X_val.shape[0], -1), y_val.ravel())],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Check calibration
        train_proba = model.predict_proba(X_train_flat[:10])[:, 1]
        print(f"🚨 XGBoost calibrated probabilities: {train_proba}")
        print(f"🚨 XGBoost actual directions: {y_train[:10].flatten()}")
        
        return model

    def _train_rf_classifier(self, X_train, y_train, X_val, y_val, n_features):
        """Enhanced Random Forest with better recall"""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Use class weights and focus on recall
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.ravel())
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        model = RandomForestClassifier(
            n_estimators=200,  # More trees
            max_depth=15,      # Deeper trees
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight=class_weight_dict,  # Use class weights
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_flat, y_train.ravel())
        return model

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


    def _train_lstm_classifier(self, X_train, y_train, X_val, y_val, n_features):
        """FIXED LSTM training with error handling"""
        try:
            # Clear any previous session
            tf.keras.backend.clear_session()
            
            # Create model with simple architecture
            model = Sequential([
                Input(shape=(self.config["window_size"], n_features)),
                LSTM(32, return_sequences=False, dropout=0.2),
                BatchNormalization(),
                Dense(16, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            
            # Compile with basic metrics only
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Calculate class weights
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.ravel())
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
            
            print(f"🚨 LSTM Class weights: {class_weight_dict}")
            
            # Train with callbacks
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=1,
                class_weight=class_weight_dict,
                callbacks=[
                    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
                    ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6)
                ]
            )
            
            # Debug predictions
            train_pred = model.predict(X_train[:5], verbose=0).flatten()
            print(f"🚨 LSTM sample predictions: {train_pred}")
            print(f"🚨 LSTM actual values: {y_train[:5].flatten()}")
            
            return model
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            # Return a simple fallback model
            return self._create_fallback_model(X_train.shape[1:])

    def _train_dense_classifier(self, X_train, y_train, X_val, y_val, n_features):
        """FIXED Dense NN training with class weights"""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        model = self._create_dense_classifier((X_train_flat.shape[1],))
        
        # Add class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.ravel())
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        print(f"🚨 TRAINING DENSE NN CLASSIFIER")
        print(f"   Input shape: {X_train_flat.shape}")
        print(f"   Target distribution: {np.unique(y_train, return_counts=True)}")
        print(f"   Class weights: {class_weight_dict}")
        
        history = model.fit(
            X_train_flat, y_train,
            validation_data=(X_val_flat, y_val),
            epochs=100,
            batch_size=32,
            verbose=1,
            class_weight=class_weight_dict,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
                ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7)
            ]
        )
        
        return model
        
        
    def _train_dnn(self, X_train, y_train, X_val, y_val, n_features):
        """FIXED Dense NN training"""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        model = self._create_dense_model((X_train_flat.shape[1],))
        
        print(f"DEBUG: Training Dense NN with input shape: {X_train_flat.shape}")

        history = model.fit(
            X_train_flat, y_train,
            validation_data=(X_val_flat, y_val),
            epochs=100,
            batch_size=32,
            verbose=1,
            callbacks=[
                EarlyStopping(patience=20, restore_best_weights=True, min_delta=0.00001),
                ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
            ]
        )
        
        sample_pred = model.predict(X_val_flat[:5], verbose=0).flatten()
        print(f"DEBUG: Dense NN validation predictions: {sample_pred}")
        print(f"DEBUG: Dense NN validation actuals: {y_val[:5].flatten()}")
        
        return model
    
    def _train_xgb(self, X_train, y_train, X_val, y_val, n_features):
        """Fixed XGBoost to prevent prediction collapse"""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # 🔥 CRITICAL: Use MUCH more aggressive parameters
        model = XGBRegressor(
            n_estimators=100,  # Reduced from 200
            learning_rate=0.1,  # Increased from 0.05
            max_depth=3,       # Reduced from 6 to prevent overfitting
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,    # Reduced regularization
            reg_lambda=0.01,   # Reduced regularization  
            gamma=0,           # No minimum loss reduction
            random_state=42,
            n_jobs=-1
        )
        
        # Add early stopping
        model.fit(
            X_train_flat, y_train.ravel(),
            eval_set=[(X_train_flat, y_train.ravel())],
            verbose=False
        )
        
        # Check if model learned anything
        sample_pred = model.predict(X_train_flat[:5])
        print(f"DEBUG: XGBoost sample predictions: {sample_pred}")
        print(f"DEBUG: XGBoost actual returns: {y_train[:5].flatten()}")
        
        return model

    def _train_rf(self, X_train, y_train, X_val, y_val, n_features):
        """Fixed Random Forest"""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        model = RandomForestRegressor(
            n_estimators=50,    # Reduced complexity
            max_depth=5,        # Reduced depth
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.3,   # Use fewer features per tree
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_flat, y_train.ravel())
        
        # Check feature importances
        importances = model.feature_importances_
        top_features = np.argsort(importances)[-5:][::-1]
        print(f"DEBUG: RF Top 5 feature importances: {importances[top_features]}")
        
        return model

    def _train_lstm(self, X_train, y_train, X_val, y_val, n_features):
        """Fixed LSTM with gradient clipping"""
        # Simpler architecture
        model = Sequential([
            Input(shape=(self.config["window_size"], n_features)),
            LSTM(32, return_sequences=False, dropout=0.2),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='linear')
        ])
        
        # Add gradient clipping
        optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Reduced epochs
            batch_size=32,
            verbose=1,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        
        return model
    def _calculate_training_metrics_classification(self, training_metrics: Dict[str, Dict]) -> pd.DataFrame:
        """FIXED: Calculate training metrics without regression metrics"""
        metrics_list = []
        
        for model_name, metrics in training_metrics.items():
            if "error" in metrics:
                metrics_list.append({
                    "Model": model_name,
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1": np.nan,
                    "ROC_AUC": np.nan,
                    "Samples": 0,
                    "Error": metrics["error"]
                })
            else:
                # Only include classification metrics
                accuracy = metrics.get("accuracy", np.nan)
                precision = metrics.get("precision", np.nan)
                recall = metrics.get("recall", np.nan)
                f1 = metrics.get("f1", np.nan)
                roc_auc = metrics.get("roc_auc", np.nan)
                samples = metrics.get("samples", 0)
                
                metrics_list.append({
                    "Model": model_name,
                    "Accuracy": float(accuracy) if not np.isnan(accuracy) else np.nan,
                    "Precision": float(precision) if not np.isnan(precision) else np.nan,
                    "Recall": float(recall) if not np.isnan(recall) else np.nan,
                    "F1": float(f1) if not np.isnan(f1) else np.nan,
                    "ROC_AUC": float(roc_auc) if not np.isnan(roc_auc) else np.nan,
                    "Samples": int(samples) if samples else 0,
                    "Error": None
                })
        
        return pd.DataFrame(metrics_list)
    def _calculate_training_metrics(self, training_metrics: Dict[str, Dict], 
                               ) -> pd.DataFrame:
        """
        Calculate training/validation metrics in a structured DataFrame
        """
        metrics_list = []
        
        for model_name, metrics in training_metrics.items():
            if "error" in metrics:
                metrics_list.append({
                    "Model": model_name,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "DirectionAccuracy": np.nan,
                    "Samples": 0,
                    "Error": metrics["error"]
                })
            else:
                metrics_list.append({
                    "Model": model_name,
                    "MAE": metrics.get("MAE", np.nan),
                    "RMSE": metrics.get("RMSE", np.nan),
                    "R2": metrics.get("R2", np.nan),
                    "DirectionAccuracy": metrics.get("DirectionAccuracy", np.nan),
                    "Samples": metrics.get("Samples", 0),
                    "Error": None
                })
        
        return pd.DataFrame(metrics_list)
        
    def _evaluate_on_validation(self, model: Any, model_type: str, 
                          X_val: np.ndarray, y_val: np.ndarray, 
                          ) -> Dict[str, float]:
        """
        Evaluate model on validation data with proper input handling
        """
        try:
            # Make predictions with proper input shaping
            if model_type == "LSTM":
                y_pred= model.predict(X_val, verbose=0).flatten()
            elif model_type == "Dense NN":
                # Flatten sequences for Dense NN
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                y_pred = model.predict(X_val_flat, verbose=0).flatten()
            else:
                # Tree-based models
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                y_pred = model.predict(X_val_flat)
            
            # Inverse transform to get actual returns
            y_true = (y_val.reshape(-1, 1)).flatten()
            y_pred = (y_pred.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            direction_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
            
            return {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "DirectionAccuracy": direction_acc,
                "Samples": len(y_true)
            }
            
        except Exception as e:
            logger.error(f"Validation evaluation failed for {model_type}: {e}")
            return {"error": str(e)}
            
   

    def _save_training_results(self, ticker: str, training_results: Dict[str, Any],
                       backtest_results: Dict[str, Any],
                       training_metrics_df: pd.DataFrame = None,
                       backtest_metrics_df: pd.DataFrame = None) -> None:
        """
        Enhanced version to save both training and backtest results with separate metrics.
        Normalize backtest_results, compute metrics, save backtests & metrics CSVs, save meta.
        Produces consistent files for visualization and JSON-safe metadata.
        """

        model_dir = self.config.get("model_dir", "saved_models")
        os.makedirs(model_dir, exist_ok=True)

        normalized = {}

        # --- Normalize input ---
        for m, val in backtest_results.items():
            if isinstance(val, pd.DataFrame):
                normalized[m] = val
            elif isinstance(val, dict) and isinstance(val.get("walk_forward"), pd.DataFrame):
                normalized[m] = val.get("walk_forward")
            else:
                # Fallback: empty placeholder
                normalized[m] = pd.DataFrame(columns=[
                    "Date", "TruePrice", "PredictedPrice", "Signal",
                    "PortfolioValue", "PredictedReturn", "y_true", "y_pred"
                ])

        # --- Save each model's backtest CSV ---
        for model_name, df in normalized.items():
            try:
                if not df.empty:
                    df_sorted = df.sort_values("Date")
                    csv_path = os.path.join(model_dir, f"{ticker}_{model_name}_backtest.csv")
                    df_sorted.to_csv(csv_path, index=False)
                    logger.info(f"[{ticker}] Saved backtest for {model_name} -> {csv_path}")
                else:
                    logger.warning(f"[{ticker}] No backtest data for {model_name}")
            except Exception as e:
                logger.error(f"[{ticker}] Failed to save backtest for {model_name}: {e}")

        # --- Handle Training Metrics (validation performance) ---
        if training_metrics_df is None:
            # Fallback: create empty training metrics if not provided
            training_metrics_df = pd.DataFrame(columns=[
                "Model", "MAE", "RMSE", "R2", "DirectionAccuracy", "Samples", "Error"
            ])
        
        training_metrics_path = os.path.join(model_dir, f"{ticker}_training_metrics.csv")
        try:
            training_metrics_df.to_csv(training_metrics_path, index=False)
            logger.info(f"[{ticker}] Saved training metrics -> {training_metrics_path}")
        except Exception as e:
            logger.error(f"[{ticker}] Failed to write training metrics CSV: {e}")

        # --- Handle Backtest Metrics (unseen data performance) ---
        if backtest_metrics_df is None:
            # Fallback: calculate backtest metrics from normalized results
            try:
                wrapped_results = {m: {"walk_forward": df} for m, df in normalized.items()}
                if MODELLING_TYPE == "classification":
                    backtest_metrics_df = self._calculate_classification_metrics(wrapped_results)
                else:    
                    backtest_metrics_df = self._calculate_advanced_metrics(wrapped_results)
            except Exception as e:
                logger.error(f"[{ticker}] Backtest metric calculation failed: {e}")
                backtest_metrics_df = pd.DataFrame(columns=[
                    "Model", "MAE", "RMSE", "R2", "DirectionAcc",
                    "FinalPortfolio", "Sharpe", "Volatility"
                ])

        backtest_metrics_path = os.path.join(model_dir, f"{ticker}_backtest_metrics.csv")
        try:
            backtest_metrics_df.to_csv(backtest_metrics_path, index=False)
            logger.info(f"[{ticker}] Saved backtest metrics -> {backtest_metrics_path}")
        except Exception as e:
            logger.error(f"[{ticker}] Failed to write backtest metrics CSV: {e}")

        # --- Prepare JSON-safe metrics ---
        training_metrics_clean = training_metrics_df.where(pd.notnull(training_metrics_df), None)
        backtest_metrics_clean = backtest_metrics_df.where(pd.notnull(backtest_metrics_df), None)

        # --- Save comprehensive metadata ---
        meta = {
            "last_trained": datetime.utcnow().isoformat(),
            "model_paths": {m: self._model_path(ticker, m) for m in training_results.keys()},
            "training_metrics_file": training_metrics_path,
            "backtest_metrics_file": backtest_metrics_path,
            "training_metrics": training_metrics_clean.to_dict(orient="records"),
            "backtest_metrics": backtest_metrics_clean.to_dict(orient="records"),
            "best_model": self._select_best_model(normalized),
            "data_split_info": self.load_meta(ticker).get('data_split', {})
        }

        try:
            self.save_meta(ticker, meta)
            logger.info(f"[{ticker}] ✅ Saved comprehensive training results")
            logger.info(f"[{ticker}]   - Training metrics: {training_metrics_path}")
            logger.info(f"[{ticker}]   - Backtest metrics: {backtest_metrics_path}")
            logger.info(f"[{ticker}]   - Best model: {meta['best_model']}")
        except Exception as e:
            logger.error(f"[{ticker}] Could not save meta: {e}")

        # --- Print summary for quick review ---
        self._print_training_summary(ticker, training_metrics_df, backtest_metrics_df, meta['best_model'])

    def _print_training_summary(self, ticker: str, training_metrics: pd.DataFrame, 
                        backtest_metrics: pd.DataFrame, best_model: str):
        """
        FIXED: Show proper classification metrics in summary
        """
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY: {ticker}")
        print(f"{'='*60}")
        
        # Training Metrics Summary
        print(f"\n📊 TRAINING METRICS (Validation Data):")
        print(f"{'-'*50}")
        if not training_metrics.empty:
            for _, row in training_metrics.iterrows():
                model = row['Model']
                accuracy = row.get('Accuracy', np.nan)  # Fixed column name
                f1 = row.get('F1', np.nan)  # Fixed column name
                print(f"  {model:20} | Accuracy: {accuracy:6.3f} | F1: {f1:6.3f}")
        else:
            print("  No training metrics available")
        
        # Backtest Metrics Summary  
        print(f"\n🎯 BACKTEST METRICS (Unseen Data):")
        print(f"{'-'*50}")
        if not backtest_metrics.empty:
            for _, row in backtest_metrics.iterrows():
                model = row['Model']
                accuracy = row.get('Accuracy', np.nan)
                f1 = row.get('F1', np.nan)
                final_port = row.get('FinalPortfolio', np.nan)
                print(f"  {model:20} | Accuracy: {accuracy:6.3f} | F1: {f1:6.3f} | Portfolio: ${final_port:,.0f}")
        else:
            print("  No backtest metrics available")
        
        # Best Model
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"{'='*60}\n")




    def _calculate_advanced_metrics(self, backtest_results: dict, ) -> pd.DataFrame:
        """
        Fixed version to properly handle backtest results structure.
        Computes performance metrics for all model backtest results.
        Handles:
            - Regression (MAE, RMSE, R2, Direction Accuracy)
            - Classification (Direction Accuracy) 
            - Portfolio metrics (Final value, Sharpe, Volatility)
        """
        import numpy as np
        import pandas as pd
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = []
        print("DEBUG: Backtest results keys:", list(backtest_results.keys()))
        
        for model_name, result in backtest_results.items():
            print(f"\nDEBUG: --- Calculating metrics for model: {model_name} ---")
            print(f"DEBUG: Result type: {type(result)}")

            # Handle different result structures
            df = None
            if isinstance(result, pd.DataFrame):
                # Result is already a DataFrame
                df = result
                print(f"DEBUG: Using result directly as DataFrame")
            elif isinstance(result, dict):
                # Result is a dict with walk_forward key
                df = result.get("walk_forward")
                print(f"DEBUG: Extracted walk_forward from dict, type: {type(df)}")
            else:
                print(f"DEBUG: Unknown result type: {type(result)}")

            # Check if we have valid data
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                print(f"DEBUG: Empty or invalid DataFrame for {model_name}")
                metrics.append({
                    "Model": model_name,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "DirectionAcc": np.nan,
                    "FinalPortfolio": np.nan,
                    "Sharpe": np.nan,
                    "Volatility": np.nan
                })
                continue

            print(f"DEBUG: DataFrame shape: {df.shape}")
            print(f"DEBUG: Columns in DataFrame: {df.columns.tolist()}")
            print(f"DEBUG: First 3 rows:\n{df.head(3)}")

            # --- Regression / NN metrics ---
            mae = rmse = r2 = dir_acc = np.nan
            
            # Check if we have prediction data (y_true and y_pred)
            if "y_true" in df.columns and "y_pred" in df.columns:
                # Create a clean copy without NaN values
                clean_df = df.dropna(subset=["y_true", "y_pred"]).copy()
                
                if not clean_df.empty:
                    y_true = clean_df["y_true"].values
                    y_pred = clean_df["y_pred"].values
                    
                    print(f"DEBUG: Clean samples: {len(y_true)}")
                    print(f"DEBUG: Sample y_true: {y_true[:5]}")
                    print(f"DEBUG: Sample y_pred: {y_pred[:5]}")

                    # Calculate regression metrics
                    try:
                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        r2 = r2_score(y_true, y_pred)
                        
                        # Calculate direction accuracy
                        dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
                        
                        print(f"DEBUG: Regression metrics - MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.6f}, DirectionAcc={dir_acc:.6f}")
                    except Exception as e:
                        print(f"DEBUG: Error calculating regression metrics: {e}")
                else:
                    print(f"DEBUG: No valid samples after dropping NaN from y_true/y_pred")
            else:
                print(f"DEBUG: y_true/y_pred columns not found")
                
                # Try to calculate direction accuracy from signals if available
                if "Signal" in df.columns and "y_true" in df.columns:
                    clean_df = df.dropna(subset=["Signal", "y_true"]).copy()
                    if not clean_df.empty:
                        y_true_dir = np.sign(clean_df["y_true"].values)
                        # Convert signals to numerical values
                        signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
                        signal_dir = np.array([signal_map.get(s, 0) for s in clean_df["Signal"].values])
                        dir_acc = np.mean(signal_dir == y_true_dir)
                        print(f"DEBUG: DirectionAcc from signals: {dir_acc:.6f}")
                    else:
                        print(f"DEBUG: No valid samples for signal-based direction accuracy")

            # --- Portfolio metrics ---
            final_port = sharpe = vol = np.nan
            if "PortfolioValue" in df.columns:
                portfolio_series = df["PortfolioValue"].dropna()
                if not portfolio_series.empty:
                    final_port = portfolio_series.iloc[-1]
                    
                    # Calculate portfolio returns and Sharpe ratio
                    portfolio_returns = portfolio_series.pct_change().dropna()
                    if len(portfolio_returns) > 1:
                        mean_ret = portfolio_returns.mean()
                        vol = portfolio_returns.std()
                        sharpe = mean_ret / vol * np.sqrt(252) if vol > 0 and not np.isnan(vol) else np.nan
                        print(f"DEBUG: Portfolio - Final: {final_port:.2f}, Sharpe: {sharpe:.6f}, Vol: {vol:.6f}")
                    else:
                        print(f"DEBUG: Not enough portfolio returns to calculate Sharpe")
                else:
                    print(f"DEBUG: PortfolioValue column exists but all values are NaN")
            else:
                print(f"DEBUG: PortfolioValue column not found")

            # Special handling for classifier models
            if model_name == "Random Forest" and "classifier" in str(type(result)).lower():
                print(f"DEBUG: Random Forest classifier detected - focusing on direction accuracy")
                # Keep direction accuracy but mark regression metrics as NaN
                mae = rmse = r2 = np.nan

            metrics.append({
                "Model": model_name,
                "MAE": float(mae) if not np.isnan(mae) else np.nan,
                "RMSE": float(rmse) if not np.isnan(rmse) else np.nan,
                "R2": float(r2) if not np.isnan(r2) else np.nan,
                "DirectionAcc": float(dir_acc) if not np.isnan(dir_acc) else np.nan,
                "FinalPortfolio": float(final_port) if not np.isnan(final_port) else np.nan,
                "Sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
                "Volatility": float(vol) if not np.isnan(vol) else np.nan
            })

        # Create and sort metrics DataFrame
        metrics_df = pd.DataFrame(metrics)
        if not metrics_df.empty and "DirectionAcc" in metrics_df.columns:
            metrics_df = metrics_df.sort_values(by="DirectionAcc", ascending=False)
        
        print(f"\nDEBUG: Final metrics DataFrame shape: {metrics_df.shape}")
        print(f"DEBUG: Final metrics DataFrame:\n{metrics_df}")
        
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

    def prepare_features_for_tomorrow_pred(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Prepare features for predicting the next day's price.
        """
        try:
            # --- Load raw data ---
            df = self.load_raw(ticker)
          
            if df is None or len(df) < self.config["min_data_points"]:
                logger.warning(f"{ticker}: Not enough raw data")
                return None

            # Store the last row BEFORE adding technical indicators
            last_row_index = df.index[-1]
            last_row_data = df.iloc[[-1]].copy()
          
            # --- Add technical indicators ---
            df = self._add_technical_indicators(df)
            
            
            # --- RESTORE THE MISSING LAST ROW ---
            if len(df) > 0 and df.index[-1] != last_row_index:
                
                # Simply append the original last row
                df = pd.concat([df, last_row_data], ignore_index=False)
                
                # --- CRITICAL: Fill NaN values for technical indicators in the last row ---
                if len(df) > 1:
                    last_idx = df.index[-1]
                    prev_idx = df.index[-2]  # Previous row
                    
                    # Fill each technical indicator column with previous value
                    for col in df.columns:
                        if col not in ['open', 'high', 'low', 'adj_close']:  # Skip raw price columns
                            if pd.isna(df.loc[last_idx, col]) and not pd.isna(df.loc[prev_idx, col]):
                                df.loc[last_idx, col] = df.loc[prev_idx, col]
                                #print(f"DEBUG: Filled {col} with previous value: {df.loc[prev_idx, col]}")
                    
                    # Special handling for log_ret (calculate it properly)
                    if 'log_ret' in df.columns and pd.isna(df.loc[last_idx, 'log_ret']):
                        if not pd.isna(df.loc[prev_idx, 'adj_close']) and not pd.isna(df.loc[last_idx, 'adj_close']):
                            df.loc[last_idx, 'log_ret'] = np.log(df.loc[last_idx, 'adj_close'] / df.loc[prev_idx, 'adj_close'])
                            #print(f"DEBUG: Calculated log_ret: {df.loc[last_idx, 'log_ret']}")


            # --- Create features ---
            df['target_price'] = df['adj_close']
            df['target_return'] = df['target_price'] / df['adj_close'] - 1.0
            df['target_direction'] = np.where(df['target_return'] > 0, 1, 0)

            # --- Fill last row targets ---
            if len(df) > 0:
                df.iloc[-1, df.columns.get_loc('target_price')] = df['adj_close'].iloc[-1]
                df.iloc[-1, df.columns.get_loc('target_return')] = 0.0
                df.iloc[-1, df.columns.get_loc('target_direction')] = 0

            # --- FINAL SANITY CHECK: Ensure no NaN values in the last row ---
            if len(df) > 0:
                last_row_nans = df.iloc[-1].isna().sum()
                if last_row_nans > 0:
                    #print(f"DEBUG: WARNING - Last row has {last_row_nans} NaN values. Filling with zeros.")
                    df.iloc[-1] = df.iloc[-1].fillna(0)
                
            return df

        except Exception as e:
            logger.error(f"Error preparing features for tomorrow: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _add_technical_indicators_to_single_row(self, row: pd.DataFrame, recent_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to a single row using recent data for context.
        This avoids the dropping behavior in the main technical indicators method.
        """
        try:
            # Combine recent data with the new row
            combined_data = pd.concat([recent_data, row], ignore_index=False)
            
            # Add indicators to the combined data
            combined_data = self._add_technical_indicators(combined_data)
            
            # Return only the last row (our target row)
            return combined_data.iloc[[-1]]
        except Exception as e:
            logger.error(f"Error adding indicators to single row: {e}")
            return row


    # ---------- prediction ----------
    def predict_tomorrow(self, ticker: str) -> Dict[str, Any]:
        """FIXED prediction with proper threshold handling"""
        df = self.prepare_features_for_tomorrow_pred(ticker)
        if df is None:
            return {"error": "Could not prepare data"}
        
        meta = self.load_meta(ticker)
        selected_features = meta.get('selected_features', None)
        
        if selected_features:
            feature_cols = [f for f in selected_features if f in df.columns]
        else:
            selector_path = os.path.join(self.config["model_dir"], f"{ticker}_feature_selector.joblib")
            if os.path.exists(selector_path):
                selector = joblib.load(selector_path)
                feature_cols = [f for f in selector.selected_features_ if f in df.columns]
            else:
                feature_cols = [c for c in df.columns if c not in ['target_price', 'target_return', 'target_direction']]
        
        features = df[feature_cols].values
        
        fs_path = os.path.join(self.config["model_dir"], f"{ticker}_feature_scaler.joblib")
        feature_scaler = joblib.load(fs_path)
        
        if features.shape[1] != feature_scaler.n_features_in_:
            return {"error": f"Feature mismatch"}
        
        scaled_features = feature_scaler.transform(features)
        window = self.config["window_size"]
        
        X_seq = []
        for i in range(window, len(scaled_features)+1):
            X_seq.append(scaled_features[i-window:i])
        X_seq = np.array(X_seq)
        
        if X_seq.size == 0:
            return {"error": "Not enough data for prediction"}
        
        last_window = X_seq[-1:]
        predictions = {}
        last_price = float(df["adj_close"].iloc[-1])
        threshold = self.config["prediction_threshold_pct"] / 100.0  # Convert to decimal
        
        model_types = ["LSTM", "Dense NN", "Random Forest", "XGBoost"]
        
        for model_type in model_types:
            try:
                model = self.load_model(model_type, ticker)
                
                if model_type == "LSTM":
                    input_data = last_window
                elif model_type == "Dense NN":
                    input_data = last_window.reshape(1, -1)
                else:
                    input_data = last_window.reshape(1, -1)
                
                if model_type in ["LSTM", "Dense NN"]:
                    pred = model.predict(input_data, verbose=0).flatten()[0]
                else:
                    pred = model.predict(input_data).flatten()[0]
                
                pred_return = float(pred)
                pred_price = last_price * (1 + pred_return)
                pct_diff = pred_return * 100  # As percentage
                
                signal = "BUY" if pred_return > threshold else ("SELL" if pred_return < -threshold else "HOLD")
                
                predictions[model_type] = {
                    "predicted_price": float(pred_price),
                    "predicted_return": pred_return,
                    "pct_diff": pct_diff,
                    "signal": signal
                }
                
            except Exception as e:
                predictions[model_type] = {"error": str(e)}
        
        ensemble_pred = self._generate_ensemble_predictions(predictions, last_price)
        if ensemble_pred:
            predictions["Ensemble"] = ensemble_pred
        
        return predictions

    def _generate_ensemble_predictions(self, model_preds: Dict[str, Dict[str, Any]], 
                                      last_price: float) -> Dict[str, Any]:
        """FIXED ensemble with proper signal generation"""
        signals = []
        returns = []
        
        for model, pred in model_preds.items():
            if "error" in pred:
                continue
            
            pred_return = pred.get("predicted_return")
            signal = pred.get("signal")
            
            if pred_return is not None and signal and signal != "HOLD":
                signals.append(signal)
                returns.append(pred_return)
        
        if not signals:
            return {
                "signal": "HOLD",
                "predicted_price": last_price,
                "predicted_return": 0.0,
                "confidence": 0.0,
                "pct_diff": 0.0
            }
        
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        total = len(signals)
        
        avg_return = np.mean(returns)
        avg_price = last_price * (1 + avg_return)
        
        if buy_count > sell_count:
            ensemble_signal = "BUY"
            confidence = buy_count / total
        elif sell_count > buy_count:
            ensemble_signal = "SELL"
            confidence = sell_count / total
        else:
            ensemble_signal = "HOLD"
            confidence = 0.0
        
        return {
            "signal": ensemble_signal,
            "predicted_price": float(avg_price),
            "predicted_return": float(avg_return),
            "confidence": confidence,
            "pct_diff": avg_return * 100
        }

    # ---------- backtesting ----------
    def walk_forward_backtest_classification(self, df: pd.DataFrame, model: Any, model_type: str,
                                   feature_scaler: Any, feature_cols: List[str], 
                                   ticker: str) -> pd.DataFrame:
        """
        FIXED: Backtesting for classification models
        """
        try:
            meta = self.load_meta(ticker)
            selected_features = meta.get('selected_features', feature_cols)
            
            if selected_features:
                available_features = [f for f in selected_features if f in df.columns]
                training_features = available_features
            else:
                training_features = [f for f in feature_cols if f in df.columns]
                
            print(f"🚨 {model_type} BACKTEST - {len(training_features)} features")

            window = self.config.get("window_size", 30)
            confidence_threshold = 0.55  # Increased threshold for better signals

            if len(df) < window + 10:
                return pd.DataFrame()

            # Prepare features
            features = df[training_features].values
            true_directions = df['target_direction'].values.reshape(-1, 1)
            prices = df['adj_close'].values
            dates = df.index

            scaled_features = feature_scaler.transform(features)
            
            # Create sequences
            X_seq = []
            y_seq = []
            prev_prices = []
            true_prices = []
            
            max_prediction_point = len(scaled_features) - 1
            
            for i in range(window, max_prediction_point):
                X_seq.append(scaled_features[i-window:i])
                y_seq.append(true_directions[i])
                prev_prices.append(prices[i-1])
                true_prices.append(prices[i])
                
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq).reshape(-1, 1)
            prev_prices = np.array(prev_prices)
            true_prices = np.array(true_prices)
            
            n = X_seq.shape[0]
            
            print(f"   Sequences: {n}, True directions - UP: {(y_seq == 1).sum()}, DOWN: {(y_seq == 0).sum()}")

            # Get probabilities
            probabilities = self._make_predictions_classification(model, model_type, X_seq)

            if probabilities is None or len(probabilities) != n:
                print(f"   ❌ Prediction failed for {model_type}")
                return pd.DataFrame()

            print(f"   Probabilities range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
            print(f"   >0.7: {(probabilities > 0.7).sum()}, <0.3: {(probabilities < 0.3).sum()}")

            # Convert probabilities to signals with asymmetric thresholds
            signals = np.where(
                probabilities > confidence_threshold, "BUY",
                np.where(probabilities < (1 - confidence_threshold), "SELL", "HOLD")
            )
            
            # Convert probabilities to binary predictions for metrics
            pred_directions = (probabilities > 0.5).astype(int)

            # Portfolio simulation
            portfolio_values = self._simulate_portfolio_classification(
            signals, true_prices, prev_prices, probabilities  # Added probabilities parameter
                )
            # Build result DataFrame
            df_result = pd.DataFrame({
                "Date": dates[window:window+n],
                "TruePrice": true_prices,
                "TrueDirection": y_seq.ravel(),
                "PredictedProbability": probabilities,
                "PredictedDirection": pred_directions,
                "Signal": signals,
                "PortfolioValue": portfolio_values,
                "y_true": y_seq.ravel(),
                "y_pred": pred_directions
            })

            return df_result

        except Exception as e:
            logger.error(f"Backtest failed for {model_type}: {e}")
            return pd.DataFrame()


    def walk_forward_backtest(self, df: pd.DataFrame, model: Any, model_type: str,
                              feature_scaler: Any, feature_cols: List[str], 
                              ticker: str) -> pd.DataFrame:
        """
        FIXED backtesting with proper signal generation and portfolio simulation
        """
        try:
            meta = self.load_meta(ticker)
            selected_features = meta.get('selected_features', feature_cols)
            
            if selected_features:
                available_features = [f for f in selected_features if f in df.columns]
                training_features = available_features
            else:
                training_features = [f for f in feature_cols if f in df.columns]
            
            print(f"\nDEBUG: ========== {model_type} BACKTEST ==========")
            print(f"DEBUG: Using {len(training_features)} features")
            
            window = self.config.get("window_size", 30)
            threshold_pct = self.config.get("prediction_threshold_pct", 0.5) / 100.0  # Convert to decimal
            
            print(f"DEBUG: Signal threshold: {threshold_pct:.4f} ({threshold_pct * 100:.2f}%)")

            if len(df) < window + 10:
                logger.warning("Not enough data for backtest")
                return pd.DataFrame()

            # Prepare features
            features = df[training_features].values
            true_returns = df['target_return'].values.reshape(-1, 1)
            prices = df['adj_close'].values
            dates = df.index

            scaled_features = feature_scaler.transform(features)
            
            # Create sequences
            X_seq = []
            y_seq = []
            prev_prices = []
            true_prices = []
            
            for i in range(window, len(scaled_features)):
                X_seq.append(scaled_features[i-window:i])
                y_seq.append(true_returns[i])
                prev_prices.append(prices[i-1])
                true_prices.append(prices[i])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq).reshape(-1, 1)
            prev_prices = np.array(prev_prices)
            true_prices = np.array(true_prices)
            
            n = X_seq.shape[0]
            
            if n == 0:
                return pd.DataFrame()
            
            print(f"DEBUG: Created {n} sequences")
            print(f"DEBUG: True returns range: [{y_seq.min():.4f}, {y_seq.max():.4f}]")

            # Make predictions
            pred_returns = self._make_predictions(model, model_type, X_seq)
            
            if pred_returns is None or len(pred_returns) != n:
                logger.error(f"Prediction failed for {model_type}")
                return pd.DataFrame()
            
            print(f"DEBUG: Predicted returns range: [{pred_returns.min():.4f}, {pred_returns.max():.4f}]")
            print(f"DEBUG: Predicted returns std: {pred_returns.std():.6f}")
            
            # Calculate predicted prices
            pred_prices = prev_prices * (1.0 + pred_returns)
            
            # FIXED: Generate signals with proper threshold
            signals = np.where(
                pred_returns > threshold_pct, "BUY",
                np.where(pred_returns < -threshold_pct, "SELL", "HOLD")
            )
            
            buy_count = (signals == "BUY").sum()
            sell_count = (signals == "SELL").sum()
            hold_count = (signals == "HOLD").sum()
            
            print(f"DEBUG: Signal distribution:")
            print(f"  BUY:  {buy_count:4d} ({buy_count/n*100:.1f}%)")
            print(f"  SELL: {sell_count:4d} ({sell_count/n*100:.1f}%)")
            print(f"  HOLD: {hold_count:4d} ({hold_count/n*100:.1f}%)")
            
            # FIXED: Portfolio simulation
            portfolio_values = self._simulate_portfolio(signals, true_prices, prev_prices)
            
            # Build result DataFrame
            df_result = pd.DataFrame({
                "Date": dates[window:window+n],
                "TruePrice": true_prices,
                "PredictedPrice": pred_prices,
                "Signal": signals,
                "PortfolioValue": portfolio_values,
                "PredictedReturn": pred_returns,
                "y_true": y_seq.ravel(),
                "y_pred": pred_returns
            })
            
            return df_result

        except Exception as e:
            logger.error(f"Backtest failed for {model_type}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _make_predictions(self, model: Any, model_type: str, X_seq: np.ndarray) -> Optional[np.ndarray]:
        """FIXED prediction function with proper handling"""
        try:
            if model_type == "LSTM":
                preds = model.predict(X_seq, verbose=0)
                return preds.ravel()
                
            elif model_type == "Dense NN":
                X_flat = X_seq.reshape(X_seq.shape[0], -1)
                preds = model.predict(X_flat, verbose=0)
                return preds.ravel()
                
            elif model_type in ["Random Forest", "XGBoost", "Ensemble"]:
                X_flat = X_seq.reshape(X_seq.shape[0], -1)
                preds = model.predict(X_flat)
                return preds.ravel()
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def _make_predictions_classification(self, model: Any, model_type: str, X_seq: np.ndarray) -> Optional[np.ndarray]:
        """FIXED: Handle different model input requirements for classification"""
        try:
            if model_type == "LSTM":
                # LSTM expects 3D input (samples, timesteps, features)
                y_proba = model.predict(X_seq, verbose=0).flatten()
                
            elif model_type == "Dense NN":
                # Dense NN expects 2D input (samples, features) - need to flatten sequences
                X_flat = X_seq.reshape(X_seq.shape[0], -1)
                y_proba = model.predict(X_flat, verbose=0).flatten()
                
            elif model_type in ["Random Forest", "XGBoost"]:
                # Tree models expect 2D input
                X_flat = X_seq.reshape(X_seq.shape[0], -1)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_flat)[:, 1]  # Probability of class 1
                else:
                    y_pred = model.predict(X_flat)
                    y_proba = y_pred.astype(float)  # Fallback to binary
                    
            return y_proba
        
        except Exception as e:
            logger.error(f"Prediction failed for {model_type}: {e}")
            return None
    
    
    def _simulate_portfolio_classification(self, signals: np.ndarray, true_prices: np.ndarray, 
                                     prev_prices: np.ndarray, probabilities: np.ndarray) -> List[float]:
        """
        Enhanced portfolio simulation using prediction confidence
        """
        initial_capital = self.config.get("initial_capital", 10000)
        cash = initial_capital
        shares = 0
        portfolio = []
        
        print(f"   Starting portfolio: ${cash:.2f}")
        
        trade_count = 0
        for i, (signal, current_price, prob) in enumerate(zip(signals, true_prices, probabilities)):
            # Calculate current portfolio value
            current_value = cash + (shares * current_price)
            portfolio.append(current_value)
            
            # Use confidence-based position sizing
            confidence = abs(prob - 0.5) * 2  # Convert to 0-1 scale
            
            # Execute trades with confidence-based sizing
            if signal == "BUY" and shares == 0:
                # Use confidence to determine position size (20-80% of capital)
                position_size = 0.2 + (confidence * 0.6)
                max_investment = cash * position_size
                shares_to_buy = int(max_investment // current_price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    shares += shares_to_buy
                    cash -= cost
                    trade_count += 1
                    if trade_count <= 5:
                        print(f"     Trade {trade_count}: BUY {shares_to_buy} @ ${current_price:.2f} (conf: {confidence:.2f})")
                        
            elif signal == "SELL" and shares > 0:
                # Sell all shares
                cash += shares * current_price
                trade_count += 1
                if trade_count <= 5:
                    print(f"     Trade {trade_count}: SELL {shares} @ ${current_price:.2f}")
                shares = 0
        
        # Close any remaining positions
        if shares > 0:
            cash += shares * true_prices[-1]
            portfolio[-1] = cash
        
        final_return = (portfolio[-1] / initial_capital - 1) * 100
        print(f"   Final: ${portfolio[-1]:.2f} ({final_return:+.2f}%), Trades: {trade_count}")
        
        return portfolio

        
    def _simulate_portfolio(self, signals: np.ndarray, true_prices: np.ndarray, 
                           prev_prices: np.ndarray) -> List[float]:
        """
        FIXED portfolio simulation with realistic trading
        """
        initial_capital = self.config.get("initial_capital", 10000)
        cash = initial_capital
        shares = 0
        portfolio = []
        
        print(f"\nDEBUG: Starting portfolio simulation with ${cash:.2f}")
        
        for i, (signal, price) in enumerate(zip(signals, true_prices)):
            # Calculate current portfolio value
            portfolio_value = cash + (shares * price)
            portfolio.append(portfolio_value)
            
            # Execute trades
            if signal == "BUY" and cash >= price:
                # Buy as many shares as possible
                shares_to_buy = int(cash / price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    shares += shares_to_buy
                    cash -= cost
                    if i < 3:  # Log first 3 trades
                        print(f"  Trade {i}: BUY {shares_to_buy} @ ${price:.2f} = ${cost:.2f}")
                        print(f"    Cash: ${cash:.2f}, Shares: {shares}, Value: ${portfolio_value:.2f}")
                        
            elif signal == "SELL" and shares > 0:
                # Sell all shares
                proceeds = shares * price
                if i < 3:  # Log first 3 trades
                    print(f"  Trade {i}: SELL {shares} @ ${price:.2f} = ${proceeds:.2f}")
                cash += proceeds
                shares = 0
                if i < 3:
                    print(f"    Cash: ${cash:.2f}, Shares: {shares}, Value: ${portfolio_value:.2f}")
        
        # Final portfolio value
        final_value = cash + (shares * true_prices[-1])
        portfolio[-1] = final_value
        
        total_return = (final_value / initial_capital - 1) * 100
        print(f"\nDEBUG: Portfolio simulation complete:")
        print(f"  Initial: ${initial_capital:.2f}")
        print(f"  Final:   ${final_value:.2f}")
        print(f"  Return:  {total_return:+.2f}%")
        
        return portfolio


    def _calculate_classification_metrics(self, results: dict, result_type: str = "backtest") -> pd.DataFrame:
        """
        FIXED: Handle the y_true reference error
        """
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                               f1_score, roc_auc_score, confusion_matrix, classification_report)
        metrics = []
        
        for model_name, result in results.items():
            print(f"🔍 Calculating metrics for: {model_name}")
            
            # Initialize variables to avoid reference errors
            y_true = None
            y_pred = None
            y_proba = None
            
            # Handle different result structures
            if result_type == "backtest":
                if isinstance(result, dict) and "walk_forward" in result:
                    df = result["walk_forward"]
                elif isinstance(result, pd.DataFrame):
                    df = result
                else:
                    df = None
                    
                if df is not None and not df.empty and "y_true" in df.columns and "y_pred" in df.columns:
                    y_true = df["y_true"].values
                    y_pred = df["y_pred"].values
                    y_proba = df["PredictedProbability"].values if "PredictedProbability" in df.columns else None
                    
            else:
                # Training results
                if isinstance(result, dict):
                    y_true = result.get("y_true")
                    y_pred = result.get("y_pred") 
                    y_proba = result.get("y_proba")
            

            # Check if we have valid data
            if y_true is None or y_pred is None or len(y_true) == 0:
                print(f"   ❌ No valid data for {model_name}")
                metrics.append(self._create_empty_metrics(model_name, result_type))
                continue
            
            try:
                # Convert to numpy arrays
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Calculate additional metrics if probabilities available
                roc_auc = np.nan
                if y_proba is not None and len(np.unique(y_true)) > 1:
                    try:
                        roc_auc = roc_auc_score(y_true, y_proba)
                    except:
                        roc_auc = np.nan

                # Calculate confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else [0, 0, 0, 0]
                
                # Calculate class distribution
                actual_up = np.sum(y_true == 1)
                actual_down = np.sum(y_true == 0)
                predicted_up = np.sum(y_pred == 1)
                predicted_down = np.sum(y_pred == 0)


                
                # Calculate profit/loss if available (for backtest)
                final_portfolio = np.nan
                sharpe = np.nan
                if result_type == "backtest" and df is not None and "PortfolioValue" in df.columns:
                    portfolio_values = df["PortfolioValue"].dropna()
                    if len(portfolio_values) > 0:
                        final_portfolio = portfolio_values.iloc[-1]
                        # Calculate Sharpe ratio
                        returns = portfolio_values.pct_change().dropna()
                        if len(returns) > 1 and returns.std() > 0:
                            sharpe = returns.mean() / returns.std() * np.sqrt(252)

                # Create metrics entry
                model_metrics = {
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "ROC_AUC": roc_auc,
                    "True_Positives": tp,
                    "False_Positives": fp,
                    "True_Negatives": tn,
                    "False_Negatives": fn,
                    "Actual_UP": actual_up,
                    "Actual_DOWN": actual_down,
                    "Predicted_UP": predicted_up,
                    "Predicted_DOWN": predicted_down,
                    "Samples": len(y_true),            
                }

                # Add financial metrics for backtest
                if result_type == "backtest":
                    model_metrics.update({
                        "FinalPortfolio": final_portfolio,
                        "Sharpe": sharpe,
                        "Total_Return": ((final_portfolio / self.config.get("initial_capital", 10000)) - 1) * 100 
                                        if not np.isnan(final_portfolio) else np.nan
                    })

                metrics.append(model_metrics)

                # Print detailed report
                print(f"   ✅ Accuracy:  {accuracy:.4f}")
                print(f"   ✅ Precision: {precision:.4f}")
                print(f"   ✅ Recall:    {recall:.4f}")
                print(f"   ✅ F1:        {f1:.4f}")
                if not np.isnan(roc_auc):
                    print(f"   ✅ ROC-AUC:   {roc_auc:.4f}")
                print(f"   📊 Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
                if result_type == "backtest" and not np.isnan(final_portfolio):
                    print(f"   💰 Final Portfolio: ${final_portfolio:,.2f}")

            except Exception as e:
                print(f"   ❌ Error calculating metrics for {model_name}: {e}")
                metrics.append(self._create_empty_metrics(model_name, result_type))

        # Create DataFrame and sort by accuracy
        metrics_df = pd.DataFrame(metrics)
        if not metrics_df.empty and "Accuracy" in metrics_df.columns:
            metrics_df = metrics_df.sort_values("Accuracy", ascending=False)
        
        print(f"\n{'='*60}")
        print(f"📈 {result_type.upper()} METRICS SUMMARY")
        print(f"{'='*60}")
        if not metrics_df.empty:
            print(metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1", "FinalPortfolio" if result_type == "backtest" else "Samples"]].to_string(index=False))
        else:
            print("   No valid metrics calculated")
        
        return metrics_df

    def _create_empty_metrics(self, model_name: str, result_type: str) -> dict:
        """Create empty metrics entry for failed models"""
        base_metrics = {
            "Model": model_name,
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "ROC_AUC": np.nan,
            "True_Positives": 0,
            "False_Positives": 0,
            "True_Negatives": 0,
            "False_Negatives": 0,
            "Actual_UP": 0,
            "Actual_DOWN": 0,
            "Predicted_UP": 0,
            "Predicted_DOWN": 0,
            "Samples": 0
        }
        
        if result_type == "backtest":
            base_metrics.update({
                "FinalPortfolio": np.nan,
                "Sharpe": np.nan,
                "Total_Return": np.nan
            })
        
        return base_metrics
    

    def _evaluate_on_validation_classification(self, model: Any, model_type: str, 
                                         X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        FIXED: Return proper classification metrics for training
        """
        try:
            # Make predictions
            if model_type == "LSTM":
                y_proba = model.predict(X_val, verbose=0).flatten()
            elif model_type == "Dense NN":
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                y_proba = model.predict(X_val_flat, verbose=0).flatten()
            else:
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_val_flat)[:, 1]
                else:
                    y_proba = model.predict(X_val_flat).astype(float)

            y_pred = (y_proba > 0.5).astype(int)
            y_true = y_val.ravel()

            # Calculate classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            roc_auc = np.nan
            if len(np.unique(y_true)) > 1:
                try:
                    roc_auc = roc_auc_score(y_true, y_proba)
                except:
                    pass
            
            
            # FIXED: Return the metrics in a consistent structure
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "samples": len(y_true),
                "y_true": y_true,  # Add these for consistency
                "y_pred": y_pred
            }
            
        except Exception as e:
            logger.error(f"Validation evaluation failed for {model_type}: {e}")
            return {"error": str(e)}

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

    def diagnose_model_issues(self, ticker: str):
        """Diagnose why models are performing poorly"""
        print(f"\n🔍 DIAGNOSING MODEL ISSUES FOR {ticker}")
        print("="*50)
        
        # Load the data
        df = self.prepare_features(ticker)
        if df is None:
            print("❌ Could not load data")
            return
        
        # Check feature correlations
        feature_cols = [c for c in df.columns if c not in ['target_price', 'target_return', 'target_direction']]
        correlations = df[feature_cols + ['target_direction']].corr()['target_direction'].abs().sort_values(ascending=False)
        
        print("Top 10 feature correlations with target:")
        print(correlations.head(10))
        
        # Check if any features have perfect correlation (data leakage)
        high_corr = correlations[correlations > 0.8]
        if len(high_corr) > 1:  # More than just target itself
            print(f"⚠️  Suspicious high correlations: {high_corr.index.tolist()}")
        
        # Check model predictions
        model_types = ["LSTM", "Dense NN", "Random Forest", "XGBoost"]
        for model_type in model_types:
            try:
                model = self.load_model(model_type, ticker)
                print(f"\n📊 {model_type} Analysis:")
                
                # Get sample predictions
                sample_features = df[feature_cols].iloc[:10].values
                if model_type == "LSTM":
                    # Would need sequences for LSTM
                    print("   LSTM requires sequence data")
                else:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(sample_features)
                        print(f"   Prediction probabilities: {proba[:, 1]}")
                    else:
                        preds = model.predict(sample_features)
                        print(f"   Predictions: {preds}")
                        
            except Exception as e:
                print(f"   ❌ Could not analyze {model_type}: {e}")