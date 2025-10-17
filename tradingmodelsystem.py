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
from feature_selector import FeatureSelector
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
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
            "window_size": 60,  
            "prediction_threshold_pct": 0.0001,
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
                
            df_adjusted = self._auto_detect_and_adjust_splits(df, ticker)
            df_final = self._get_sensible_data_range(df_adjusted, ticker)
            return df_final
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None

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
            
            # 1. Rate of change (momentum)
            df['ROC_5'] = talib.ROC(df['adj_close'], timeperiod=5)
            df['ROC_10'] = talib.ROC(df['adj_close'], timeperiod=10)
            
            # 2. Moving average crossovers (strong signal)
            df['SMA_10'] = talib.SMA(df['adj_close'], timeperiod=10)
            df['SMA_50'] = talib.SMA(df['adj_close'], timeperiod=50)
            df['MA_crossover'] = (df['SMA_10'] - df['SMA_50']) / df['adj_close']
            
            # 3. Volatility breakout
            df['vol_breakout'] = (df['volatility'] / df['volatility'].rolling(60).mean()) - 1
            
            # 4. Price distance from moving averages
            df['price_to_sma20'] = (df['adj_close'] - talib.SMA(df['adj_close'], 20)) / df['adj_close']
            df['price_to_sma50'] = (df['adj_close'] - talib.SMA(df['adj_close'], 50)) / df['adj_close']
            
            # 5. RSI momentum
            df['RSI_change'] = df['RSI_14'].diff()
            
            # 6. Volume momentum (if available)
            if 'volume' in df.columns:
                df['volume_momentum'] = df['volume'] / df['volume'].rolling(20).mean()
                df['price_volume_trend'] = df['log_ret'] * df['volume_momentum']
            
            # 7. Intraday volatility
            df['intraday_vol'] = (df['high'] - df['low']) / df['adj_close']
            df['gap'] = (df['open'] - df['adj_close'].shift(1)) / df['adj_close'].shift(1)
            
            # 8. Multi-timeframe returns
            for period in [3, 7, 14, 21]:
                df[f'return_{period}d'] = df['adj_close'].pct_change(period)
            
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
            
            """# Target variables
            df['target_price'] = df['adj_close'].shift(-1)
            df['target_return'] = df['target_price'] / df['adj_close'] - 1.0
            df['target_direction'] = np.where(df['target_return'] > 0, 1, 0)"""
            
            # Market regime
            df['trend_strength'] = talib.ADX(df['high'], df['low'], df['adj_close'], timeperiod=14)
            df['market_regime'] = np.where(df['trend_strength'] > 25, 1, 0)
            
            # Volatility regime
            df['volatility_regime'] = (df['volatility'] > df['volatility'].rolling(60).mean()).astype(int)
            
            # Price momentum
            df['momentum_5'] = df['adj_close'].pct_change(5)
            df['momentum_10'] = df['adj_close'].pct_change(10)
            df['momentum_20'] = df['adj_close'].pct_change(20)
            
            # Volume patterns
            if 'volume' in df.columns:
                df['volume_trend'] = df['volume'] / df['volume'].rolling(20).mean()
                df['price_volume_corr'] = df['log_ret'].rolling(20).corr(df['volume'].pct_change())
            
            # Intraday range
            df['high_low_ratio'] = (df['high'] - df['low']) / df['adj_close']
            
            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'return_mean_{window}'] = df['log_ret'].rolling(window).mean()
                df[f'return_std_{window}'] = df['log_ret'].rolling(window).std()
                df[f'return_skew_{window}'] = df['log_ret'].rolling(window).skew()
            
            """# Targets LAST (avoid leakage)
            df['target_price'] = df['adj_close'].shift(-1)
            df['target_return'] = df['target_price'] / df['adj_close'] - 1.0
            df['target_direction'] = np.where(df['target_return'] > 0, 1, 0)"""
            
            return df.dropna()
        except Exception as e:
            logger.error(f"Error in _add_technical_indicators: {e}")
            raise

    def prepare_features(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            print(f"DEBUG: Starting feature preparation for {ticker}")
            
            # Load raw data
            df = self.load_raw(ticker)
            
            if df is None or len(df) < self.config["min_data_points"]:
                logger.warning(f"{ticker}: Not enough raw data or failed to load")
                return None

            # Drop initial NA values
            df = df.dropna()

            # Add technical indicators (NO TARGETS CREATED HERE)
            df = self._add_technical_indicators(df)
            
            # Drop NA values created by technical indicators
            df = df.dropna()

            # Check if we still have enough data
            if len(df) < self.config["min_data_points"]:
                logger.warning(f"{ticker}: Only {len(df)} rows after feature engineering")
                return None

            # ✅ CREATE TARGETS HERE WITH PROPER CLIPPING
            df['target_price'] = df['adj_close'].shift(-1)
            df['target_return'] = (df['target_price'] / df['adj_close']) - 1.0
            
            # 🔥 CRITICAL: Clip BEFORE any other operations
            df['target_return'] = df['target_return'].clip(-0.15, 0.15)
            
            df['target_direction'] = np.where(df['target_return'] > 0, 1, 0)
            
            # Remove any rows with NaN targets
            df = df.dropna(subset=['target_return'])
            
            # Additional outlier removal (now working on already-clipped data)
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

    def _create_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            LSTM(64, return_sequences=True, dropout=0.2),
            LSTM(32, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae']
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
        target_scaler = StandardScaler()

        # Clip extreme returns before scaling to prevent outliers
        train_returns = train_df['target_return'].values.copy()
        
        # Clip returns to 5 standard deviations to handle outliers
        returns_std = train_returns.std()
        returns_mean = train_returns.mean()
        clip_threshold = 5 * returns_std
        train_returns_clipped = np.clip(train_returns, 
                                    returns_mean - clip_threshold, 
                                    returns_mean + clip_threshold)
        
        print(f"DEBUG: Returns before clipping - Min: {train_returns.min():.6f}, Max: {train_returns.max():.6f}")
        print(f"DEBUG: Returns after clipping - Min: {train_returns_clipped.min():.6f}, Max: {train_returns_clipped.max():.6f}")
        
        # 👉 NOW USE SELECTED FEATURES (feature_cols has been updated above)
        X_train_scaled = feature_scaler.fit_transform(train_df[feature_cols].values)
        y_train_scaled = target_scaler.fit_transform(train_returns_clipped.reshape(-1, 1))
        
        X_val_scaled = feature_scaler.transform(val_df[feature_cols].values)
        y_val_scaled = target_scaler.transform(val_df[['target_return']].values)
        
        X_backtest_scaled = feature_scaler.transform(backtest_df[feature_cols].values)
        y_backtest_scaled = target_scaler.transform(backtest_df[['target_return']].values)
        
        print(f"DEBUG: Scaled target range: [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")

        # Save scalers (trained on training data only)
        joblib.dump(feature_scaler, os.path.join(self.config["model_dir"], f"{ticker}_feature_scaler.joblib"))
        joblib.dump(target_scaler, os.path.join(self.config["model_dir"], f"{ticker}_target_scaler.joblib"))

        # 6. Create sequences for time-series models
        window = self.config["window_size"]
        
        # Training sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled, window)
        
        # Validation sequences (for early stopping)
        X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val_scaled, window)
        
        # Backtest sequences (for final evaluation on unseen data)
        X_backtest_seq, y_backtest_seq = self.create_sequences(X_backtest_scaled, y_backtest_scaled, window)

        print(f"DEBUG: Sequence shapes - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Backtest: {X_backtest_seq.shape}")

        # 7. Model Training with Enhanced Error Handling
        models = {
            "LSTM": self._train_lstm,
            "Dense NN": self._train_dnn,  
            "Random Forest": self._train_rf,
            "XGBoost": self._train_xgb,
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

        # ... rest of your code remains the same ...

        # 8. Ensemble Training (only if we have at least 2 regressors)
        try:
            sk_models = {
                name: m for name, m in trained_models.items()
                if name in ("Random Forest", "XGBoost") and "Regressor" in str(type(m))
            }

            if len(sk_models) >= 2:
                ensemble = VotingRegressor(list(sk_models.items()))
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
                val_metrics = self._evaluate_on_validation(
                    model, name, X_val_seq, y_val_seq, target_scaler
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
                bt_df = self.walk_forward_backtest(
                    backtest_df.copy(), model, name,
                    feature_scaler, target_scaler,
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
        training_metrics_df = self._calculate_training_metrics(training_metrics, target_scaler)
        backtest_metrics_df = self._calculate_advanced_metrics(backtest_results, target_scaler)
        
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

    def _calculate_training_metrics(self, training_metrics: Dict[str, Dict], 
                              target_scaler: Any) -> pd.DataFrame:
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
                          target_scaler: Any) -> Dict[str, float]:
        """
        Evaluate model on validation data with proper input handling
        """
        try:
            # Make predictions with proper input shaping
            if model_type == "LSTM":
                y_pred_scaled = model.predict(X_val, verbose=0).flatten()
            elif model_type == "Dense NN":
                # Flatten sequences for Dense NN
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                y_pred_scaled = model.predict(X_val_flat, verbose=0).flatten()
            else:
                # Tree-based models
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                y_pred_scaled = model.predict(X_val_flat)
            
            # Inverse transform to get actual returns
            y_true = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
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
            
    def _train_rf(self, X_train, y_train, X_val, y_val, n_features):
        """Fit a random forest regression"""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        print(f"DEBUG: RF Training - X_train shape: {X_train_flat.shape}, y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
        
        # Ensure we're using Regressor (NOT Classifier)
        model = RandomForestRegressor(
            n_estimators=100,  # Reduced for faster training
            max_depth=8,       # Reduced to prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_flat, y_train.ravel())
        
        # Check predictions aren't zeros
        sample_pred = model.predict(X_val_flat[:5])
        print(f"DEBUG: RF Sample predictions: {sample_pred}")
        
        return model


    def _train_xgb(self, X_train, y_train, X_val, y_val, n_features):
        """FIXED XGBoost training without early stopping for ensemble compatibility"""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        print(f"DEBUG: XGB Training - X_train shape: {X_train_flat.shape}, y_train range: [{y_train.min():.6f}, {y_train.max():.6f}]")

        model = XGBRegressor(
            n_estimators=100,  # Reduced number
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
            # Removed early_stopping_rounds for ensemble compatibility
        )
        
        # Simple fit without early stopping
        model.fit(X_train_flat, y_train.ravel())
        
        # Check predictions
        sample_pred = model.predict(X_val_flat[:5])
        print(f"DEBUG: XGB Sample predictions: {sample_pred}")
        
        return model

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
        Print a clean summary of training results for quick review.
        """
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY: {ticker}")
        print(f"{'='*60}")
        
        # Training Metrics Summary
        print(f"\n📊 TRAINING METRICS (Validation Data):")
        print(f"{'-'*40}")
        if not training_metrics.empty:
            for _, row in training_metrics.iterrows():
                model = row['Model']
                r2 = row.get('R2', np.nan)
                dir_acc = row.get('DirectionAccuracy', np.nan)
                print(f"  {model:20} | R²: {r2:6.3f} | Dir Acc: {dir_acc:6.3f}")
        else:
            print("  No training metrics available")
        
        # Backtest Metrics Summary  
        print(f"\n🎯 BACKTEST METRICS (Unseen Data):")
        print(f"{'-'*40}")
        if not backtest_metrics.empty:
            for _, row in backtest_metrics.iterrows():
                model = row['Model']
                r2 = row.get('R2', np.nan)
                dir_acc = row.get('DirectionAcc', np.nan)
                final_port = row.get('FinalPortfolio', np.nan)
                print(f"  {model:20} | R²: {r2:6.3f} | Dir Acc: {dir_acc:6.3f} | Portfolio: ${final_port:,.0f}")
        else:
            print("  No backtest metrics available")
        
        # Best Model
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"{'='*60}\n")




    def _calculate_advanced_metrics(self, backtest_results: dict, target_scaler=None) -> pd.DataFrame:
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
        """
        Predict next-day price/return for all models and generate an ensemble.
        Returns structured dictionary with individual predictions + ensemble.
        """
        df = self.prepare_features_for_tomorrow_pred(ticker)
        if df is None:
            return {"error": "Could not prepare data"}
        # ============================================================
        # FIX: Load selected features from metadata or selector
        # ============================================================
        meta = self.load_meta(ticker)
        selected_features = meta.get('selected_features', None)
        
        if selected_features:
            feature_cols = [f for f in selected_features if f in df.columns]
            print(f"Using {len(feature_cols)} selected features from metadata")
        else:
            # Fallback: try loading selector
            selector_path = os.path.join(self.config["model_dir"], f"{ticker}_feature_selector.joblib")
            if os.path.exists(selector_path):
                selector = joblib.load(selector_path)
                feature_cols = [f for f in selector.selected_features_ if f in df.columns]
                print(f"Using {len(feature_cols)} selected features from selector")
            else:
                # Last resort: use all features
                feature_cols = [c for c in df.columns if c not in ['target_price', 'target_return', 'target_direction']]
                print(f"Using all {len(feature_cols)} features (no selection found)")
        
        # ============================================================
        # END FIX
        # ============================================================
        
        features = df[feature_cols].values

        # Load scalers
        fs_path = os.path.join(self.config["model_dir"], f"{ticker}_feature_scaler.joblib")
        ts_path = os.path.join(self.config["model_dir"], f"{ticker}_target_scaler.joblib")
        
        if not os.path.exists(fs_path) or not os.path.exists(ts_path):
            return {"error": "Scalers not found. Train models first."}
        
        feature_scaler = joblib.load(fs_path)
        target_scaler = joblib.load(ts_path)
        
        # Verify feature count
        if features.shape[1] != feature_scaler.n_features_in_:
            return {"error": f"Feature mismatch: have {features.shape[1]}, expected {feature_scaler.n_features_in_}"}

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
                
                # Prepare input data based on model type
                if model_type == "LSTM":
                    input_data = last_window
                elif model_type == "Dense NN":
                    input_data = last_window.reshape(1, -1)
                else:  # Tree-based models
                    input_data = last_window.reshape(1, -1)

                # Predict
                if model_type in ["LSTM", "Dense NN"]:
                    pred_scaled = model.predict(input_data, verbose=0).flatten()[0]
                else:
                    pred_scaled = model.predict(input_data).flatten()[0]
                
                # VALIDATE: Check if prediction is reasonable
                if abs(pred_scaled) > 10:  # Unusually large prediction
                    print(f"DEBUG: {model_type} prediction suspicious: {pred_scaled}")
                    pred_scaled = 0  # Fallback to no prediction

                # Inverse scale to returns
                pred_return = target_scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0]
                pred_price = last_price * (1 + pred_return)
                pct_diff = (pred_price - last_price) / last_price * 100
                
                # Convert to float
                pred_price = float(pred_price)
                pred_return = float(pred_return)
                pct_diff = float(pct_diff)
                
                # Determine signal
                signal = "BUY" if pct_diff > threshold else ("SELL" if pct_diff < -threshold else "HOLD")
                
                # REMOVED classifier logic since all your models are regressors
                
                predictions[model_type] = {
                    "predicted_price": pred_price,
                    "predicted_return": pred_return,
                    "pct_diff": pct_diff,
                    "signal": signal
                }
                
            except Exception as e:
                predictions[model_type] = {"error": str(e)}

        # Generate ensemble prediction automatically
        ensemble_pred = self._generate_ensemble_predictions(predictions)
        print("Ensemble raw:", ensemble_pred)
        
        # FIX: Calculate proper pct_diff for ensemble
        if ensemble_pred and "predicted_price" in ensemble_pred:
            ensemble_price = ensemble_pred["predicted_price"]
            ensemble_pred["pct_diff"] = (ensemble_price - last_price) / last_price * 100  # PERCENTAGE difference
        
        if ensemble_pred:
            predictions["Ensemble"] = ensemble_pred

        return predictions



    def _generate_ensemble_predictions(self, model_preds: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Improved ensemble prediction with better logic"""
        signals = []
        prices = []
        pct_diffs = []
        
        # Collect valid predictions
        for model, pred in model_preds.items():
            if "error" in pred:
                continue
                
            signal = pred.get("signal")
            pred_price = pred.get("predicted_price")
            pct_diff = pred.get("pct_diff")
            
            # Skip if missing critical data
            if pred_price is None or pct_diff is None:
                continue
                
            # Only consider models that made a decisive prediction
            if signal and signal != "HOLD":
                signals.append(signal)
                prices.append(pred_price)
                pct_diffs.append(pct_diff)
        
        # If no decisive signals, return HOLD with average price
        if not signals:
            valid_prices = [p.get("predicted_price") for p in model_preds.values() 
                        if "error" not in p and p.get("predicted_price") is not None]
            if valid_prices:
                avg_price = float(np.mean(valid_prices))
                return {
                    "signal": "HOLD", 
                    "predicted_price": avg_price,
                    "confidence": 0,
                    "pct_diff": 0
                }
            else:
                return {"signal": "HOLD", "predicted_price": None, "confidence": 0, "pct_diff": 0}
        
        # Count signals
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        total_votes = len(signals)
        
        # Calculate confidence based on agreement and magnitude
        avg_pct_diff = np.mean(pct_diffs)
        agreement_ratio = max(buy_count, sell_count) / total_votes
        
        # Determine signal based on majority vote
        if buy_count > sell_count:
            ensemble_signal = "BUY"
            confidence = min(agreement_ratio * (1 + abs(avg_pct_diff)/10), 1.0)
        elif sell_count > buy_count:
            ensemble_signal = "SELL" 
            confidence = min(agreement_ratio * (1 + abs(avg_pct_diff)/10), 1.0)
        else:
            ensemble_signal = "HOLD"
            confidence = 0
        
        # Use weighted average price (weighted by confidence)
        ensemble_price = float(np.mean(prices))
        
        return {
            "signal": ensemble_signal,
            "predicted_price": ensemble_price,
            "confidence": confidence,
            "pct_diff": avg_pct_diff  # Use average percentage difference
        }

    # ---------- backtesting ----------
    def walk_forward_backtest(self, df: pd.DataFrame, model: Any, model_type: str,
                  feature_scaler: Any, target_scaler: Any,
                  feature_cols: List[str], ticker: str) -> pd.DataFrame:
        """
        Fixed version with proper feature selection handling
        """
        try:
            # ============================================================
            # FIX: Load selected features from metadata
            # ============================================================
            meta = self.load_meta(ticker)
            selected_features = meta.get('selected_features', feature_cols)
            
            # If selected_features exist, use them; otherwise use provided feature_cols
            if selected_features:
                # Ensure all selected features exist in df
                available_features = [f for f in selected_features if f in df.columns]
                if len(available_features) < len(selected_features):
                    missing = set(selected_features) - set(available_features)
                    logger.warning(f"Missing {len(missing)} features in backtest data: {missing}")
                training_features = available_features
            else:
                training_features = [f for f in feature_cols if f in df.columns]
            
            print(f"DEBUG: Backtest using {len(training_features)} features (scaler expects {feature_scaler.n_features_in_})")
            
            # Verify feature count matches scaler
            if len(training_features) != feature_scaler.n_features_in_:
                logger.error(f"Feature mismatch! Training features: {len(training_features)}, Scaler expects: {feature_scaler.n_features_in_}")
                # Try to load from selector
                selector_path = os.path.join(self.config["model_dir"], f"{ticker}_feature_selector.joblib")
                if os.path.exists(selector_path):
                    selector = joblib.load(selector_path)
                    training_features = [f for f in selector.selected_features_ if f in df.columns]
                    print(f"DEBUG: Loaded features from selector: {len(training_features)} features")
            
            # ============================================================
            # END FIX
            # ============================================================
            
            window = self.config.get("window_size", 30)
            threshold_pct = self.config.get("prediction_threshold_pct", 0.001)

            if len(df) < window + 10:
                logger.warning("Not enough data for backtest")
                return pd.DataFrame(columns=[
                    "Date","TruePrice","PredictedPrice","Signal","PortfolioValue",
                    "PredictedReturn","y_true","y_pred"
                ])

            # --- Prepare features and targets (USE SELECTED FEATURES) ---
            features = df[training_features].values  # 👈 FIX: Use training_features
            true_returns = df['target_return'].values.reshape(-1, 1)
            prices = df['adj_close'].values
            dates = df.index

            scaled_features = feature_scaler.transform(features)
            scaled_returns = target_scaler.transform(true_returns)

            # --- Build sequences ---
            X_seq = np.array([scaled_features[i-window:i] for i in range(window, len(scaled_returns))])
            y_seq = np.array([scaled_returns[i] for i in range(window, len(scaled_returns))]).reshape(-1,1)
            n = X_seq.shape[0]

            if n == 0:
                logger.warning("No sequences created for backtest")
                return pd.DataFrame(columns=[
                    "Date","TruePrice","PredictedPrice","Signal","PortfolioValue",
                    "PredictedReturn","y_true","y_pred"
                ])

            # --- FIX: Define prev_prices properly ---
            prev_prices = prices[window-1:-1]  # Prices at the start of each prediction period
            true_prices = prices[window:]      # Actual prices at prediction time
            
            print(f"DEBUG: {model_type} backtest - prev_prices shape: {prev_prices.shape}, true_prices shape: {true_prices.shape}, X_seq shape: {X_seq.shape}")

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
                

                signals = np.where(pred_returns > 0.002, "BUY", 
                                np.where(pred_returns < -0.002, "SELL", "HOLD"))
                
                print(f"DEBUG: {model_type} Signal distribution - BUY: {(signals == 'BUY').sum()}, "
                    f"SELL: {(signals == 'SELL').sum()}, HOLD: {(signals == 'HOLD').sum()}")

            elif pred_info["kind"] == "classifier":
                cls_preds = pred_info["preds"]
                if len(cls_preds) != n:
                    cls_preds = np.resize(cls_preds, n)
                proba = pred_info.get("proba")
                
                # More sensitive classifier
                hold_mask = (proba is not None) and (np.max(proba, axis=1) < 0.55)
                signals = np.where(cls_preds == 1, "BUY", "SELL")
                if proba is not None:
                    signals[hold_mask] = "HOLD"
                pred_prices = prev_prices

            else:
                logger.warning("Prediction kind unknown")
                pred_returns = np.full(n, np.nan)

            # --- EXTREMELY AGGRESSIVE portfolio simulation ---
            cash = self.config.get("initial_capital", 10000)
            positions = 0
            portfolio = []
            
            trade_count = 0
            for i, sig in enumerate(signals):
                price = true_prices[i]
                
                # Trade on EVERY signal (except HOLD)
                if sig == "BUY" and positions == 0:
                    # Buy with 90% of cash
                    qty = int((cash * 0.9) // price)
                    if qty > 0:
                        positions = qty
                        cash -= qty * price
                        trade_count += 1
                        if trade_count <= 5:
                            print(f"DEBUG: {model_type} BUY {qty} shares at {price:.2f}, cash: {cash:.2f}")
                        
                elif sig == "SELL" and positions > 0:
                    # Sell all positions
                    cash += positions * price
                    trade_count += 1
                    if trade_count <= 5:
                        print(f"DEBUG: {model_type} SELL {positions} shares at {price:.2f}, cash: {cash:.2f}")
                    positions = 0
                    
                portfolio.append(cash + positions * price)
            
            # Close any remaining positions
            if positions > 0:
                cash += positions * prices[-1]
                portfolio[-1] = cash
            
            print(f"DEBUG: {model_type} Total trades: {trade_count}, Final portfolio: ${portfolio[-1]:.2f}")

            # --- Build result DataFrame ---
            df_result = pd.DataFrame({
                "Date": dates[window:],
                "TruePrice": true_prices,
                "PredictedPrice": pred_prices,
                "Signal": signals,
                "PortfolioValue": portfolio,
                "PredictedReturn": pred_returns,
                "y_true": y_true,
                "y_pred": pred_returns
            })

            return df_result

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            import traceback
            traceback.print_exc()
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

