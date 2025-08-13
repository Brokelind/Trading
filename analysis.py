import os
import json
import time
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Any, Union

import numpy as np
import pandas as pd
import joblib
import pandas_ta as ta

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ---- CONFIG ----
MODEL_DIR_DEFAULT = "saved_models"
DATA_DIR_DEFAULT = "data"
MODEL_META_FILENAME = "model_meta.json"
RETRAIN_DAYS = 7  # autoretrain interval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TradingModelSystem")


class TradingModelSystem:
    """
    Train & evaluate multiple models for a ticker:
    - LSTM (sequence -> 1)
    - Dense NN (flat -> 1)
    - Random Forest
    - XGBoost

    Files saved:
      saved_models/<TICKER>_<MODEL>.joblib  (sklearn/xgb)
      saved_models/<TICKER>_<MODEL>.keras   (tf models)
      saved_models/<TICKER>_meta.json      (last_trained, metrics)
      saved_models/<TICKER>_metrics.csv    (per-model metrics summary)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = {
            "data_dir": DATA_DIR_DEFAULT,
            "model_dir": MODEL_DIR_DEFAULT,
            "window_size": 20,
            "prediction_threshold_pct": 0.3,  # pct change for BUY/SELL signals (percent)
            "initial_capital": 10000,
            "min_data_points": 120,
            "retrain_days": RETRAIN_DAYS,
            "verbose": False,
        }
        if config:
            cfg.update(config)
        self.config = cfg

        os.makedirs(self.config["data_dir"], exist_ok=True)
        os.makedirs(self.config["model_dir"], exist_ok=True)

    # ---------- Data / features ----------
    def load_raw(self, ticker: str) -> Optional[pd.DataFrame]:
        path = os.path.join(self.config["data_dir"], f"{ticker}_data.csv")
        if not os.path.exists(path):
            logger.warning(f"No data file for {ticker} at {path}")
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]
        return df

    def prepare_features(self, ticker: str) -> Optional[pd.DataFrame]:
        df = self.load_raw(ticker)
        if df is None:
            return None
        if len(df) < self.config["min_data_points"]:
            logger.warning(f"{ticker} has too few rows ({len(df)})")
            return None

        df = df.copy()
        # base transforms
        df["log_ret"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
        df["volatility"] = df["log_ret"].rolling(21).std() * np.sqrt(252)

        # TA indicators via pandas_ta
        try:
            df["RSI_14"] = ta.rsi(df["adj_close"], length=14)
            macd = ta.macd(df["adj_close"])
            if "MACD_12_26_9" in macd.columns:
                df["MACD"] = macd["MACD_12_26_9"]
            adx = ta.adx(df.get("high"), df.get("low"), df.get("adj_close"))
            if "ADX_14" in adx.columns:
                df["ADX_14"] = adx["ADX_14"]
        except Exception as e:
            logger.debug("pandas_ta failed: %s", e)

        if "volume" in df.columns:
            df["volume_ma10"] = df["volume"].rolling(10).mean()

        # target = next-day price (absolute) and direction
        df["target_price"] = df["adj_close"].shift(-1)
        df["target_return"] = df["target_price"] / df["adj_close"] - 1.0
        df["target_direction"] = np.where(df["target_return"] > 0, 1, 0)

        df = df.dropna()
        return df

    # ---------- model path & metadata ----------
    def _model_path(self, ticker: str, model_type: str) -> str:
        safe = f"{ticker}_{model_type.replace(' ', '_')}"
        ext = ".keras" if model_type in ("LSTM", "Dense NN") else ".joblib"
        return os.path.join(self.config["model_dir"], safe + ext)

    def _meta_path(self, ticker: str) -> str:
        return os.path.join(self.config["model_dir"], f"{ticker}_{MODEL_META_FILENAME}")

    def load_meta(self, ticker: str) -> Dict[str, Any]:
        p = self._meta_path(ticker)
        if os.path.exists(p):
            try:
                return json.load(open(p, "r"))
            except Exception:
                return {}
        return {}

    def save_meta(self, ticker: str, meta: Dict[str, Any]):
        p = self._meta_path(ticker)
        json.dump(meta, open(p, "w"), indent=2, default=str)

    # ---------- training helpers ----------
    def create_sequences(self, arr: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(window, len(arr)):
            X.append(arr[i - window:i, 0])
            y.append(arr[i, 0])
        return np.array(X), np.array(y)

    def train_all_models(self, ticker: str, force: bool = False) -> Dict[str, Any]:
        """
        Train or load existing models for a ticker. Retrains if older than retrain_days or if force.
        Returns dict with model names and training status + metrics.
        """
        df = self.prepare_features(ticker)
        if df is None:
            raise ValueError("Insufficient data")

        meta = self.load_meta(ticker)
        last_trained = meta.get("last_trained")
        needs_retrain = force
        if last_trained:
            try:
                last = datetime.fromisoformat(last_trained)
                if datetime.utcnow() - last > timedelta(days=self.config["retrain_days"]):
                    needs_retrain = True
            except Exception:
                needs_retrain = True
        else:
            needs_retrain = True

        results = {}

        model_types = ["LSTM", "Dense NN", "Random Forest", "XGBoost"]
        # Prepare supervised arrays (predict normalized price)
        data_prices = df[["adj_close"]].values.astype(float)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data_prices)

        window = self.config["window_size"]
        X_all, y_all = self.create_sequences(scaled, window)

        # Split last 20 windows off for validation/backtest
        val_size = min( int(len(X_all) * 0.2), 20 )
        if val_size < 1:
            val_size = 1
        X_train_raw = X_all[:-val_size]
        y_train_raw = y_all[:-val_size]
        X_val_raw = X_all[-val_size:]
        y_val_raw = y_all[-val_size:]

        # reshape for model types
        def reshape_for(model_type, X):
            if model_type == "LSTM":
                return X.reshape(X.shape[0], X.shape[1], 1)
            else:
                return X.reshape(X.shape[0], -1)

        # Train each model (can parallelize if needed)
        for m in model_types:
            model_path = self._model_path(ticker, m)
            trained = False
            try:
                # if model exists and not forcing retrain -> try load
                if os.path.exists(model_path) and not needs_retrain:
                    logger.info(f"Loading existing model {m} for {ticker}")
                    _ = self.load_model(m, ticker)
                    results[m] = {"status": "loaded", "path": model_path}
                    continue

                logger.info(f"Training {m} for {ticker} (force={force}, needs_retrain={needs_retrain})")
                Xtr = reshape_for(m, X_train_raw)
                ytr = y_train_raw

                if m == "LSTM":
                    model = Sequential([
                        Input(shape=(Xtr.shape[1], Xtr.shape[2])),
                        LSTM(64, return_sequences=False),
                        Dropout(0.1),
                        Dense(32, activation="relu"),
                        Dense(1)
                    ])
                    model.compile(optimizer=Adam(1e-3), loss="mse")
                    es = EarlyStopping(monitor="loss", patience=6, restore_best_weights=True)
                    model.fit(Xtr, ytr, epochs=50, batch_size=32, verbose=0, callbacks=[es])
                    save_model(model, model_path)
                elif m == "Dense NN":
                    Xtr2 = Xtr.reshape(Xtr.shape[0], -1)
                    model = Sequential([
                        Input(shape=(Xtr2.shape[1],)),
                        Dense(256, activation="relu"),
                        Dropout(0.2),
                        Dense(64, activation="relu"),
                        Dense(1)
                    ])
                    model.compile(optimizer=Adam(1e-3), loss="mse")
                    model.fit(Xtr2, ytr, epochs=50, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor="loss", patience=6)])
                    save_model(model, model_path)
                elif m == "Random Forest":
                    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
                    rf.fit(Xtr, ytr.ravel())
                    joblib.dump(rf, model_path)
                elif m == "XGBoost":
                    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0)
                    xgb.fit(Xtr, ytr.ravel())
                    joblib.dump(xgb, model_path)
                else:
                    raise RuntimeError("Unknown model type")

                trained = True
                results[m] = {"status": "trained", "path": model_path}
            except Exception as e:
                logger.exception(f"Training failed for {m} {ticker}: {e}")
                results[m] = {"status": f"error: {e}"}

        # After training attempt, run backtests and compute metrics
        backtest_results = {}
        for m in model_types:
            try:
                bt = self.walk_forward_backtest(df.copy(), m, ticker)
                backtest_results[m] = bt
            except Exception as e:
                logger.warning(f"Backtest failed for {m}: {e}")
                backtest_results[m] = pd.DataFrame()

        # Save metrics summary
        metrics_df = self.calculate_metrics(backtest_results)
        metrics_csv = os.path.join(self.config["model_dir"], f"{ticker}_metrics.csv")
        metrics_df.to_csv(metrics_csv)

        # Update meta
        meta = {
            "last_trained": datetime.utcnow().isoformat(),
            "model_paths": {m: self._model_path(ticker, m) for m in model_types},
            "metrics_file": metrics_csv,
            "metrics": metrics_df.to_dict(),
        }
        self.save_meta(ticker, meta)
        logger.info(f"Saved meta for {ticker} to {self._meta_path(ticker)}")
        return {"training_results": results, "backtest": "done", "metrics": metrics_df}

    # ---------- load/save ----------
    def load_model(self, model_type: str, ticker: str):
        path = self._model_path(ticker, model_type)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if model_type in ("LSTM", "Dense NN"):
            return load_model(path, compile=False)
        else:
            return joblib.load(path)

    # ---------- prediction ----------
    def predict_tomorrow(self, ticker: str) -> Dict[str, Dict[str, Any]]:
        """
        Load saved models and predict next day price. Returns per-model predictions + signal.
        """
        df = self.prepare_features(ticker)
        if df is None:
            return {}

        window = self.config["window_size"]
        data = df[["adj_close"]].values.astype(float)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        last_window = scaled[-window:].reshape(1, window, 1)
        flat_last = scaled[-window:].reshape(1, -1)

        preds = {}
        for model_type in ["LSTM", "Dense NN", "Random Forest", "XGBoost"]:
            try:
                model = self.load_model(model_type, ticker)
            except Exception:
                preds[model_type] = {"error": "model missing"}
                continue

            if model_type == "LSTM":
                pred_scaled = model.predict(last_window, verbose=0).flatten()[0]
            elif model_type == "Dense NN":
                pred_scaled = model.predict(flat_last, verbose=0).flatten()[0]
            else:
                pred_scaled = model.predict(flat_last).flatten()[0]

            # inverse scale (single value)
            pred_price = scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0]
            last_price = float(df["adj_close"].iloc[-1])
            pct_diff = (pred_price - last_price) / last_price * 100.0
            signal = "BUY" if pct_diff > self.config["prediction_threshold_pct"] else ("SELL" if pct_diff < -self.config["prediction_threshold_pct"] else "HOLD")
            preds[model_type] = {"predicted_price": float(pred_price), "pct_diff": float(pct_diff), "signal": signal}
        return preds

    # ---------- backtesting ----------
    def walk_forward_backtest(self, df: pd.DataFrame, model_type: str, ticker: str) -> pd.DataFrame:
        """
        Simple walk-forward using saved model or training on rolling windows.
        Returns DataFrame with Date, TruePrice, PredictedPrice, Signal, PortfolioValue.
        """
        window = self.config["window_size"]
        threshold_pct = self.config["prediction_threshold_pct"]

        if len(df) < window + 10:
            logger.warning("Not enough data for backtest")
            return pd.DataFrame()

        prices = df["adj_close"].values.astype(float)
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

        X, y = [], []
        for i in range(window, len(prices)):
            X.append(scaled_prices[i - window:i])
            y.append(scaled_prices[i])
        X = np.array(X)
        y = np.array(y)

        # reshape for models
        if model_type == "LSTM":
            Xm = X.reshape(X.shape[0], X.shape[1], 1)
            # Try load model else train quickly on first part
            try:
                model = self.load_model(model_type, ticker)
            except Exception:
                # fallback: train on first 80% then predict on rest
                # quick train (small epochs)
                split = int(Xm.shape[0] * 0.8)
                model = Sequential([Input(shape=(Xm.shape[1], Xm.shape[2])), LSTM(32), Dense(1)])
                model.compile(optimizer=Adam(1e-3), loss="mse")
                model.fit(Xm[:split], y[:split], epochs=10, batch_size=32, verbose=0)
        else:
            Xm = X.reshape(X.shape[0], -1)
            try:
                model = self.load_model(model_type, ticker)
            except Exception:
                if model_type == "Random Forest":
                    mdl = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    mdl.fit(Xm[: int(Xm.shape[0] * 0.8)], y[: int(y.shape[0] * 0.8)])
                    model = mdl
                else:
                    mdl = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
                    mdl.fit(Xm[: int(Xm.shape[0] * 0.8)], y[: int(y.shape[0] * 0.8)])
                    model = mdl

        # predict all
        if model_type in ("LSTM", "Dense NN"):
            preds_scaled = model.predict(Xm, verbose=0).reshape(-1)
        else:
            preds_scaled = model.predict(Xm).reshape(-1)

        preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        true_prices = prices[window:]

        pct_diffs = (preds - true_prices) / true_prices * 100.0
        signals = np.where(pct_diffs > threshold_pct, "BUY", np.where(pct_diffs < -threshold_pct, "SELL", "HOLD"))

        cash = self.config["initial_capital"]
        positions = 0
        portfolio = []
        for i in range(len(true_prices)):
            price = true_prices[i]
            sig = signals[i]
            if sig == "BUY" and positions == 0:
                positions = cash // price
                cash -= positions * price
            elif sig == "SELL" and positions > 0:
                cash += positions * price
                positions = 0
            portfolio.append(cash + positions * price)

        result = pd.DataFrame({
            "Date": df.index[window:],
            "TruePrice": true_prices,
            "PredictedPrice": preds,
            "Signal": signals,
            "PortfolioValue": portfolio
        })
        return result

    # ---------- metrics ----------
    def calculate_metrics(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        metrics = {}
        for name, df in results.items():
            if df is None or df.empty:
                continue
            returns = df["PortfolioValue"].pct_change().dropna()
            cum_return = (df["PortfolioValue"].iloc[-1] / self.config["initial_capital"] - 1) * 100
            rolling_max = df["PortfolioValue"].cummax()
            drawdown = (df["PortfolioValue"] - rolling_max) / rolling_max
            max_dd = drawdown.min() * 100
            vol = returns.std() * math.sqrt(252) * 100 if len(returns) > 1 else 0.0
            sharpe = returns.mean() / returns.std() * math.sqrt(252) if returns.std() > 0 else 0.0

            pred_metrics = {}
            if set(["TruePrice", "PredictedPrice"]).issubset(df.columns):
                errors = np.abs(df["PredictedPrice"] - df["TruePrice"])
                mae_pct = (errors / df["TruePrice"]).mean() * 100
                dir_acc = (np.sign(df["PredictedPrice"].diff()) == np.sign(df["TruePrice"].diff())).mean() * 100
                pred_metrics = {"MAE (%)": mae_pct, "Direction Accuracy (%)": dir_acc}

            metrics[name] = {
                "Return (%)": cum_return,
                "Max Drawdown (%)": float(max_dd),
                "Volatility (%)": float(vol),
                "Sharpe Ratio": float(sharpe),
                **pred_metrics
            }
        return pd.DataFrame(metrics).T

    # ---------- convenience ----------
    def ensure_trained(self, ticker: str, force: bool = False) -> Dict[str, Any]:
        """
        Public wrapper: trains if needed and returns metrics.
        """
        try:
            res = self.train_all_models(ticker, force=force)
            return res
        except Exception as e:
            logger.exception("ensure_trained failed: %s", e)
            return {"error": str(e)}

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    system = TradingModelSystem()
    t = "AAPL"
    out = system.ensure_trained(t, force=False)
    print("Train output summary keys:", out.keys())
    preds = system.predict_tomorrow(t)
    print("Tomorrow preds:", preds)
