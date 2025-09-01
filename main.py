# main.py
import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from tradingmodelsystem import TradingModelSystem #from tradingmodelsystem import TradingModelSystem  
from news_sentiment import analyze_news_sentiment
import alpaca_trader
import call_market
from distribute_results import *  
from alpaca.trading.enums import TimeInForce
from visualize_results import visualize_results, visualize_predictions_chart, visualize_backtest_chart, visualize_comprehensive

# only for local use
try:
    import env
except ImportError:
    env = None


SKIP_TRAINING_ON_CI = os.environ.get("SKIP_TRAINING_ON_CI", "false").lower() in ("1", "true", "yes") or getattr(env, "SKIP_TRAINING_ON_CI", False)



# config
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Only retrain on CI if explicit flag is false
class TradingExecutor:
    def __init__(self, tickers=None):
        self.ticker_list = tickers or [
            # ETFs
            "SPY", "QQQ", "DIA", "VTI", "IWM",

            # Mega-cap Tech
            "AAPL", "MSFT", "AMZN", "GOOG", "TSLA", "META", "NVDA",

            # Mid/High-growth Tech
            "CRM", "ADBE", "INTU", "SNOW", "PLTR", "UBER",

            # AI/Chip Stocks
            "AMD", "AVGO", "TSM", "QCOM", "SMCI", "ARM",

            # Financials
            "JPM", "BAC", "GS", "MS", "WFC",

            # Energy
            "XOM", "CVX", "SLB", "COP", "PSX",

            # Healthcare
            "UNH", "JNJ", "PFE", "LLY", "MRK", "CVS",

            # Consumer Discretionary
            "HD", "LOW", "NKE", "SBUX", "MCD", "CMG", "COST",

            # Industrials
            "BA", "GE", "CAT", "DE", "LMT", "HON",

            # Utilities
            "NEE", "DUK", "SO", "D", "EXC",

            # Materials
            "LIN", "FCX", "NEM", "APD", "DD"
        ]

        self.max_trades_per_day = 50
        self.current_trades = 0
        # Initialize with custom configuration
        self.model_system = TradingModelSystem({
            "window_size": 30,
            "prediction_threshold_pct": 0.25,
            "enable_uncertainty": True,
            "min_data_points": 200
        })

    def execute_strategy(self, ticker: str):
        print(f"\n=== Processing {ticker} === {datetime.utcnow().isoformat()}")
        # ensure we have data
        data_path = f"data/{ticker}_data.csv"
        if not os.path.exists(data_path):
            print(f"No data for {ticker}, fetching...")
            call_market.get_data(ticker)

        # ensure models trained (or load existing). Respect SKIP on CI.
        try:
            if SKIP_TRAINING_ON_CI:
                print("SKIP_TRAINING_ON_CI set - will NOT retrain; attempt to load existing models.")
                meta = self.model_system.load_meta(ticker)
                if not meta:
                    print("No trained models found; skipping this ticker on CI.")
                    return
                metrics = None
            else:
                res = self.model_system.ensure_trained(ticker, force=False)
                if "error" in res:
                    print("Training failed or insufficient data:", res["error"])
                    return
                metrics = res.get("metrics", {})
        except Exception as e:
            print("Training/ensure failed for", ticker, e)
            return

        # get predictions (from saved models)
        preds = self.model_system.predict_tomorrow(ticker)
        if not preds:
            print("No predictions available for", ticker)
            return

        # get sentiment
        sentiment = analyze_news_sentiment(ticker) or {}
        sentiment_score = sentiment.get("score", 0)
        sentiment_conf = sentiment.get("confidence", 0)

        # Enhanced model selection logic
        best_model = None
        best_score = None
        
        if metrics:
            # Use the best model identified by the system if metrics are available
            best_model = metrics.get('best_model', None)
            
            # Fallback to our own selection if not available
            if not best_model:
                for model in metrics["MAE (%)"].keys():
                    mae = metrics["MAE (%)"][model]
                    accuracy = metrics["Direction Accuracy (%)"][model]
                    sharpe = metrics.get("Sharpe Ratio", {}).get(model, 0)
                    
                    # More sophisticated scoring incorporating multiple metrics
                    score = (accuracy * 0.4) - (mae * 0.3) + (sharpe * 0.3)
                    
                    if best_score is None or score > best_score:
                        best_score = score
                        best_model = model
        else:
            # If no metrics available (CI mode), use the ensemble prediction
            best_model = "Ensemble" if "Ensemble" in preds else next(iter(preds.keys()), None)

        print(f"Selected model: {best_model}")

        if not best_model:
            print("No suitable model found")
            return

        model_pred = preds[best_model]
        
        # Handle both old and new prediction formats
        if "signal" in model_pred:
            # Old format
            signal = model_pred.get("signal", "HOLD")
            pct_diff = model_pred.get("pct_diff", 0)
        else:
            # New ensemble format
            signal = model_pred.get("signal_weighted", "HOLD")
            pct_diff = model_pred.get("pct_diff_weighted", 0)

        last_price = None
        try:
            import pandas as pd
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            last_price = float(df["adj_close"].iloc[-1])
        except Exception:
            pass

        # Enhanced sentiment integration
        sentiment_weight = min(sentiment_conf * 2, 1.0)  # Scale confidence to 0-1 range
        sentiment_effect = sentiment_score * sentiment_weight
        
        # Adjust signal based on sentiment
        if signal == "BUY" and sentiment_effect < -0.3:
            print(f"Model says BUY but negative sentiment (score: {sentiment_score:.2f}, conf: {sentiment_conf:.2f}) -> downgrade to HOLD")
            signal = "HOLD"
        elif signal == "SELL" and sentiment_effect > 0.3:
            print(f"Model says SELL but positive sentiment (score: {sentiment_score:.2f}, conf: {sentiment_conf:.2f}) -> downgrade to HOLD")
            signal = "HOLD"
        elif signal == "HOLD":
            # Consider upgrading to trade if sentiment strongly confirms
            if abs(pct_diff) > 1.5 and ((pct_diff > 0 and sentiment_effect > 0.4) or (pct_diff < 0 and sentiment_effect < -0.4)):
                new_signal = "BUY" if pct_diff > 0 else "SELL"
                print(f"Upgrading HOLD to {new_signal} due to strong signal ({pct_diff:.2f}%) and confirming sentiment ({sentiment_effect:.2f})")
                signal = new_signal

        # Position sizing with volatility adjustment
        perf_index = min(max(abs(pct_diff) / 5.0, 0.0), 1.0)
        
        # Incorporate model confidence if available
        if "std_dev" in model_pred:
            confidence = 1.0 - min(model_pred["std_dev"] / abs(pct_diff), 1.0) if pct_diff != 0 else 0.0
            perf_index *= max(confidence, 0.1)  # Don't go below 10% of original size
            
        qty = alpaca_trader.qty_to_trade(
            ticker, 
            signal, 
            perf_index, 
            last_price or 0.0, 
            predicted_diff=pct_diff
        )
        
        print(f"Final decision: {signal} ({pct_diff:.2f}%), sentiment effect: {sentiment_effect:.2f}, qty: {qty}")

        # Execute trade if conditions met
        if qty > 0 and signal in ("BUY", "SELL") and self.current_trades < self.max_trades_per_day:
            try:
                # Use tighter stop-loss for more volatile predictions
                stop_loss_pct = 0.05 if abs(pct_diff) > 3 else 0.03
                alpaca_trader.make_trade(
                    ticker, 
                    signal, 
                    qty, 
                    time_in_force=TimeInForce.GTC, 
                    stop_loss_pct=stop_loss_pct
                )
                self.current_trades += 1
            except Exception as e:
                print("Trade attempt failed:", e)

        # Save comprehensive results
        out = {
            "ticker": ticker,
            "timestamp": datetime.utcnow().isoformat(),
            "predictions": preds,
            "chosen_model": best_model,
            "signal": signal,
            "pct_diff": pct_diff,
            "qty": qty,
            "sentiment": sentiment,
            "position_size_factor": perf_index,
            "metrics_path": None
        }

        try:
            meta = self.model_system.load_meta(ticker)
            out["metrics_path"] = meta.get("metrics_file")
            out["model_metrics"] = meta.get("metrics", {})
        except Exception:
            pass

        # Write JSON result
        fname = os.path.join(RESULTS_DIR, f"{ticker}_summary.json")
        with open(fname, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("Wrote result:", fname)

    def run_daily_trading(self):
        print("Starting daily trading run")
        for t in tqdm(self.ticker_list, desc="Processing Tickers"):
            try:
                self.execute_strategy(t)
                if self.current_trades >= self.max_trades_per_day:
                    print(f"Reached maximum trades per day ({self.max_trades_per_day})")
                    break
            except Exception as e:
                print("Error executing", t, e)
        
    def post_results(self):
        # 1. Load strong signals summaries
        summaries = load_results()
        

        if not summaries:
            print("No strong signals to report today.")
            
            strong_tickers = self.ticker_list   # fallback to full list
        else:
            strong_tickers = [s["ticker"] for s in summaries]

        print(f"📈 Compiling signal summary for {len(strong_tickers)} tickers... {strong_tickers}")

        # 3. Generate visualization images for those tickers
        backtest_paths = visualize_backtest_chart(strong_tickers)
        prediction_paths = visualize_predictions_chart(strong_tickers)
        comprehensive_paths = visualize_comprehensive(strong_tickers)
        
        # Combine all images paths
        all_images = []
        if backtest_paths:
            all_images.extend(backtest_paths if isinstance(backtest_paths, list) else [backtest_paths])
        if prediction_paths:
            all_images.extend(prediction_paths if isinstance(prediction_paths, list) else [prediction_paths])

        # 4. Compose enhanced email with model performance metrics
        html_body = compose_html_email(
            summaries,
            image_cids=[f"chart{i}" for i in range(len(all_images))],
            #include_model_metrics=True  # Add this parameter to your email composition function
        )

        # 5. Send email with inline images attached
        send_email(
            "Daily Trading Summary - Strong Signals",
            html_body,
            inline_images=all_images
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trading executor")
    parser.add_argument(
        "--debug_ticker",
        type=str,
        help="Run trading only on this ticker (for debugging)"
    )
    args = parser.parse_args()

    if args.debug_ticker:
        tickers = [args.debug_ticker.upper()]
        print(f"Debug mode enabled for ticker: {tickers}")
    else:
        tickers = None  # default to full list

    trader = TradingExecutor(tickers=tickers)
    print("Running script...")
    trader.run_daily_trading()
    trader.post_results()