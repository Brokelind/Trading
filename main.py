# main.py
import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from tradingmodelsystem import TradingModelSystem
from news_sentiment import analyze_news_sentiment
import alpaca_trader
import call_market
from distribute_results import *  # This will import the fixed email functions
from alpaca.trading.enums import TimeInForce
from visualize_results import visualize_backtest_chart
from web_dev.web_dashboard import generate_dashboard


# only for local use
try:
    import env
except ImportError:
    env = None

SKIP_TRAINING_ON_CI = os.environ.get("SKIP_TRAINING_ON_CI") or getattr(env, "SKIP_TRAINING_ON_CI", False)

# config
data_path = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
        call_market.get_data(ticker)

        # ensure models trained (or load existing). Respect SKIP on CI.
        
        try:

            self.model_system.analyze_features(ticker, save_plot=False)

            if SKIP_TRAINING_ON_CI:
                print("SKIP_TRAINING_ON_CI set - will NOT retrain; attempt to load existing models.")
                meta = self.model_system.load_meta(ticker)
                if not meta:
                    print("No trained models found; skipping this ticker on CI.")
                    return
                metrics = None
            else:
                res = self.model_system.ensure_trained(ticker, force=True)
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
        
        signal = model_pred.get("signal", "HOLD")
        pct_diff = model_pred.get("pct_diff", 0)


        last_price = None
        try:
            import pandas as pd
            df = pd.read_csv(data_path+"/"+ticker+"_data.csv", index_col=0, parse_dates=True)
            last_price = float(df["adj_close"].iloc[-1])
            if np.isnan(last_price):
                last_price = float(df["adj_close"].iloc[-2])

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

        # Position sizing with volatility and confidence adjustment
        # Current perf_index calculation
        perf_index = min(max(abs(pct_diff) / 5.0, 0.0), 1.0)

        # Incorporate model confidence if available
        if "std_dev" in model_pred:
            confidence = 1.0 - min(model_pred["std_dev"] / (abs(pct_diff) + 1e-6), 1.0)
            perf_index *= max(confidence, 0.2)  # Don't go below 20% instead of 10%

        # Apply sentiment multiplier
        sentiment_multiplier = 1 + min(max(sentiment_effect, -0.5), 0.5)  # boost/dampen ±50%
        perf_index *= sentiment_multiplier

        # Ensure minimum trade size
        MIN_TRADE_FACTOR = 0.1
        perf_index = max(perf_index, MIN_TRADE_FACTOR)

        qty = alpaca_trader.qty_to_trade(
            ticker, 
            signal, 
            perf_index, 
            last_price or 1e-6 , 
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
            "last_price": last_price,
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
            out["training_metrics"] = meta.get("training_metrics", {})
            out["backtest_metrics"] = meta.get("backtest_metrics", {})

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
        

    # Update the post_results method in TradingExecutor class:
    def post_results(self, skip_email=False):
        # Load strong signals summaries
        summaries = load_results()

        if not summaries:
            print("No strong signals meeting criteria today.")
            # Still generate dashboard with no signals message
            generate_dashboard()
            return

        print(f"📈 Found {len(summaries)} strong signals.")
        
        # Generate web dashboard
        signal_count = generate_dashboard()
        print(f"🌐 Generated web dashboard with {signal_count} signals")
        
        # Generate enhanced charts as HTML
        try:
            print("📊 Generating backtest charts...")
            backtest_paths = visualize_backtest_chart([s["ticker"] for s in summaries])
            print(f"✅ Generated {len(backtest_paths)} backtest charts")
            
            
        except Exception as e:
            print(f"Chart generation failed (non-critical): {e}")
        
        # Send minimal email notification with link to dashboard
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5;">
            <div style="max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;">
                <h2 style="color: #333;">📊 Trading Signals Ready</h2>
                <p>Your daily trading analysis is complete with <strong>{len(summaries)}</strong> strong signals.</p>
                <p>View the complete dashboard at:</p>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="file://{os.path.abspath('web_dashboard/index.html')}" 
                    style="background: #007bff; color: white; padding: 15px 30px; 
                            text-decoration: none; border-radius: 5px; font-weight: bold;">
                    Open Dashboard
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
        
        
        if html_body and not skip_email:
            success = send_email(
                f"Daily Trading Summary - {len(summaries)} Strong Signals", 
                html_body
            )
            if success:
                print("✅ Email sent successfully")
            else:
                print("❌ Failed to send email")
        else:
            print("❌ Failed to compose email body")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trading executor")
    parser.add_argument(
        "--debug_ticker",
        type=str,
        help="Run trading only on this ticker (for debugging)"
    )
    parser.add_argument(
        "--skip_email",
        action="store_true",
        help="Skip sending email (for testing)"
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
    trader.post_results(skip_email=args.skip_email)```