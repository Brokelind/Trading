# main.py
import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from analysis import TradingModelSystem
from news_sentiment import analyze_news_sentiment
import alpaca_trader
import call_market
from distribute_results import *  # I'll give a tiny helper below
from alpaca.trading.enums import TimeInForce
from visualize_results import visualize_results, visualize_predictions_chart, visualize_predictions_chart,  visualize_backtest_chart

# config
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Only retrain on CI if explicit flag is false
SKIP_TRAINING_ON_CI = os.environ.get("SKIP_TRAINING_ON_CI", "false").lower() in ("1", "true", "yes")

class TradingExecutor:
    def __init__(self, tickers = None):
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
        self.model_system = TradingModelSystem()

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
                metrics = res.get("metrics", None)
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

        # pick a model (simple pick by best expected pct_diff magnitude and available model)
        best_model = None
        best_abs_diff = 0
        for m, info in preds.items():
            if "pct_diff" in info:
                if abs(info["pct_diff"]) > best_abs_diff:
                    best_abs_diff = abs(info["pct_diff"])
                    best_model = m

        # final signal logic: require model+sentiment alignment or high conviction
        if not best_model:
            print("No best model found")
            return

        model_pred = preds[best_model]
        signal = model_pred.get("signal", "HOLD")
        pct_diff = model_pred.get("pct_diff", 0)
        last_price = None
        # attempt to read last price from data file
        try:
            import pandas as pd
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            last_price = float(df["adj_close"].iloc[-1])
        except Exception:
            pass

        # adjust decision using sentiment
        # simple rules:
        #  - if model is BUY but sentiment strongly negative -> downgrade to HOLD
        #  - if model is SELL but sentiment strongly positive -> downgrade to HOLD
        if signal == "BUY" and sentiment_score < -0.3 and sentiment_conf > 0.5:
            print("Model says BUY but sentiment strongly negative -> override to HOLD")
            signal = "HOLD"
        if signal == "SELL" and sentiment_score > 0.3 and sentiment_conf > 0.5:
            print("Model says SELL but sentiment strongly positive -> override to HOLD")
            signal = "HOLD"

        # size: use your alpaca qty_to_trade, but scale by confidence (abs pct_diff normalized)
        perf_index = min(max(abs(pct_diff) / 5.0, 0.0), 1.0)  # crude normalizer (5% -> 1.0)
        qty = alpaca_trader.qty_to_trade(ticker, signal, perf_index, last_price or 0.0, predicted_diff=pct_diff)
        print(f"Chosen model: {best_model} pred {model_pred} sentiment {sentiment_score:.2f}, qty {qty}")

        # attempt trade (guarded)
        if qty > 0 and signal in ("BUY", "SELL"):
            try:
                # GTC / example stop loss as you used earlier
                alpaca_trader.make_trade(ticker, signal, qty, time_in_force=TimeInForce.GTC, stop_loss_pct=0.03)
                self.current_trades += 1
            except Exception as e:
                print("Trade attempt failed:", e)

        # Save summary result for distribute_results
        out = {
            "ticker": ticker,
            "timestamp": datetime.utcnow().isoformat(),
            "predictions": preds,
            "chosen_model": best_model,
            "signal": signal,
            "qty": qty,
            "sentiment": sentiment,
            "metrics_path": None
        }

        try:
            meta = self.model_system.load_meta(ticker)
            out["metrics_path"] = meta.get("metrics_file")
        except Exception:
            pass

        # write JSON result file to results/
        fname = os.path.join(RESULTS_DIR, f"{ticker}_summary.json")
        with open(fname, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("Wrote result:", fname)

    def run_daily_trading(self):
        print("Starting daily trading run")
        for t in tqdm(self.ticker_list, desc="Processing Tickers"):
            try:
                self.execute_strategy(t)
            except Exception as e:
                print("Error executing", t, e)
        
    def post_results(self):
        # 1. Load strong signals summaries
        summaries = load_results()
        if not summaries:
            print("No strong signals to report today.")
            return

        # 2. Extract tickers from summaries
        strong_tickers = [s["ticker"] for s in summaries]

        # 3. Generate visualization images for those tickers
        backtest_paths = visualize_backtest_chart(strong_tickers)  # returns list of paths or one path
        prediction_paths = visualize_predictions_chart(strong_tickers)  # returns list of paths or one path

        # Combine all images paths into one list
        all_images = []
        if backtest_paths:
            all_images.extend(backtest_paths if isinstance(backtest_paths, list) else [backtest_paths])
        if prediction_paths:
            all_images.extend(prediction_paths if isinstance(prediction_paths, list) else [prediction_paths])

        # 4. Compose email with inline images referencing
        html_body = compose_html_email(
            summaries,
            image_cids=[f"chart{i}" for i in range(len(all_images))]
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
        print(f"Debug mode enabled for ticker: {tickers[0]}")
    else:
        tickers = None  # default to full list

    trader = TradingExecutor(tickers=tickers)

    print("Running script...")
    
    trader.run_daily_trading()
    trader.post_results()